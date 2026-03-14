"""
NN-driven ArrayBuilder for multi-DEX arbitrage paths.

Builds all possible loan_asset → middle_tokens → loan_asset cycles across
every configured Polygon zkEVM DEX, using DFS path enumeration over the
live exchange-rate graph.  Only paths where the final USD value exceeds
the initial USD value (accounting for the Aave flash-loan premium) are
retained.

Each discovered cycle is represented as a ``PathArray`` and scored by
an optional ``NeuralArbModel``.  Paths are returned sorted by NN score
(when a model is provided) or by profit percentage.

Relationship to existing detectors
------------------------------------
- ``TriangularArbitrage``   — Bellman-Ford; best for large graphs.
- ``ArbitrageMatrix``       — Floyd-Warshall; best for dense grids.
- ``ArrayBuilder``          — DFS enumeration; explicit USD-profit filter,
                              NN scoring, DEX-diversity awareness.
"""

from __future__ import annotations

import math
import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from src.blockchain.dex_price_feed import PoolPrice
from src.ml.feature_engineering import FEATURE_NAMES, N_FEATURES
from src.utils.logger import get_logger

log = get_logger(__name__)

# Aave V3 flash-loan premium in basis points (5 bps = 0.05 %)
_FLASH_PREMIUM_BPS = 5


# ── Internal graph edge ───────────────────────────────────────────────────────

@dataclass
class _EdgeData:
    """A directed edge in the multi-DEX exchange-rate graph."""
    rate: float          # Effective rate after swap fee
    fee_bps: int         # Pool fee in basis points
    liquidity: int       # On-chain liquidity (raw)
    dex_name: str        # Name of the DEX this pool belongs to


# ── Public result type ────────────────────────────────────────────────────────

@dataclass
class PathArray:
    """
    A complete arbitrage cycle as a token-path array.

    ``token_path`` starts and ends with ``loan_asset``, e.g.::

        [USDC, WETH, WMATIC, USDC]

    The ``initial_usd < final_usd`` invariant is guaranteed by the builder.
    """
    loan_asset: str               # Flash-loan asset address (= token_path[0] = token_path[-1])
    token_path: List[str]         # Full path incl. closing token
    dex_path: List[str]           # DEX name for each hop
    rate_path: List[float]        # Effective exchange rate per hop
    fee_path: List[int]           # Fee in bps per hop
    liquidity_path: List[int]     # On-chain liquidity at each hop's pool
    cumulative_rate: float        # Product of hop rates × (1 − flash_premium); > 1 ⇒ profitable
    initial_usd: float            # Flash-loan amount in USD
    final_usd: float              # Expected output USD after all swaps and premium
    profit_usd: float             # final_usd − initial_usd  (always > 0)
    profit_pct: float             # (profit_usd / initial_usd) × 100
    nn_score: float = 0.0         # NeuralArbModel score in [0, 1]

    @property
    def num_hops(self) -> int:
        return len(self.token_path) - 1

    @property
    def unique_dexes(self) -> int:
        return len(set(self.dex_path))

    def __repr__(self) -> str:
        path_str = " → ".join(self.token_path)
        return (
            f"PathArray({path_str} | profit={self.profit_pct:.4f}% "
            f"${self.profit_usd:.2f} nn={self.nn_score:.4f})"
        )


# ── Feature extraction (PathArray → 15-feature vector) ───────────────────────

def path_to_features(
    path: PathArray,
    gas_cost_usd: float = 0.0,
    historical_success_rate: float = 0.5,
    block_utilization_pct: float = 50.0,
) -> np.ndarray:
    """
    Convert a ``PathArray`` to the 15-element feature vector expected by
    both ``ArbitrageModel`` (GBR) and ``NeuralArbModel`` (MLP).

    Parameters
    ----------
    path:                     Discovered arbitrage path.
    gas_cost_usd:             Estimated gas cost in USD (filled in by caller).
    historical_success_rate:  Rolling success rate for this path type.
    block_utilization_pct:    Current block utilisation (0–100).
    """
    num_hops = path.num_hops
    fee_sum  = sum(path.fee_path) if path.fee_path else num_hops * 30

    # Liquidity stats
    liq_vals  = [max(l, 1) for l in path.liquidity_path] if path.liquidity_path else [1]
    min_liq   = min(liq_vals)
    max_liq   = max(liq_vals)
    liq_log   = math.log10(float(min_liq))
    liq_bal   = min_liq / max_liq if max_liq > 0 else 1.0

    # Spread & fee ratios
    total_fee_pct     = fee_sum / 100.0 + _FLASH_PREMIUM_BPS / 100.0
    spread_fee_ratio  = path.profit_pct / total_fee_pct if total_fee_pct > 0 else 0.0
    slippage_pct      = num_hops * 0.1  # conservative 0.1 % per hop

    # Time-of-day cyclic encoding
    ts   = _time.time()
    gmt  = _time.gmtime(ts)
    hour = gmt.tm_hour + gmt.tm_min / 60.0
    tod_sin = math.sin(2 * math.pi * hour / 24.0)
    tod_cos = math.cos(2 * math.pi * hour / 24.0)

    vec = np.array([
        path.profit_pct,                              # price_spread_pct
        path.profit_pct,                              # estimated_profit_pct
        math.log10(max(path.initial_usd, 1.0)),       # loan_amount_usd_log
        gas_cost_usd,                                 # gas_cost_usd
        liq_log,                                      # liquidity_depth_log
        historical_success_rate,                      # historical_success_rate
        slippage_pct,                                 # slippage_estimate_pct
        float(num_hops),                              # num_hops
        1.0 if num_hops == 3 else 0.0,                # is_triangular
        float(fee_sum),                               # fee_bps_sum
        spread_fee_ratio,                             # spread_to_fee_ratio
        liq_bal,                                      # liquidity_balance
        block_utilization_pct,                        # block_utilization_pct
        tod_sin,                                      # time_of_day_sin
        tod_cos,                                      # time_of_day_cos
    ], dtype=np.float64)

    assert len(vec) == N_FEATURES, f"Feature length mismatch: {len(vec)} vs {N_FEATURES}"
    return vec


# ── ArrayBuilder ──────────────────────────────────────────────────────────────

class ArrayBuilder:
    """
    Enumerates all profitable multi-DEX arbitrage path arrays.

    Algorithm
    ---------
    1. Build a directed exchange-rate graph from live pool prices.
       For each token pair the *best effective rate* (post-fee) across all
       pools is stored; the DEX name, fee, and liquidity are also kept.
    2. For every ``loan_asset`` in ``loan_assets``, run DFS from the asset
       back to itself, enumerating cycles with 2–``max_hops`` intermediate
       tokens.
    3. For each discovered cycle, compute the cumulative effective rate
       including the Aave flash-loan premium (0.05 %).  Discard cycles
       where the output USD ≤ input USD.
    4. Score surviving cycles with the provided ``NeuralArbModel``, then
       sort by NN score (or by profit_pct if no model is supplied).

    Parameters
    ----------
    max_hops:         Maximum number of swap hops per cycle (default 4).
    min_profit_pct:   Minimum acceptable profit percentage (default 0.05 %).
    """

    def __init__(
        self,
        max_hops: int = 4,
        min_profit_pct: float = 0.05,
    ) -> None:
        self._max_hops       = max_hops
        self._min_profit_pct = min_profit_pct
        # graph: src_token → {dst_token: _EdgeData}
        self._graph: Dict[str, Dict[str, _EdgeData]] = {}

    # ── Graph construction ────────────────────────────────────────────────────

    def update_prices(self, pool_prices: List[PoolPrice]) -> None:
        """
        Rebuild the exchange-rate graph from current pool price snapshots.

        The effective rate for each hop is the raw price adjusted for the
        pool's swap fee::

            effective_rate = raw_price × (1 − fee_bps / 10_000)

        When multiple pools offer the same token pair, the one with the
        highest effective rate is kept.
        """
        self._graph.clear()
        for pp in pool_prices:
            t0 = pp.token0.lower()
            t1 = pp.token1.lower()
            # Effective rates after deducting pool swap fee
            r01 = pp.price_token1_per_token0 * (1.0 - pp.fee_bps / 10_000)
            r10 = pp.price_token0_per_token1 * (1.0 - pp.fee_bps / 10_000)
            self._add_edge(t0, t1, r01, pp.fee_bps, pp.liquidity, pp.dex_name)
            self._add_edge(t1, t0, r10, pp.fee_bps, pp.liquidity, pp.dex_name)

        log.debug(
            "ArrayBuilder graph: %d tokens, %d directed edges",
            len(self._graph),
            sum(len(v) for v in self._graph.values()),
        )

    def _add_edge(
        self,
        src: str,
        dst: str,
        rate: float,
        fee_bps: int,
        liquidity: int,
        dex_name: str,
    ) -> None:
        if rate <= 0:
            return
        if src not in self._graph:
            self._graph[src] = {}
        existing = self._graph[src].get(dst)
        if existing is None or rate > existing.rate:
            self._graph[src][dst] = _EdgeData(rate, fee_bps, liquidity, dex_name)

    # ── Public API ────────────────────────────────────────────────────────────

    def build_profitable_arrays(
        self,
        loan_assets: List[str],
        loan_amount_usd: float = 10_000.0,
        nn_model: Optional[object] = None,
        gas_cost_usd: float = 0.0,
        historical_success_rate: float = 0.5,
        block_utilization_pct: float = 50.0,
    ) -> List[PathArray]:
        """
        Enumerate all profitable arbitrage cycles for each ``loan_asset``.

        Parameters
        ----------
        loan_assets:              Token addresses to use as the flash-loan
                                  asset (start and end of every cycle).
        loan_amount_usd:          Loan amount in USD for profit computation.
        nn_model:                 Optional ``NeuralArbModel`` instance used to
                                  score surviving paths.  Must expose
                                  ``predict_score(np.ndarray) -> float``.
        gas_cost_usd:             Estimated gas cost passed to feature extractor.
        historical_success_rate:  Passed to feature extractor.
        block_utilization_pct:    Passed to feature extractor.

        Returns
        -------
        Sorted list of ``PathArray`` objects; sorted by ``nn_score`` when a
        model is provided, otherwise by ``profit_pct``.
        """
        results: List[PathArray] = []
        seen: Set[Tuple[str, ...]] = set()

        for asset in loan_assets:
            token = asset.lower()
            if token not in self._graph:
                log.debug("Loan asset %s not in graph — skipping", token)
                continue
            self._dfs(
                start=token,
                current=token,
                path=[token],
                rate_product=1.0,
                rate_path=[],
                dex_path=[],
                fee_path=[],
                liq_path=[],
                loan_amount_usd=loan_amount_usd,
                results=results,
                seen=seen,
            )

        # Score with NN model
        if nn_model is not None:
            for pa in results:
                try:
                    features = path_to_features(
                        pa,
                        gas_cost_usd=gas_cost_usd,
                        historical_success_rate=historical_success_rate,
                        block_utilization_pct=block_utilization_pct,
                    )
                    pa.nn_score = float(nn_model.predict_score(features))  # type: ignore[union-attr]
                except Exception as exc:
                    log.debug("NN scoring error for path %s: %s", pa.token_path, exc)
                    pa.nn_score = 0.0

        # Sort: NN score (desc) when available, otherwise profit_pct (desc)
        if nn_model is not None:
            results.sort(key=lambda p: p.nn_score, reverse=True)
        else:
            results.sort(key=lambda p: p.profit_pct, reverse=True)

        log.info(
            "ArrayBuilder: %d profitable paths found across %d loan assets",
            len(results), len(loan_assets),
        )
        return results

    # ── DFS path enumeration ──────────────────────────────────────────────────

    def _dfs(
        self,
        start: str,
        current: str,
        path: List[str],
        rate_product: float,
        rate_path: List[float],
        dex_path: List[str],
        fee_path: List[int],
        liq_path: List[int],
        loan_amount_usd: float,
        results: List[PathArray],
        seen: Set[Tuple[str, ...]],
    ) -> None:
        # Check if we have closed the cycle (≥ 1 intermediate token)
        if len(path) > 2 and current == start:
            # Apply flash-loan premium (5 bps = 0.05%) to the cumulative rate
            effective_rate = rate_product * (1.0 - _FLASH_PREMIUM_BPS / 10_000)
            profit_pct     = (effective_rate - 1.0) * 100.0
            if profit_pct >= self._min_profit_pct:
                # Canonical key: rotate inner tokens to min, preserve direction
                inner    = path[1:-1]   # strip opening and closing asset
                min_pos  = min(range(len(inner)), key=lambda i: inner[i])
                canon    = tuple(inner[min_pos:] + inner[:min_pos] + [start])
                if canon not in seen:
                    seen.add(canon)
                    final_usd = loan_amount_usd * effective_rate
                    results.append(PathArray(
                        loan_asset=start,
                        token_path=list(path),
                        dex_path=list(dex_path),
                        rate_path=list(rate_path),
                        fee_path=list(fee_path),
                        liquidity_path=list(liq_path),
                        cumulative_rate=effective_rate,
                        initial_usd=loan_amount_usd,
                        final_usd=final_usd,
                        profit_usd=final_usd - loan_amount_usd,
                        profit_pct=profit_pct,
                    ))
            return

        # Prune: too many hops
        if len(path) > self._max_hops + 1:
            return

        # Explore neighbours
        for neighbour, edge in self._graph.get(current, {}).items():
            # Allow returning to start only if we already have ≥ 2 tokens
            if neighbour == start:
                if len(path) < 3:
                    continue   # need at least one intermediate token
            elif neighbour in path:
                continue       # don't revisit intermediate tokens

            self._dfs(
                start=start,
                current=neighbour,
                path=path + [neighbour],
                rate_product=rate_product * edge.rate,
                rate_path=rate_path + [edge.rate],
                dex_path=dex_path + [edge.dex_name],
                fee_path=fee_path + [edge.fee_bps],
                liq_path=liq_path + [edge.liquidity],
                loan_amount_usd=loan_amount_usd,
                results=results,
                seen=seen,
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def token_count(self) -> int:
        return len(self._graph)

    @property
    def tokens(self) -> List[str]:
        return list(self._graph.keys())
