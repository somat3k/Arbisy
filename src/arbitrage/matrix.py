"""
Matrix/array-based arbitrage path finder.

Represents exchange rates as an (n × n) numpy matrix R where R[i,j] is the
exchange rate from token i to token j.  By log-transforming the matrix and
applying shortest-path algorithms (Floyd-Warshall or Bellman-Ford), we can
efficiently find all n-hop arbitrage paths.

This is complementary to the graph-based TriangularArbitrage: it is faster
for dense token grids and handles multi-hop paths naturally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.blockchain.dex_price_feed import PoolPrice
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class MatrixArbPath:
    """An arbitrage path discovered via matrix analysis."""
    token_indices: List[int]
    token_names: List[str]
    log_profit: float          # Σ log(rates) along the cycle; positive = profitable
    profit_multiplier: float   # e^log_profit; >1 means profitable
    profit_pct: float

    def __repr__(self) -> str:
        path = " → ".join(self.token_names)
        return f"MatrixArbPath({path} | {self.profit_pct:.4f}%)"


class ArbitrageMatrix:
    """
    Numpy-based arbitrage path detector.

    Algorithm
    ---------
    1. Build rate matrix R[i,j] = best exchange rate token_i → token_j.
    2. Apply log transform: L[i,j] = -log(R[i,j]).
    3. Run Floyd-Warshall shortest-path on L.
    4. Negative diagonal entries (L[i,i] < 0) reveal profitable cycles.
    5. Reconstruct the cycle path via the predecessor matrix.
    """

    def __init__(self) -> None:
        self._tokens: List[str] = []
        self._token_index: Dict[str, int] = {}
        self._rate_matrix: Optional[np.ndarray] = None
        self._dex_matrix: Optional[np.ndarray] = None   # dex name per cell (object array)

    # ── Matrix construction ───────────────────────────────────────────────────

    def build_from_prices(self, pool_prices: List[PoolPrice]) -> None:
        """
        Construct the rate matrix from pool price snapshots.
        Call this before running find_opportunities().
        """
        # Collect all unique tokens
        token_set: Dict[str, None] = {}
        for pp in pool_prices:
            token_set[pp.token0.lower()] = None
            token_set[pp.token1.lower()] = None
        self._tokens = list(token_set.keys())
        self._token_index = {t: i for i, t in enumerate(self._tokens)}

        n = len(self._tokens)
        # Off-diagonal initialised to 0 (= no edge).  Only the diagonal is 1
        # (self-rate).  Using 1 for off-diagonal would create artificial free
        # edges between every token pair, generating false arbitrage cycles.
        R = np.zeros((n, n))
        np.fill_diagonal(R, 1.0)
        # Use None to mark cells with no known DEX; np.empty leaves garbage values.
        D = np.full((n, n), None, dtype=object)

        for pp in pool_prices:
            i = self._token_index[pp.token0.lower()]
            j = self._token_index[pp.token1.lower()]
            r_ij = pp.price_token1_per_token0
            r_ji = pp.price_token0_per_token1
            # Keep best rate
            if r_ij > R[i, j]:
                R[i, j] = r_ij
                D[i, j] = pp.dex_name
            if r_ji > R[j, i]:
                R[j, i] = r_ji
                D[j, i] = pp.dex_name

        self._rate_matrix = R
        self._dex_matrix  = D
        log.debug("Rate matrix built: %d×%d", n, n)

    # ── Floyd-Warshall on log-transformed matrix ──────────────────────────────

    def _build_log_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (L, next_hop) where:
        - L[i,j] = -log(best_rate[i,j])  (or +inf if no direct edge)
        - next_hop[i,j] = next token index on shortest i→j path

        Off-diagonal entries in R are 0 when no pool connects the pair.
        We must produce +inf (not -inf) for those cells so Floyd-Warshall
        does not treat absent edges as free paths.
        """
        n = len(self._tokens)
        R = self._rate_matrix

        # Build L: start with +inf everywhere, then fill in -log(rate) only
        # where a real pool edge exists (R > 0 and off-diagonal).
        L = np.full((n, n), np.inf)
        mask = (R > 0) & (~np.eye(n, dtype=bool))
        with np.errstate(divide="ignore", invalid="ignore"):
            L[mask] = -np.log(R[mask])
        np.fill_diagonal(L, 0.0)

        next_hop = np.full((n, n), -1, dtype=int)
        for i in range(n):
            for j in range(n):
                if i != j and not np.isinf(L[i, j]):
                    next_hop[i, j] = j

        return L, next_hop

    def _floyd_warshall(
        self,
        L: np.ndarray,
        next_hop: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Floyd-Warshall in-place; return (dist, next_hop).

        Two separate improvement conditions are used:
        - ``dist_improved`` (strict ``<``) drives distance relaxation so that
          even tiny improvements accumulate correctly.
        - ``path_improved`` uses a small epsilon to avoid floating-point noise
          (e.g. ``-log(1.1) + log(1.1)`` ≈ -1.5e-17 instead of 0) from
          incorrectly updating the predecessor matrix and corrupting paths.
        """
        _PATH_EPS = 1e-12  # minimum meaningful improvement for path tracking

        n = L.shape[0]
        dist = L.copy()

        for k in range(n):
            new_dist = dist[:, k:k+1] + dist[k:k+1, :]
            dist_improved = new_dist < dist
            path_improved = new_dist < dist - _PATH_EPS
            dist = np.where(dist_improved, new_dist, dist)
            # Capture column k *before* reassignment to avoid aliasing issues
            col_k = next_hop[:, k:k+1].copy()
            next_hop = np.where(path_improved, col_k, next_hop)

        return dist, next_hop

    # ── Cycle reconstruction ──────────────────────────────────────────────────

    def _reconstruct_path(
        self,
        start: int,
        end: int,
        next_hop: np.ndarray,
        max_len: int = 10,
    ) -> Optional[List[int]]:
        """Reconstruct path from start to end via next_hop matrix."""
        if next_hop[start, end] == -1:
            return None
        path = [start]
        current = start
        for _ in range(max_len):
            current = next_hop[current, end]
            if current == -1:
                return None
            path.append(current)
            if current == end:
                break
        return path if path[-1] == end else None

    def _trace_negative_cycle(
        self,
        i: int,
        dist: np.ndarray,
        next_hop: np.ndarray,
    ) -> Optional[List[int]]:
        """
        Reconstruct a negative-weight cycle passing through node ``i``.

        Because ``next_hop[i, i]`` is always -1 (the diagonal is never set),
        ``_reconstruct_path(i, i, ...)`` returns None immediately.  Instead we
        find the intermediate node ``k`` that gives the most negative
        ``dist[i,k] + dist[k,i]``, reconstruct both halves, and join them.
        """
        n = len(self._tokens)
        best_k = -1
        best_w = 0.0

        for k in range(n):
            if k == i:
                continue
            if next_hop[i, k] == -1 or next_hop[k, i] == -1:
                continue
            w = float(dist[i, k]) + float(dist[k, i])
            if w < best_w:
                best_w = w
                best_k = k

        if best_k == -1:
            return None

        path_to_k = self._reconstruct_path(i, best_k, next_hop)
        path_back = self._reconstruct_path(best_k, i, next_hop)

        if path_to_k is None or path_back is None:
            return None

        # Combine: i → ... → k → ... → i (avoid duplicating k at the join)
        return path_to_k + path_back[1:]

    # ── Public API ────────────────────────────────────────────────────────────

    def find_opportunities(
        self,
        min_profit_pct: float = 0.05,
    ) -> List[MatrixArbPath]:
        """
        Run Floyd-Warshall and return all profitable cycles.

        A cycle exists whenever dist[i,i] < 0 after Floyd-Warshall.
        """
        if self._rate_matrix is None or len(self._tokens) < 3:
            log.debug("Matrix not built or too few tokens — skipping")
            return []

        L, next_hop = self._build_log_matrix()
        dist, next_hop = self._floyd_warshall(L, next_hop)

        n = len(self._tokens)
        results: List[MatrixArbPath] = []
        seen: set = set()

        for i in range(n):
            neg_log = dist[i, i]
            if neg_log >= -1e-8:
                continue

            log_profit = -neg_log            # positive value
            multiplier = math.exp(log_profit)
            profit_pct = (multiplier - 1.0) * 100.0

            if profit_pct < min_profit_pct:
                continue

            # Reconstruct cycle: i → ... → i via dedicated cycle tracer.
            # _reconstruct_path(i, i, ...) always fails because next_hop[i,i]=-1.
            cycle_indices = self._trace_negative_cycle(i, dist, next_hop)
            if cycle_indices is None:
                log.debug("Could not reconstruct cycle for token %d — skipping", i)
                continue

            # Canonical key: rotate inner path to the smallest index,
            # keeping direction (not sorting), to avoid collapsing distinct cycles.
            inner = cycle_indices[:-1]  # drop the closing duplicate of start
            if not inner:
                continue
            min_pos = min(range(len(inner)), key=lambda idx: inner[idx])
            canonical = tuple(inner[min_pos:] + inner[:min_pos])
            if canonical in seen:
                continue
            seen.add(canonical)

            token_names = [self._tokens[idx] for idx in cycle_indices]
            results.append(MatrixArbPath(
                token_indices=cycle_indices,
                token_names=token_names,
                log_profit=log_profit,
                profit_multiplier=multiplier,
                profit_pct=profit_pct,
            ))
            log.info("Matrix arb found: %s", results[-1])

        results.sort(key=lambda p: p.profit_pct, reverse=True)
        return results

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_rate(self, token_a: str, token_b: str) -> float:
        """Direct rate lookup from the matrix."""
        i = self._token_index.get(token_a.lower())
        j = self._token_index.get(token_b.lower())
        if i is None or j is None or self._rate_matrix is None:
            return 0.0
        return float(self._rate_matrix[i, j])

    def get_hop_dex(self, from_idx: int, to_idx: int) -> Optional[str]:
        """Return the DEX name used for the best rate from token[from_idx] to token[to_idx]."""
        if self._dex_matrix is None:
            return None
        try:
            return self._dex_matrix[from_idx, to_idx]
        except Exception:
            return None

    def build_hop_list(self, path: "MatrixArbPath") -> List[Dict[str, Any]]:
        """
        Convert a MatrixArbPath into a list of swap-hop dicts for the
        execution pipeline.

        Each dict has keys: token_in, token_out, dex_name, router_address,
        fee_bps, is_v3, estimated_amount_out.

        Parameters
        ----------
        path: MatrixArbPath returned by find_opportunities().

        Returns
        -------
        List of hop dicts, one per edge in the cycle (last returns to start).
        """
        hops: List[Dict[str, Any]] = []
        indices = path.token_indices
        for k in range(len(indices) - 1):
            from_idx = indices[k]
            to_idx   = indices[k + 1]
            token_in  = self._tokens[from_idx]
            token_out = self._tokens[to_idx]
            dex_name  = self.get_hop_dex(from_idx, to_idx) or "unknown"
            rate = float(self._rate_matrix[from_idx, to_idx]) if self._rate_matrix is not None else 0.0
            hops.append({
                "token_in":             token_in,
                "token_out":            token_out,
                "dex_name":             dex_name,
                # "0x0" is a sentinel meaning "resolve router from dex_name at execution time".
                # ExecutionAgent / ArbitrageAgent replaces this with the real router address.
                "router_address":       "0x0",
                "fee_bps":              30,      # default 0.30%; overridden by DEX-specific logic
                "is_v3":                True,
                "estimated_amount_out": 0.0,     # no per-hop estimate available from matrix
            })
        return hops

    @property
    def token_count(self) -> int:
        return len(self._tokens)

    @property
    def tokens(self) -> List[str]:
        return list(self._tokens)
