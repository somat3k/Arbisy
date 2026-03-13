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
from typing import Dict, List, Optional, Tuple

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
        R = np.ones((n, n))           # self-rate = 1
        D = np.empty((n, n), dtype=object)

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
        - L[i,j] = -log(best_rate[i,j])  (or +inf if no direct path)
        - next_hop[i,j] = next token index on shortest i→j path
        """
        n = len(self._tokens)
        R = self._rate_matrix

        # Replace 1.0 self-rates with 0.0 after transform (no movement)
        with np.errstate(divide="ignore", invalid="ignore"):
            L = -np.log(np.where(R > 0, R, np.inf))
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
        """Run Floyd-Warshall in-place; return (dist, next_hop)."""
        n = L.shape[0]
        dist = L.copy()

        for k in range(n):
            new_dist = dist[:, k:k+1] + dist[k:k+1, :]
            improved = new_dist < dist
            dist = np.where(improved, new_dist, dist)
            next_hop = np.where(improved, next_hop[:, k:k+1], next_hop)

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

            # Reconstruct cycle: i → ... → i via next_hop
            cycle_indices = self._reconstruct_path(i, i, next_hop)
            if cycle_indices is None:
                # Fallback: just report the token
                cycle_indices = [i, i]

            key = tuple(sorted(cycle_indices))
            if key in seen:
                continue
            seen.add(key)

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

    @property
    def token_count(self) -> int:
        return len(self._tokens)

    @property
    def tokens(self) -> List[str]:
        return list(self._tokens)
