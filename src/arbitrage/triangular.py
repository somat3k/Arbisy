"""
Triangular arbitrage detection using Bellman-Ford.

Builds a directed weighted graph of token pairs from live DEX prices.
Applies Bellman-Ford on log-transformed edge weights to detect negative
cycles, which correspond to profitable triangular arbitrage paths.

Complexity: O(V × E) per scan — suitable for up to ~50 tokens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from src.blockchain.dex_price_feed import PoolPrice
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class ArbPath:
    """A sequence of token swaps forming a complete cycle."""
    tokens: List[str]          # e.g. [USDC, WETH, WMATIC, USDC]
    dexes:  List[str]          # DEX used for each hop
    rates:  List[float]        # Exchange rate for each hop
    cumulative_product: float  # Product of all rates; >1.0 means profitable
    estimated_profit_pct: float

    @property
    def hops(self) -> int:
        return len(self.tokens) - 1

    def __repr__(self) -> str:
        path_str = " → ".join(self.tokens)
        return f"ArbPath({path_str} | profit={self.estimated_profit_pct:.4f}%)"


@dataclass
class GraphEdge:
    """A directed edge in the token exchange-rate graph."""
    source: str
    target: str
    rate: float        # How many target tokens per 1 source token
    dex_name: str
    log_neg_rate: float = field(init=False)

    def __post_init__(self) -> None:
        # For Bellman-Ford: we look for negative-weight cycles.
        # Edge weight = -log(rate); negative cycle ⟺ Σ(-log(r)) < 0 ⟺ Π(r) > 1
        self.log_neg_rate = -math.log(self.rate) if self.rate > 0 else float("inf")


class TriangularArbitrage:
    """
    Detects triangular (and longer) arbitrage cycles in a token graph.

    Usage
    -----
    detector = TriangularArbitrage()
    detector.update_prices(pool_prices)
    paths = detector.find_opportunities(min_profit_pct=0.1)
    """

    def __init__(self, max_path_length: int = 4) -> None:
        self.max_path_length = max_path_length
        # Graph: source → {target: best_edge}
        self._graph: Dict[str, Dict[str, GraphEdge]] = {}
        self._all_tokens: Set[str] = set()

    # ── Graph construction ────────────────────────────────────────────────────

    def update_prices(self, pool_prices: List[PoolPrice]) -> None:
        """Rebuild the exchange-rate graph from fresh pool prices."""
        self._graph.clear()
        self._all_tokens.clear()

        for pp in pool_prices:
            self._add_edge(
                pp.token0, pp.token1,
                pp.price_token1_per_token0, pp.dex_name
            )
            self._add_edge(
                pp.token1, pp.token0,
                pp.price_token0_per_token1, pp.dex_name
            )

        log.debug(
            "Graph built: %d tokens, %d directed edges",
            len(self._all_tokens),
            sum(len(v) for v in self._graph.values()),
        )

    def _add_edge(
        self, src: str, dst: str, rate: float, dex: str
    ) -> None:
        if rate <= 0:
            return
        src = src.lower()
        dst = dst.lower()
        self._all_tokens.update([src, dst])
        if src not in self._graph:
            self._graph[src] = {}
        existing = self._graph[src].get(dst)
        new_edge  = GraphEdge(src, dst, rate, dex)
        # Keep the best (highest) rate for each src→dst pair
        if existing is None or new_edge.rate > existing.rate:
            self._graph[src][dst] = new_edge

    # ── Bellman-Ford negative cycle detection ─────────────────────────────────

    def _bellman_ford(self, source: str) -> Optional[List[str]]:
        """
        Run Bellman-Ford from `source`.  Return a list of tokens forming
        a negative cycle if one is reachable, or None.
        """
        tokens = list(self._all_tokens)
        n = len(tokens)
        idx = {t: i for i, t in enumerate(tokens)}

        INF = float("inf")
        dist = [INF] * n
        pred: List[Optional[int]] = [None] * n
        pred_edge: List[Optional[GraphEdge]] = [None] * n

        dist[idx.get(source, 0)] = 0.0

        for _ in range(n - 1):
            for src, targets in self._graph.items():
                u = idx.get(src)
                if u is None or dist[u] == INF:
                    continue
                for dst, edge in targets.items():
                    v = idx.get(dst)
                    if v is None:
                        continue
                    new_dist = dist[u] + edge.log_neg_rate
                    if new_dist < dist[v]:
                        dist[v] = new_dist
                        pred[v] = u
                        pred_edge[v] = edge

        # Check for negative-weight cycle
        for src, targets in self._graph.items():
            u = idx.get(src)
            if u is None or dist[u] == INF:
                continue
            for dst, edge in targets.items():
                v = idx.get(dst)
                if v is None:
                    continue
                if dist[u] + edge.log_neg_rate < dist[v] - 1e-10:
                    # Negative cycle detected — reconstruct it
                    return self._reconstruct_cycle(v, pred, tokens, n)
        return None

    @staticmethod
    def _reconstruct_cycle(
        start: int,
        pred: List[Optional[int]],
        tokens: List[str],
        n: int,
    ) -> List[str]:
        """Walk backwards through predecessors to find the cycle."""
        visited = [False] * n
        node = start
        for _ in range(n):
            node = pred[node]  # type: ignore[assignment]
        # node is now inside the cycle
        cycle_node = node
        cycle: List[str] = []
        current = cycle_node
        while True:
            cycle.append(tokens[current])
            current = pred[current]  # type: ignore[assignment]
            if current == cycle_node:
                break
        cycle.append(tokens[cycle_node])
        cycle.reverse()
        return cycle

    # ── Public API ────────────────────────────────────────────────────────────

    def find_opportunities(
        self,
        min_profit_pct: float = 0.05,
        start_tokens: Optional[List[str]] = None,
    ) -> List[ArbPath]:
        """
        Scan the graph for profitable cycles.

        Parameters
        ----------
        min_profit_pct:  Minimum profit percentage (e.g. 0.1 = 0.1%).
        start_tokens:    Limit search to cycles starting from these tokens.
                         Defaults to all tokens.
        """
        if not self._graph:
            return []

        candidates: List[ArbPath] = []
        seen_cycles: Set[str] = set()

        search_tokens = (
            [t.lower() for t in start_tokens]
            if start_tokens
            else list(self._all_tokens)
        )

        for token in search_tokens:
            cycle = self._bellman_ford(token)
            if cycle is None:
                continue

            key = "→".join(sorted(cycle))
            if key in seen_cycles:
                continue
            seen_cycles.add(key)

            path = self._build_arb_path(cycle)
            if path and path.estimated_profit_pct >= min_profit_pct:
                candidates.append(path)
                log.info("Triangular arb found: %s", path)

        candidates.sort(key=lambda p: p.estimated_profit_pct, reverse=True)
        return candidates

    def _build_arb_path(self, cycle: List[str]) -> Optional[ArbPath]:
        """Convert a raw cycle token list into an ArbPath."""
        rates: List[float] = []
        dexes: List[str]   = []

        for i in range(len(cycle) - 1):
            src = cycle[i]
            dst = cycle[i + 1]
            edge = self._graph.get(src, {}).get(dst)
            if edge is None:
                return None
            rates.append(edge.rate)
            dexes.append(edge.dex_name)

        product = 1.0
        for r in rates:
            product *= r

        profit_pct = (product - 1.0) * 100.0
        return ArbPath(
            tokens=cycle,
            dexes=dexes,
            rates=rates,
            cumulative_product=product,
            estimated_profit_pct=profit_pct,
        )

    # ── DFS brute-force for short paths (fallback) ─────────────────────────────

    def find_cycles_dfs(
        self,
        start_token: str,
        min_profit_pct: float = 0.05,
    ) -> List[ArbPath]:
        """
        DFS-based cycle enumeration up to max_path_length hops.
        More exhaustive but slower than Bellman-Ford for large graphs.
        """
        start = start_token.lower()
        results: List[ArbPath] = []

        def dfs(
            current: str,
            path_tokens: List[str],
            path_rates: List[float],
            path_dexes: List[str],
            product: float,
        ) -> None:
            if len(path_tokens) > self.max_path_length + 1:
                return
            if len(path_tokens) > 2 and current == start:
                profit_pct = (product - 1.0) * 100.0
                if profit_pct >= min_profit_pct:
                    results.append(ArbPath(
                        tokens=list(path_tokens),
                        dexes=list(path_dexes),
                        rates=list(path_rates),
                        cumulative_product=product,
                        estimated_profit_pct=profit_pct,
                    ))
                return

            for neighbor, edge in self._graph.get(current, {}).items():
                if neighbor in path_tokens[1:]:
                    continue
                if neighbor != start and len(path_tokens) >= self.max_path_length:
                    continue
                dfs(
                    neighbor,
                    path_tokens + [neighbor],
                    path_rates + [edge.rate],
                    path_dexes + [edge.dex_name],
                    product * edge.rate,
                )

        dfs(start, [start], [], [], 1.0)
        results.sort(key=lambda p: p.estimated_profit_pct, reverse=True)
        return results
