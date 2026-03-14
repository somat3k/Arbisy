"""
Cross-platform (cross-DEX) arbitrage detection.

Compares prices for the same token pair across multiple DEXes.
When the spread between the best bid on one DEX and best ask on
another exceeds fees + gas, we have a profitable opportunity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.blockchain.dex_price_feed import PoolPrice
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class CrossPlatformOpportunity:
    """Represents a two-leg cross-DEX arbitrage."""
    token_a: str
    token_b: str
    buy_dex: str
    sell_dex: str
    buy_price: float     # price of token_a in token_b on buy_dex
    sell_price: float    # price of token_a in token_b on sell_dex
    spread_pct: float    # (sell - buy) / buy * 100
    estimated_profit_pct: float   # after fees
    buy_pool:  str
    sell_pool: str
    buy_fee_bps:  int
    sell_fee_bps: int
    liquidity_buy:  int
    liquidity_sell: int
    timestamp: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        pair = f"{self.token_a[:6]}/{self.token_b[:6]}"
        return (
            f"CrossArb({pair} | buy={self.buy_dex}@{self.buy_price:.6f} "
            f"sell={self.sell_dex}@{self.sell_price:.6f} "
            f"profit={self.estimated_profit_pct:.4f}%)"
        )


class CrossPlatformArbitrage:
    """
    Detects price discrepancies for the same token pair across DEXes.

    Usage
    -----
    detector = CrossPlatformArbitrage()
    detector.update_prices(pool_prices)
    opps = detector.find_opportunities(min_spread_pct=0.1)
    """

    FLASH_PREMIUM_PCT = 0.05   # Aave V3 flash loan fee

    def __init__(self) -> None:
        # (token0, token1) → {dex_name: PoolPrice}
        self._prices: Dict[Tuple[str, str], Dict[str, PoolPrice]] = {}

    # ── Price ingestion ───────────────────────────────────────────────────────

    def update_prices(self, pool_prices: List[PoolPrice]) -> None:
        """Ingest fresh pool price snapshots."""
        self._prices.clear()
        for pp in pool_prices:
            key = self._pair_key(pp.token0, pp.token1)
            if key not in self._prices:
                self._prices[key] = {}
            self._prices[key][pp.dex_name] = pp

        log.debug(
            "CrossPlatform: %d token pairs across %d DEX snapshots",
            len(self._prices),
            len(pool_prices),
        )

    @staticmethod
    def _pair_key(token_a: str, token_b: str) -> Tuple[str, str]:
        """Canonical (sorted) pair key so token order doesn't matter."""
        a, b = token_a.lower(), token_b.lower()
        return (a, b) if a < b else (b, a)

    # ── Opportunity detection ─────────────────────────────────────────────────

    def find_opportunities(
        self,
        min_spread_pct: float = 0.1,
    ) -> List[CrossPlatformOpportunity]:
        """
        Find all cross-DEX opportunities where spread > min_spread_pct.

        The spread is computed after subtracting:
        - Both DEX trading fees
        - Aave flash loan premium (0.05%)

        Returns opportunities sorted by descending estimated profit.
        """
        results: List[CrossPlatformOpportunity] = []

        for (t0, t1), dex_map in self._prices.items():
            if len(dex_map) < 2:
                continue

            dex_list = list(dex_map.values())
            for i in range(len(dex_list)):
                for j in range(i + 1, len(dex_list)):
                    opp = self._compare_pair(t0, t1, dex_list[i], dex_list[j], min_spread_pct)
                    if opp:
                        results.append(opp)

        results.sort(key=lambda o: o.estimated_profit_pct, reverse=True)
        log.debug("CrossPlatform found %d opportunities", len(results))
        return results

    def _compare_pair(
        self,
        t0: str,
        t1: str,
        pp_a: PoolPrice,
        pp_b: PoolPrice,
        min_spread_pct: float,
    ) -> Optional[CrossPlatformOpportunity]:
        """
        Compare two pool prices for the same pair.
        Returns an opportunity if the spread (after fees) meets the threshold.
        """
        # Normalise both prices to the same direction: token1 per token0
        price_a = self._normalised_price(pp_a, t0)
        price_b = self._normalised_price(pp_b, t0)

        if price_a <= 0 or price_b <= 0:
            return None

        # Determine which DEX is cheaper (buy) and which is more expensive (sell)
        if price_a < price_b:
            buy_pp, sell_pp = pp_a, pp_b
            buy_price, sell_price = price_a, price_b
        else:
            buy_pp, sell_pp = pp_b, pp_a
            buy_price, sell_price = price_b, price_a

        raw_spread_pct = (sell_price - buy_price) / buy_price * 100

        # Total fees: buy_fee + sell_fee + flash_premium (all in %)
        fee_pct = (
            (buy_pp.fee_bps / 100)
            + (sell_pp.fee_bps / 100)
            + self.FLASH_PREMIUM_PCT
        )
        net_profit_pct = raw_spread_pct - fee_pct

        if net_profit_pct < min_spread_pct:
            return None

        return CrossPlatformOpportunity(
            token_a=t0,
            token_b=t1,
            buy_dex=buy_pp.dex_name,
            sell_dex=sell_pp.dex_name,
            buy_price=buy_price,
            sell_price=sell_price,
            spread_pct=raw_spread_pct,
            estimated_profit_pct=net_profit_pct,
            buy_pool=buy_pp.pool_address,
            sell_pool=sell_pp.pool_address,
            buy_fee_bps=buy_pp.fee_bps,
            sell_fee_bps=sell_pp.fee_bps,
            liquidity_buy=buy_pp.liquidity,
            liquidity_sell=sell_pp.liquidity,
        )

    @staticmethod
    def _normalised_price(pp: PoolPrice, base_token: str) -> float:
        """Return price such that we get quote tokens per 1 base token."""
        if pp.token0.lower() == base_token.lower():
            return pp.price_token1_per_token0
        return pp.price_token0_per_token1

    # ── Liquidity filter ──────────────────────────────────────────────────────

    def filter_by_liquidity(
        self,
        opportunities: List[CrossPlatformOpportunity],
        min_liquidity: int = 10_000,
    ) -> List[CrossPlatformOpportunity]:
        """Remove opportunities where either pool has insufficient liquidity."""
        return [
            o for o in opportunities
            if o.liquidity_buy >= min_liquidity
            and o.liquidity_sell >= min_liquidity
        ]
