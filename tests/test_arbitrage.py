"""
Tests for arbitrage detection modules.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from src.arbitrage.cross_platform import CrossPlatformArbitrage
from src.arbitrage.matrix import ArbitrageMatrix
from src.arbitrage.triangular import GraphEdge, TriangularArbitrage
from src.blockchain.dex_price_feed import PoolPrice, sqrt_price_x96_to_price


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_pool(
    dex: str,
    pool: str,
    token0: str,
    token1: str,
    price_1_per_0: float,
    fee_bps: int = 30,
    liquidity: int = 1_000_000,
) -> PoolPrice:
    return PoolPrice(
        dex_name=dex,
        pool_address=pool,
        token0=token0,
        token1=token1,
        token0_symbol=token0[:4],
        token1_symbol=token1[:4],
        price_token1_per_token0=price_1_per_0,
        price_token0_per_token1=1.0 / price_1_per_0 if price_1_per_0 > 0 else 0,
        sqrt_price_x96=0,
        tick=0,
        liquidity=liquidity,
        fee_bps=fee_bps,
        block_number=1000,
    )


# ── sqrt_price_x96_to_price ───────────────────────────────────────────────────

class TestSqrtPriceX96:
    def test_round_trip(self) -> None:
        """sqrtPriceX96 for price=1.0 should return (1.0, 1.0) when decimals match."""
        Q96 = 2 ** 96
        sqrt_price_x96 = int(math.sqrt(1.0) * Q96)
        p1, p0 = sqrt_price_x96_to_price(sqrt_price_x96, 18, 18)
        assert abs(p1 - 1.0) < 1e-6
        assert abs(p0 - 1.0) < 1e-6

    def test_known_price(self) -> None:
        """Verify correct calculation for a known sqrtPriceX96 value."""
        Q96 = 2 ** 96
        target_price = 1800.0   # 1800 USDC per WETH (dec0=18, dec1=6)
        # raw price in smallest units (no decimal adjustment): target * 10^6 / 10^18
        raw_price = target_price * (10 ** 6) / (10 ** 18)
        sqrt_price_x96 = int(math.sqrt(raw_price) * Q96)
        p1, p0 = sqrt_price_x96_to_price(sqrt_price_x96, decimals0=18, decimals1=6)
        assert abs(p1 - target_price) / target_price < 0.01   # within 1%


# ── TriangularArbitrage ───────────────────────────────────────────────────────

class TestTriangularArbitrage:
    def _build_detector_with_cycle(self) -> TriangularArbitrage:
        """
        Create a 3-token cycle where A→B→C→A yields > 1.
        Rates: A→B=1.01, B→C=1.01, C→A=1.01 → product ≈ 1.030 (+3%)
        """
        det = TriangularArbitrage(max_path_length=4)
        pools = [
            make_pool("dex1", "0xpool1", "0xAAA", "0xBBB", 1.01),
            make_pool("dex2", "0xpool2", "0xBBB", "0xCCC", 1.01),
            make_pool("dex3", "0xpool3", "0xCCC", "0xAAA", 1.01),
        ]
        det.update_prices(pools)
        return det

    def test_graph_built(self) -> None:
        det = self._build_detector_with_cycle()
        assert len(det._all_tokens) == 3
        # 3 tokens → 6 directed edges (2 per pool)
        edge_count = sum(len(v) for v in det._graph.values())
        assert edge_count == 6

    def test_graph_edge_log_weight(self) -> None:
        edge = GraphEdge("A", "B", rate=2.0, dex_name="test")
        assert abs(edge.log_neg_rate - (-math.log(2.0))) < 1e-10

    def test_find_opportunities_dfs(self) -> None:
        det = self._build_detector_with_cycle()
        opps = det.find_cycles_dfs("0xaaa", min_profit_pct=0.1)
        assert len(opps) >= 1
        assert opps[0].estimated_profit_pct > 0.1

    def test_no_opportunity_when_rates_equal_one(self) -> None:
        # All rates = 1.0 → no cycle has product > 1 in any direction
        det = TriangularArbitrage()
        pools = [
            make_pool("dex1", "0xpool1", "0xAAA", "0xBBB", 1.0),
            make_pool("dex2", "0xpool2", "0xBBB", "0xCCC", 1.0),
            make_pool("dex3", "0xpool3", "0xCCC", "0xAAA", 1.0),
        ]
        det.update_prices(pools)
        opps = det.find_cycles_dfs("0xaaa", min_profit_pct=0.01)
        assert len(opps) == 0

    def test_empty_graph_returns_no_opportunities(self) -> None:
        det = TriangularArbitrage()
        opps = det.find_opportunities()
        assert opps == []


# ── CrossPlatformArbitrage ────────────────────────────────────────────────────

class TestCrossPlatformArbitrage:
    def test_finds_spread(self) -> None:
        det = CrossPlatformArbitrage()
        pools = [
            make_pool("dex1", "0xp1", "0xUSDC", "0xWETH", 1800.0, fee_bps=30),
            make_pool("dex2", "0xp2", "0xUSDC", "0xWETH", 1850.0, fee_bps=30),
        ]
        det.update_prices(pools)
        opps = det.find_opportunities(min_spread_pct=0.01)
        assert len(opps) >= 1
        best = opps[0]
        assert best.buy_dex == "dex1"
        assert best.sell_dex == "dex2"
        assert best.estimated_profit_pct > 0

    def test_no_opportunity_below_threshold(self) -> None:
        det = CrossPlatformArbitrage()
        pools = [
            make_pool("dex1", "0xp1", "0xUSDC", "0xWETH", 1800.0, fee_bps=30),
            make_pool("dex2", "0xp2", "0xUSDC", "0xWETH", 1800.01, fee_bps=30),
        ]
        det.update_prices(pools)
        # Tiny spread will be wiped out by fees
        opps = det.find_opportunities(min_spread_pct=1.0)
        assert len(opps) == 0

    def test_liquidity_filter(self) -> None:
        det = CrossPlatformArbitrage()
        pools = [
            make_pool("dex1", "0xp1", "0xUSDC", "0xWETH", 1800.0, liquidity=5_000),
            make_pool("dex2", "0xp2", "0xUSDC", "0xWETH", 1900.0, liquidity=5_000),
        ]
        det.update_prices(pools)
        opps = det.find_opportunities(min_spread_pct=0.01)
        filtered = det.filter_by_liquidity(opps, min_liquidity=10_000)
        assert len(filtered) == 0


# ── ArbitrageMatrix ───────────────────────────────────────────────────────────

class TestArbitrageMatrix:
    def test_token_count(self) -> None:
        mat = ArbitrageMatrix()
        pools = [
            make_pool("dex1", "0xp1", "0xA", "0xB", 1.5),
            make_pool("dex2", "0xp2", "0xB", "0xC", 1.5),
            make_pool("dex3", "0xp3", "0xC", "0xA", 1.5),
        ]
        mat.build_from_prices(pools)
        assert mat.token_count == 3

    def test_rate_lookup(self) -> None:
        mat = ArbitrageMatrix()
        pools = [make_pool("dex1", "0xp1", "0xa", "0xb", 2.0)]
        mat.build_from_prices(pools)
        assert abs(mat.get_rate("0xa", "0xb") - 2.0) < 1e-9

    def test_finds_profitable_cycle(self) -> None:
        mat = ArbitrageMatrix()
        # A→B=1.1, B→C=1.1, C→A=1.1 → product ≈ 1.331 (profitable)
        pools = [
            make_pool("dex1", "0xp1", "0xa", "0xb", 1.1),
            make_pool("dex2", "0xp2", "0xb", "0xc", 1.1),
            make_pool("dex3", "0xp3", "0xc", "0xa", 1.1),
        ]
        mat.build_from_prices(pools)
        opps = mat.find_opportunities(min_profit_pct=1.0)
        assert len(opps) >= 1
        assert opps[0].profit_pct > 1.0

    def test_empty_prices(self) -> None:
        mat = ArbitrageMatrix()
        mat.build_from_prices([])
        opps = mat.find_opportunities()
        assert opps == []
