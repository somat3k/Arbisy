"""
Tests for ArrayBuilder and PathArray (src/arbitrage/array_builder.py).
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
import pytest

from src.arbitrage.array_builder import ArrayBuilder, PathArray, path_to_features
from src.blockchain.dex_price_feed import PoolPrice
from src.ml.feature_engineering import N_FEATURES

# ── Helpers ───────────────────────────────────────────────────────────────────

TOKEN_A = "0xaaaa"
TOKEN_B = "0xbbbb"
TOKEN_C = "0xcccc"
TOKEN_D = "0xdddd"


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


def make_profitable_cycle_pools(rate: float = 1.1, fee_bps: int = 30) -> List[PoolPrice]:
    """3-hop profitable cycle: A→B→C→A with given rate per hop."""
    return [
        make_pool("dex1", "0xpool1", TOKEN_A, TOKEN_B, rate, fee_bps=fee_bps),
        make_pool("dex2", "0xpool2", TOKEN_B, TOKEN_C, rate, fee_bps=fee_bps),
        make_pool("dex3", "0xpool3", TOKEN_C, TOKEN_A, rate, fee_bps=fee_bps),
    ]


# ── Graph construction ────────────────────────────────────────────────────────

class TestArrayBuilderGraph:
    def test_empty_prices_empty_graph(self) -> None:
        ab = ArrayBuilder()
        ab.update_prices([])
        assert ab.token_count == 0

    def test_token_count_after_prices(self) -> None:
        ab = ArrayBuilder()
        ab.update_prices(make_profitable_cycle_pools())
        assert ab.token_count == 3

    def test_tokens_list(self) -> None:
        ab = ArrayBuilder()
        ab.update_prices(make_profitable_cycle_pools())
        tokens = ab.tokens
        assert TOKEN_A.lower() in tokens
        assert TOKEN_B.lower() in tokens
        assert TOKEN_C.lower() in tokens

    def test_best_rate_kept(self) -> None:
        """When two pools offer the same pair, the higher effective rate wins."""
        ab = ArrayBuilder()
        pools = [
            make_pool("dex1", "0xp1", TOKEN_A, TOKEN_B, 1.5, fee_bps=30),
            make_pool("dex2", "0xp2", TOKEN_A, TOKEN_B, 2.0, fee_bps=30),
        ]
        ab.update_prices(pools)
        # Effective rate for dex2: 2.0 * (1 - 30/10000) = 2.0 * 0.997 = 1.994
        # Effective rate for dex1: 1.5 * 0.997 = 1.4955  → dex2 wins
        edge = ab._graph[TOKEN_A.lower()][TOKEN_B.lower()]
        assert edge.dex_name == "dex2"
        assert abs(edge.rate - 2.0 * (1 - 30 / 10_000)) < 1e-9

    def test_fee_applied_to_effective_rate(self) -> None:
        ab = ArrayBuilder()
        fee_bps = 500  # 5%
        ab.update_prices([make_pool("dex1", "0xp1", TOKEN_A, TOKEN_B, 2.0, fee_bps=fee_bps)])
        edge = ab._graph[TOKEN_A.lower()][TOKEN_B.lower()]
        expected = 2.0 * (1 - fee_bps / 10_000)
        assert abs(edge.rate - expected) < 1e-9


# ── Profitable path discovery ─────────────────────────────────────────────────

class TestArrayBuilderPaths:
    def test_finds_profitable_3hop_cycle(self) -> None:
        ab = ArrayBuilder(max_hops=4, min_profit_pct=0.01)
        ab.update_prices(make_profitable_cycle_pools(rate=1.1, fee_bps=0))
        results = ab.build_profitable_arrays(
            loan_assets=[TOKEN_A], loan_amount_usd=10_000.0
        )
        assert len(results) >= 1
        assert results[0].profit_usd > 0
        assert results[0].final_usd > results[0].initial_usd

    def test_no_opportunity_when_rate_is_one(self) -> None:
        ab = ArrayBuilder(min_profit_pct=0.01)
        ab.update_prices(make_profitable_cycle_pools(rate=1.0, fee_bps=0))
        results = ab.build_profitable_arrays(loan_assets=[TOKEN_A])
        assert results == []

    def test_path_starts_and_ends_with_loan_asset(self) -> None:
        ab = ArrayBuilder(max_hops=4, min_profit_pct=0.01)
        ab.update_prices(make_profitable_cycle_pools(rate=1.1, fee_bps=0))
        results = ab.build_profitable_arrays(loan_assets=[TOKEN_A])
        for pa in results:
            assert pa.token_path[0]  == TOKEN_A.lower()
            assert pa.token_path[-1] == TOKEN_A.lower()
            assert pa.loan_asset     == TOKEN_A.lower()

    def test_profit_usd_equals_final_minus_initial(self) -> None:
        ab = ArrayBuilder(max_hops=4, min_profit_pct=0.01)
        ab.update_prices(make_profitable_cycle_pools(rate=1.1, fee_bps=0))
        results = ab.build_profitable_arrays(loan_assets=[TOKEN_A], loan_amount_usd=5_000.0)
        for pa in results:
            assert abs(pa.profit_usd - (pa.final_usd - pa.initial_usd)) < 1e-6

    def test_cumulative_rate_consistent(self) -> None:
        ab = ArrayBuilder(max_hops=4, min_profit_pct=0.01)
        ab.update_prices(make_profitable_cycle_pools(rate=1.1, fee_bps=0))
        results = ab.build_profitable_arrays(loan_assets=[TOKEN_A], loan_amount_usd=1.0)
        for pa in results:
            expected_final = pa.initial_usd * pa.cumulative_rate
            assert abs(pa.final_usd - expected_final) < 1e-9

    def test_flash_premium_deducted(self) -> None:
        """Ensure 0.05% (5 bps) flash premium is applied before the profitability check."""
        ab = ArrayBuilder(max_hops=4, min_profit_pct=0.0)
        # Very high rate: 1.5 per hop × 3 hops with 0 fee = 1.5^3 = 3.375
        # After 5 bps (0.05%) premium: 3.375 * (1 - 5/10_000) = 3.375 * 0.9995 = 3.3733
        ab.update_prices(make_profitable_cycle_pools(rate=1.5, fee_bps=0))
        results = ab.build_profitable_arrays(loan_assets=[TOKEN_A], loan_amount_usd=100.0)
        assert len(results) >= 1
        pa = results[0]
        product_of_rates = pa.rate_path[0] * pa.rate_path[1] * pa.rate_path[2]
        expected_eff = product_of_rates * (1 - 5 / 10_000)
        assert abs(pa.cumulative_rate - expected_eff) < 1e-9

    def test_no_duplicate_paths(self) -> None:
        ab = ArrayBuilder(max_hops=4, min_profit_pct=0.01)
        ab.update_prices(make_profitable_cycle_pools(rate=1.1, fee_bps=0))
        results = ab.build_profitable_arrays(loan_assets=[TOKEN_A])
        # Token paths should be unique
        paths = [tuple(p.token_path) for p in results]
        assert len(paths) == len(set(paths))

    def test_loan_asset_not_in_graph_returns_empty(self) -> None:
        ab = ArrayBuilder()
        ab.update_prices(make_profitable_cycle_pools(rate=1.1))
        # TOKEN_D is not in the graph
        results = ab.build_profitable_arrays(loan_assets=[TOKEN_D])
        assert results == []

    def test_respects_max_hops(self) -> None:
        ab = ArrayBuilder(max_hops=3, min_profit_pct=0.0)
        ab.update_prices(make_profitable_cycle_pools(rate=1.5, fee_bps=0))
        results = ab.build_profitable_arrays(loan_assets=[TOKEN_A])
        for pa in results:
            assert pa.num_hops <= 3

    def test_4hop_cycle_found(self) -> None:
        """Test that a 4-hop cycle A→B→C→D→A is discovered when profitable."""
        ab = ArrayBuilder(max_hops=4, min_profit_pct=0.01)
        pools = [
            make_pool("dex1", "0xp1", TOKEN_A, TOKEN_B, 1.15, fee_bps=0),
            make_pool("dex2", "0xp2", TOKEN_B, TOKEN_C, 1.15, fee_bps=0),
            make_pool("dex3", "0xp3", TOKEN_C, TOKEN_D, 1.15, fee_bps=0),
            make_pool("dex4", "0xp4", TOKEN_D, TOKEN_A, 1.15, fee_bps=0),
        ]
        ab.update_prices(pools)
        results = ab.build_profitable_arrays(loan_assets=[TOKEN_A])
        four_hop = [p for p in results if p.num_hops == 4]
        assert len(four_hop) >= 1

    def test_sorted_by_profit_without_model(self) -> None:
        ab = ArrayBuilder(max_hops=4, min_profit_pct=0.01)
        # Mix of A-only and multi-asset loan
        pools = make_profitable_cycle_pools(rate=1.1, fee_bps=0)
        ab.update_prices(pools)
        results = ab.build_profitable_arrays(loan_assets=[TOKEN_A, TOKEN_B, TOKEN_C])
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].profit_pct >= results[i + 1].profit_pct

    def test_sorted_by_nn_score_with_model(self) -> None:
        """When a model is supplied, results are sorted by nn_score (descending)."""
        from unittest.mock import MagicMock

        # Return deterministically descending scores: 1.0, 0.5, 0.333..., etc.
        call_count: list = [0]

        def _deterministic_score(x: "np.ndarray") -> float:
            call_count[0] += 1
            return 1.0 / call_count[0]

        mock_model = MagicMock()
        mock_model.predict_score.side_effect = _deterministic_score

        ab = ArrayBuilder(max_hops=4, min_profit_pct=0.01)
        ab.update_prices(make_profitable_cycle_pools(rate=1.1, fee_bps=0))
        results = ab.build_profitable_arrays(
            loan_assets=[TOKEN_A], nn_model=mock_model
        )
        if len(results) >= 2:
            for i in range(len(results) - 1):
                assert results[i].nn_score >= results[i + 1].nn_score


# ── PathArray dataclass ────────────────────────────────────────────────────────

class TestPathArray:
    def _make_path(self) -> PathArray:
        return PathArray(
            loan_asset=TOKEN_A.lower(),
            token_path=[TOKEN_A.lower(), TOKEN_B.lower(), TOKEN_C.lower(), TOKEN_A.lower()],
            dex_path=["dex1", "dex2", "dex3"],
            rate_path=[1.09967, 1.09967, 1.09967],
            fee_path=[30, 30, 30],
            liquidity_path=[500_000, 400_000, 600_000],
            cumulative_rate=1.3289,
            initial_usd=10_000.0,
            final_usd=13_289.0,
            profit_usd=3_289.0,
            profit_pct=32.89,
        )

    def test_num_hops(self) -> None:
        pa = self._make_path()
        assert pa.num_hops == 3

    def test_unique_dexes(self) -> None:
        pa = self._make_path()
        assert pa.unique_dexes == 3

    def test_repr(self) -> None:
        pa = self._make_path()
        r = repr(pa)
        assert "PathArray" in r
        assert "profit=" in r

    def test_final_usd_gt_initial(self) -> None:
        pa = self._make_path()
        assert pa.final_usd > pa.initial_usd
        assert pa.profit_usd > 0


# ── path_to_features ──────────────────────────────────────────────────────────

class TestPathToFeatures:
    def _make_path(self) -> PathArray:
        return PathArray(
            loan_asset=TOKEN_A.lower(),
            token_path=[TOKEN_A.lower(), TOKEN_B.lower(), TOKEN_C.lower(), TOKEN_A.lower()],
            dex_path=["dex1", "dex2", "dex3"],
            rate_path=[1.09967, 1.09967, 1.09967],
            fee_path=[30, 30, 30],
            liquidity_path=[500_000, 400_000, 600_000],
            cumulative_rate=1.3289,
            initial_usd=10_000.0,
            final_usd=13_289.0,
            profit_usd=3_289.0,
            profit_pct=32.89,
        )

    def test_feature_vector_length(self) -> None:
        pa  = self._make_path()
        vec = path_to_features(pa)
        assert vec.shape == (N_FEATURES,)

    def test_no_nan_or_inf(self) -> None:
        pa  = self._make_path()
        vec = path_to_features(pa)
        assert not np.any(np.isnan(vec))
        assert not np.any(np.isinf(vec))

    def test_is_triangular_flag(self) -> None:
        from src.ml.feature_engineering import FEATURE_NAMES
        tri_idx = FEATURE_NAMES.index("is_triangular")
        pa_3hop = self._make_path()
        vec3 = path_to_features(pa_3hop)
        assert vec3[tri_idx] == 1.0

        pa_4hop = PathArray(
            loan_asset=TOKEN_A.lower(),
            token_path=[TOKEN_A.lower(), TOKEN_B.lower(), TOKEN_C.lower(), TOKEN_D.lower(), TOKEN_A.lower()],
            dex_path=["d1", "d2", "d3", "d4"],
            rate_path=[1.1, 1.1, 1.1, 1.1],
            fee_path=[30, 30, 30, 30],
            liquidity_path=[100, 100, 100, 100],
            cumulative_rate=1.4,
            initial_usd=1_000.0,
            final_usd=1_400.0,
            profit_usd=400.0,
            profit_pct=40.0,
        )
        vec4 = path_to_features(pa_4hop)
        assert vec4[tri_idx] == 0.0

    def test_gas_cost_passed_through(self) -> None:
        from src.ml.feature_engineering import FEATURE_NAMES
        gas_idx = FEATURE_NAMES.index("gas_cost_usd")
        pa  = self._make_path()
        vec = path_to_features(pa, gas_cost_usd=3.5)
        assert abs(vec[gas_idx] - 3.5) < 1e-9
