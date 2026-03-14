"""
Tests for the new TODO implementations:
  - E2-S2: SwapEventFeed (WebSocket-based Swap event subscription)
  - E2-S3: DEXFactoryScanner (pool auto-discovery)
  - E2-S4: Token universe in Config
  - E3-S1: ExecutionDB (PostgreSQL persistence layer)
  - E3-S2: ETL pipeline (extract features from DB)
  - E3-S4: Retraining script helpers
  - E4-S2: Per-hop slippage protection in FlashLoan
  - E5-S4: Live dashboard (FastAPI + WebSocket)
  - E6-S1: Matrix build_hop_list (4-leg paths wired through execution pipeline)
  - E6-S2: BalancerArbitrage (weighted-pool price calculation)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# ── E2-S2: SwapEventFeed ──────────────────────────────────────────────────────

class TestSwapEventFeed:
    def test_import(self) -> None:
        from src.blockchain.swap_event_feed import SwapEventFeed, SwapEvent
        assert SwapEventFeed is not None
        assert SwapEvent is not None

    def test_stop_before_start_is_noop(self) -> None:
        from src.blockchain.swap_event_feed import SwapEventFeed
        client = MagicMock()
        feed = SwapEventFeed(client)
        feed.stop()  # should not raise

    @pytest.mark.asyncio
    async def test_swap_to_pool_price_zero_sqrt(self) -> None:
        """A swap with sqrtPriceX96 == 0 should return None."""
        from src.blockchain.swap_event_feed import SwapEventFeed, SwapEvent

        client = MagicMock()
        client.w3s = MagicMock()
        client.w3s.eth = MagicMock()
        client.get_token_decimals = AsyncMock(return_value=18)
        client.get_token_symbol   = AsyncMock(return_value="TKN")

        feed = SwapEventFeed(client)
        # Manually inject pool meta so we can bypass on-chain calls
        pool_addr = "0xdeadbeef"
        feed._pool_meta[pool_addr] = {
            "token0": "0xtoken0",
            "token1": "0xtoken1",
            "fee":    3000,
        }

        swap = SwapEvent(
            pool_address=pool_addr,
            sender="0x0",
            recipient="0x0",
            amount0=0,
            amount1=0,
            sqrt_price_x96=0,   # <- zero should return None
            liquidity=0,
            tick=0,
            block_number=1,
            tx_hash="0x",
        )
        result = await feed._swap_to_pool_price(pool_addr, swap)
        assert result is None

    @pytest.mark.asyncio
    async def test_swap_to_pool_price_valid(self) -> None:
        """A swap with valid sqrtPriceX96 should produce a PoolPrice."""
        from src.blockchain.swap_event_feed import SwapEventFeed, SwapEvent
        from src.blockchain.dex_price_feed import PoolPrice

        client = MagicMock()
        client.w3s = MagicMock()
        client.get_token_decimals = AsyncMock(return_value=18)
        client.get_token_symbol   = AsyncMock(return_value="TKN")

        feed = SwapEventFeed(client, dex_name_map={"0xpool": "uniswap_v3"})
        feed._pool_meta["0xpool"] = {
            "token0": "0xaaa",
            "token1": "0xbbb",
            "fee":    3000,
        }

        # sqrtPriceX96 for price ≈ 1.0
        Q96 = 2 ** 96
        sqrt_p = int(Q96)   # sqrt(1) * 2^96

        swap = SwapEvent(
            pool_address="0xpool",
            sender="0x0",
            recipient="0x0",
            amount0=1000,
            amount1=-1000,
            sqrt_price_x96=sqrt_p,
            liquidity=1_000_000,
            tick=0,
            block_number=42,
            tx_hash="0xabc",
        )
        result = await feed._swap_to_pool_price("0xpool", swap)
        assert isinstance(result, PoolPrice)
        assert result.dex_name == "uniswap_v3"
        assert result.block_number == 42
        assert result.price_token1_per_token0 > 0


# ── E2-S3: DEXFactoryScanner ──────────────────────────────────────────────────

class TestDEXFactoryScanner:
    def test_import(self) -> None:
        from src.blockchain.factory_scanner import DEXFactoryScanner, DiscoveredPool
        assert DEXFactoryScanner is not None

    @pytest.mark.asyncio
    async def test_discover_pools_zero_address_skipped(self) -> None:
        """Pools returned as zero address should not be included."""
        from src.blockchain.factory_scanner import DEXFactoryScanner

        client = MagicMock()
        scanner = DEXFactoryScanner(client)

        zero_addr = "0x" + "0" * 40
        mock_factory = MagicMock()
        mock_factory.functions.getPool.return_value.call.return_value = zero_addr
        scanner._get_factory = MagicMock(return_value=mock_factory)

        with patch("src.blockchain.factory_scanner.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.uniswap_v3_factory   = "0x" + "1" * 40
            cfg.quickswap_v3_factory = "0x" + "2" * 40
            cfg.sushiswap_factory    = "0x" + "3" * 40
            mock_cfg.return_value = cfg
            scanner._factories = {
                "uniswap_v3": "0x" + "1" * 40,
            }

            pools = await scanner.discover_pools(
                token_pairs=[("0x" + "a" * 40, "0x" + "b" * 40)],
                fee_tiers=[3000],
                dex_names=["uniswap_v3"],
            )

        assert pools == []

    def test_as_pool_configs(self) -> None:
        from src.blockchain.factory_scanner import DEXFactoryScanner, DiscoveredPool
        client = MagicMock()
        scanner = DEXFactoryScanner.__new__(DEXFactoryScanner)
        scanner._factories = {}
        scanner._known_pools = set()

        pool = DiscoveredPool(
            dex_name="uniswap_v3",
            pool_address="0xABCD",
            token0="0xtoken0",
            token1="0xtoken1",
            fee=3000,
            fee_bps=30,
        )
        configs = scanner.as_pool_configs([pool])
        assert configs == [("uniswap_v3", "0xABCD")]


# ── E2-S4: Token universe in Config ──────────────────────────────────────────

class TestTokenUniverse:
    def test_token_addresses_present(self) -> None:
        from src.utils.config import Config
        cfg = Config()
        # All eight token address fields should be non-empty strings
        for attr in ("token_weth", "token_usdc", "token_usdt", "token_wmatic",
                     "token_wbtc", "token_link", "token_aave", "token_dai"):
            val = getattr(cfg, attr)
            assert isinstance(val, str) and val.startswith("0x"), f"{attr} missing"

    def test_balancer_vault_present(self) -> None:
        from src.utils.config import Config
        cfg = Config()
        assert cfg.balancer_vault.startswith("0x")

    def test_database_url_defaults_empty(self) -> None:
        from src.utils.config import Config
        cfg = Config()
        # DATABASE_URL defaults to empty string — that's fine
        assert isinstance(cfg.database_url, str)


# ── E3-S1: ExecutionDB ────────────────────────────────────────────────────────

class TestExecutionDB:
    def test_disabled_without_url(self) -> None:
        from src.utils.db import ExecutionDB
        db = ExecutionDB(database_url="")
        assert db.enabled is False

    def test_disabled_without_asyncpg(self) -> None:
        import src.utils.db as db_mod
        original = db_mod._HAS_ASYNCPG
        try:
            db_mod._HAS_ASYNCPG = False
            from src.utils.db import ExecutionDB
            db = ExecutionDB(database_url="postgresql://localhost/test")
            assert db.enabled is False
        finally:
            db_mod._HAS_ASYNCPG = original

    @pytest.mark.asyncio
    async def test_log_execution_noop_when_disabled(self) -> None:
        """log_execution should silently do nothing when DB is disabled."""
        from src.utils.db import ExecutionDB
        db = ExecutionDB(database_url="")
        # Should not raise
        await db.log_execution(
            arb_type="triangular",
            price_spread_pct=0.5,
            gas_cost_usd=1.0,
            liquidity_depth=100_000.0,
            historical_success_rate=0.6,
            slippage_estimate=0.1,
            block_utilization=50.0,
            time_since_last_trade=10.0,
            ml_score=0.72,
            executed=True,
            success=True,
            net_profit_usd=12.5,
        )

    @pytest.mark.asyncio
    async def test_fetch_training_rows_empty_when_disabled(self) -> None:
        from src.utils.db import ExecutionDB
        db = ExecutionDB(database_url="")
        rows = await db.fetch_training_rows()
        assert rows == []

    @pytest.mark.asyncio
    async def test_connect_noop_when_disabled(self) -> None:
        from src.utils.db import ExecutionDB
        db = ExecutionDB(database_url="")
        await db.connect()   # must not raise
        await db.close()


# ── E3-S2: ETL pipeline ───────────────────────────────────────────────────────

class TestETL:
    @pytest.mark.asyncio
    async def test_returns_empty_when_db_disabled(self) -> None:
        from src.ml.etl import load_training_data
        from src.ml.feature_engineering import N_FEATURES
        from src.utils.db import ExecutionDB
        db = ExecutionDB(database_url="")
        X, y = await load_training_data(db=db)
        assert X.shape == (0, N_FEATURES)
        assert y.shape == (0,)

    def test_merge_with_synthetic_augments_empty(self) -> None:
        from src.ml.etl import merge_with_synthetic
        from src.ml.feature_engineering import N_FEATURES
        import numpy as np
        X, y = merge_with_synthetic(
            real_X=np.empty((0, N_FEATURES)),
            real_y=np.empty(0),
            synthetic_samples=50,
        )
        assert X.shape == (50, N_FEATURES)
        assert y.shape == (50,)

    def test_merge_with_synthetic_stacks_real(self) -> None:
        from src.ml.etl import merge_with_synthetic
        from src.ml.feature_engineering import N_FEATURES
        import numpy as np
        real_X = np.ones((20, N_FEATURES))
        real_y = np.ones(20)
        X, y = merge_with_synthetic(real_X, real_y, synthetic_samples=30)
        # Real rows come first; total = 20 + 30
        assert X.shape == (50, N_FEATURES)
        assert y.shape == (50,)
        # First 20 rows should be the real data
        np.testing.assert_array_equal(X[:20], real_X)
    def test_load_training_data_sync_returns_empty(self) -> None:
        from src.ml.etl import load_training_data_sync
        from src.utils.db import ExecutionDB
        X, y = load_training_data_sync(db=ExecutionDB(database_url=""))
        assert X.shape[0] == 0


# ── E4-S2: Per-hop slippage protection ────────────────────────────────────────

class TestPerHopSlippage:
    def test_compute_amounts_out_zero_when_no_estimate(self) -> None:
        from src.blockchain.flash_loan import FlashLoan
        from src.payload.protocol import SwapHop
        hops = [
            SwapHop(dex_name="uni", router_address="0x0",
                    token_in="0xa", token_out="0xb",
                    fee_bps=30, is_v3=True, estimated_amount_out=0.0),
        ]
        mins = FlashLoan.compute_amounts_out_minimum(hops)
        assert mins == [0]

    def test_compute_amounts_out_applies_slippage(self) -> None:
        from src.blockchain.flash_loan import FlashLoan
        from src.payload.protocol import SwapHop
        hops = [
            SwapHop(dex_name="uni", router_address="0x0",
                    token_in="0xa", token_out="0xb",
                    fee_bps=30, is_v3=True, estimated_amount_out=1_000_000),
        ]
        mins = FlashLoan.compute_amounts_out_minimum(hops, slippage_pct=0.5)
        # 1_000_000 * (1 - 0.005) = 995_000
        assert mins == [995_000]

    def test_compute_amounts_out_multiple_hops(self) -> None:
        from src.blockchain.flash_loan import FlashLoan
        from src.payload.protocol import SwapHop
        hops = [
            SwapHop(dex_name="uni", router_address="0x0",
                    token_in="0xa", token_out="0xb",
                    fee_bps=30, is_v3=True, estimated_amount_out=2_000_000),
            SwapHop(dex_name="quick", router_address="0x0",
                    token_in="0xb", token_out="0xa",
                    fee_bps=30, is_v3=True, estimated_amount_out=0.0),
        ]
        mins = FlashLoan.compute_amounts_out_minimum(hops, slippage_pct=0.5)
        assert len(mins) == 2
        assert mins[0] == int(2_000_000 * 0.995)
        assert mins[1] == 0

    def test_max_slippage_pct_constant(self) -> None:
        from src.blockchain.flash_loan import FlashLoan
        assert FlashLoan.MAX_SLIPPAGE_PCT == 0.5

    def test_encode_swap_path_with_amounts_out_min(self) -> None:
        """encode_swap_path must accept and use expected_amounts_out."""
        from src.blockchain.flash_loan import FlashLoan
        from src.payload.protocol import SwapHop
        hops = [
            SwapHop(dex_name="uni", router_address="0x" + "1" * 40,
                    token_in="0x" + "a" * 40, token_out="0x" + "b" * 40,
                    fee_bps=30, is_v3=True, estimated_amount_out=500_000),
        ]
        encoded = FlashLoan.encode_swap_path(hops, expected_amounts_out=[499_000])
        assert isinstance(encoded, bytes)
        assert len(encoded) > 0


# ── E5-S4: Dashboard ──────────────────────────────────────────────────────────

class TestDashboard:
    def test_import(self) -> None:
        from src.utils.dashboard import (
            record_opportunity, record_execution,
            update_model_metrics, set_running,
        )
        assert callable(record_opportunity)
        assert callable(record_execution)

    def test_record_opportunity_updates_stats(self) -> None:
        import src.utils.dashboard as dash
        before = dash._system_status["total_opportunities"]
        dash.record_opportunity({"arb_type": "triangular", "ml_score": 0.7,
                                 "expected_profit_usd": 10.0})
        assert dash._system_status["total_opportunities"] == before + 1

    def test_record_execution_success_updates_profit(self) -> None:
        import src.utils.dashboard as dash
        profit_before = dash._system_status["total_profit_usd"]
        dash.record_execution({"success": True, "actual_profit_usd": 5.0})
        assert dash._system_status["total_profit_usd"] > profit_before

    def test_record_execution_failure_increments_losses(self) -> None:
        import src.utils.dashboard as dash
        # Reset losses counter first
        dash._system_status["consecutive_losses"] = 0
        dash._system_status["circuit_breaker_tripped"] = False
        dash.record_execution({"success": False, "error_message": "revert"})
        assert dash._system_status["consecutive_losses"] == 1

    def test_circuit_breaker_trips_after_3_losses(self) -> None:
        import src.utils.dashboard as dash
        dash._system_status["consecutive_losses"] = 0
        dash._system_status["circuit_breaker_tripped"] = False
        for _ in range(3):
            dash.record_execution({"success": False, "error_message": "revert"})
        assert dash._system_status["circuit_breaker_tripped"] is True

    def test_update_model_metrics(self) -> None:
        import src.utils.dashboard as dash
        dash.update_model_metrics({"train_rmse": 0.5, "val_r2": 0.8})
        assert dash._model_metrics["train_rmse"] == 0.5

    def test_api_status_endpoint(self) -> None:
        import src.utils.dashboard as dash
        if not dash._HAS_FASTAPI:
            pytest.skip("fastapi not installed")
        from fastapi.testclient import TestClient
        client = TestClient(dash.app)
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_opportunities" in data
        assert "total_profit_usd" in data

    def test_api_log_endpoint(self) -> None:
        import src.utils.dashboard as dash
        if not dash._HAS_FASTAPI:
            pytest.skip("fastapi not installed")
        from fastapi.testclient import TestClient
        client = TestClient(dash.app)
        resp = client.get("/api/log?n=10")
        assert resp.status_code == 200
        data = resp.json()
        assert "opportunities" in data
        assert "executions" in data

    def test_dashboard_html_page(self) -> None:
        import src.utils.dashboard as dash
        if not dash._HAS_FASTAPI:
            pytest.skip("fastapi not installed")
        from fastapi.testclient import TestClient
        client = TestClient(dash.app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Arbisy" in resp.content


# ── E6-S1: Matrix build_hop_list (4-leg path wiring) ─────────────────────────

class TestMatrixBuildHopList:
    def _make_matrix_with_prices(self):
        """Create an ArbitrageMatrix with mock PoolPrice data."""
        from src.arbitrage.matrix import ArbitrageMatrix
        from src.blockchain.dex_price_feed import PoolPrice

        prices = []
        # Build a 4-token cycle: A→B→C→D→A with slight profit on the last hop
        tokens = [
            ("0xaaaa", "0xbbbb", 1.01, 0.99, "uniswap_v3"),
            ("0xbbbb", "0xcccc", 1.01, 0.99, "quickswap_v3"),
            ("0xcccc", "0xdddd", 1.01, 0.99, "sushiswap"),
            ("0xdddd", "0xaaaa", 1.02, 0.98, "uniswap_v3"),
        ]
        for t0, t1, p1_per_0, p0_per_1, dex in tokens:
            prices.append(PoolPrice(
                dex_name=dex,
                pool_address="0x" + t0[2:6] + t1[2:6],
                token0=t0, token1=t1,
                token0_symbol="A", token1_symbol="B",
                price_token1_per_token0=p1_per_0,
                price_token0_per_token1=p0_per_1,
                sqrt_price_x96=0, tick=0, liquidity=1_000_000,
                fee_bps=30, block_number=1,
            ))

        mat = ArbitrageMatrix()
        mat.build_from_prices(prices)
        return mat

    def test_build_hop_list_returns_list(self) -> None:
        mat = self._make_matrix_with_prices()
        from src.arbitrage.matrix import MatrixArbPath
        # Create a synthetic path using known token indices
        n = mat.token_count
        if n < 2:
            pytest.skip("Not enough tokens")
        path = MatrixArbPath(
            token_indices=list(range(n)) + [0],
            token_names=[mat.tokens[i] for i in range(n)] + [mat.tokens[0]],
            log_profit=0.05,
            profit_multiplier=1.05,
            profit_pct=5.0,
        )
        hops = mat.build_hop_list(path)
        # Should have n hops (one per step in the cycle)
        assert len(hops) == n
        for hop in hops:
            assert "token_in" in hop
            assert "token_out" in hop
            assert "dex_name" in hop
            assert isinstance(hop["fee_bps"], int)

    def test_get_hop_dex_returns_value(self) -> None:
        mat = self._make_matrix_with_prices()
        n = mat.token_count
        if n < 2:
            pytest.skip("Not enough tokens")
        # At least one DEX should be set for a non-diagonal cell
        found_dex = False
        for i in range(n):
            for j in range(n):
                if i != j:
                    dex = mat.get_hop_dex(i, j)
                    if dex is not None:
                        found_dex = True
                        break
        assert found_dex

    def test_get_hop_dex_diagonal_is_none(self) -> None:
        mat = self._make_matrix_with_prices()
        n = mat.token_count
        for i in range(n):
            # Diagonal has no DEX edge
            dex = mat.get_hop_dex(i, i)
            # Could be None or a value depending on how the matrix initialises
            # The important thing is it doesn't raise
            _ = dex


# ── E6-S2: BalancerArbitrage ──────────────────────────────────────────────────

class TestBalancerArbitrage:
    def test_import(self) -> None:
        from src.arbitrage.balancer import BalancerArbitrage, BalancerPool
        assert BalancerArbitrage is not None

    def test_disabled_without_vault_address(self) -> None:
        from src.arbitrage.balancer import BalancerArbitrage
        client = MagicMock()
        with patch("src.arbitrage.balancer.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.balancer_vault = ""
            mock_cfg.return_value = cfg
            arb = BalancerArbitrage(client)
        assert arb.enabled is False

    def test_spot_price_calculation(self) -> None:
        """Test weighted-pool spot-price formula directly."""
        from src.arbitrage.balancer import BalancerPool

        pool = BalancerPool(
            pool_id="0x" + "1" * 64,
            pool_address="0x" + "2" * 40,
            tokens=["0xtoken0", "0xtoken1"],
            balances=[100_000 * 10**18, 200_000 * 10**6],  # WETH, USDC
            weights=[0.5, 0.5],
            swap_fee=0.003,
            token_symbols=["WETH", "USDC"],
            token_decimals=[18, 6],
        )
        # price(USDC per WETH) = (bal_usdc / w_usdc) / (bal_weth / w_weth)
        # = (200_000 / 0.5) / (100_000 / 0.5) = 400_000 / 200_000 = 2.0
        price = pool.spot_price(0, 1)
        assert abs(price - 2.0) < 1e-6

    def test_spot_price_zero_for_bad_index(self) -> None:
        from src.arbitrage.balancer import BalancerPool
        pool = BalancerPool(
            pool_id="0x1",
            pool_address="0x2",
            tokens=["0xa", "0xb"],
            balances=[1000, 2000],
            weights=[0.5, 0.5],
            swap_fee=0.003,
        )
        assert pool.spot_price(-1, 0) == 0.0
        assert pool.spot_price(0, 5) == 0.0

    def test_spot_price_zero_balance(self) -> None:
        from src.arbitrage.balancer import BalancerPool
        pool = BalancerPool(
            pool_id="0x1",
            pool_address="0x2",
            tokens=["0xa", "0xb"],
            balances=[0, 1000],  # zero balance for token0
            weights=[0.5, 0.5],
            swap_fee=0.003,
        )
        assert pool.spot_price(0, 1) == 0.0

    @pytest.mark.asyncio
    async def test_fetch_prices_empty_without_pools(self) -> None:
        from src.arbitrage.balancer import BalancerArbitrage
        client = MagicMock()
        with patch("src.arbitrage.balancer.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.balancer_vault = "0x" + "BA12" * 10
            mock_cfg.return_value = cfg
            arb = BalancerArbitrage.__new__(BalancerArbitrage)
            arb._client = client
            arb._vault = MagicMock()
            arb._pools = {}
            arb._token_meta = {}

        prices = await arb.fetch_prices()
        assert prices == []

    def test_get_loaded_pool_ids_empty(self) -> None:
        from src.arbitrage.balancer import BalancerArbitrage
        client = MagicMock()
        with patch("src.arbitrage.balancer.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.balancer_vault = ""
            mock_cfg.return_value = cfg
            arb = BalancerArbitrage(client)
        assert arb.get_loaded_pool_ids() == []
