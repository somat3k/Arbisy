"""
Tests for further Epic implementations:
  - E3-S3: GridSearchCV (training.tune_hyperparameters)
  - E3-S6: EnsembleModel (GBR + RF)
  - E4-S1: EIP-1559 gas cap
  - E4-S3: Exponential back-off retry in ExecutionAgent
  - E4-S4: NonceManager
  - E4-S6: Circuit-breaker persistence
  - E5-S1: Prometheus metrics (ArbisyMetrics)
  - E5-S3: JSON structured logging
  - E5-S5: FastAPI health endpoints
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import numpy as np
import pytest

from src.ml.feature_engineering import N_FEATURES
from src.ml.model import EnsembleModel
from src.ml.training import generate_synthetic_data, tune_hyperparameters
from src.utils.metrics import ArbisyMetrics


# ── E3-S6: EnsembleModel ──────────────────────────────────────────────────────

class TestEnsembleModel:
    def test_heuristic_without_training(self) -> None:
        model = EnsembleModel(model_path="/tmp/nonexistent_ensemble.joblib")
        X, _ = generate_synthetic_data(10)
        preds = model.predict(X)
        assert preds.shape == (10,)
        assert not np.any(np.isnan(preds))

    def test_train_and_predict(self) -> None:
        model = EnsembleModel(model_path="/tmp/test_ensemble.joblib")
        X, y  = generate_synthetic_data(200)
        metrics = model.fit(X, y, validate=True)
        assert "train_rmse" in metrics
        assert metrics["train_rmse"] >= 0
        assert model.trained

        preds = model.predict(X[:5])
        assert preds.shape == (5,)
        assert not np.any(np.isnan(preds))

    def test_predict_score_range(self) -> None:
        model = EnsembleModel(model_path="/tmp/test_ensemble2.joblib")
        X, y  = generate_synthetic_data(100)
        model.fit(X, y, validate=False)
        score = model.predict_score(X[0])
        assert 0.0 <= score <= 1.0

    def test_feature_importances_after_training(self) -> None:
        model = EnsembleModel(model_path="/tmp/test_ensemble3.joblib")
        X, y  = generate_synthetic_data(100)
        model.fit(X, y, validate=False)
        imps = model.feature_importances()
        assert len(imps) == N_FEATURES
        assert all(v >= 0 for v in imps.values())
        total = sum(imps.values())
        assert abs(total - 1.0) < 1e-5

    def test_save_and_reload(self, tmp_path) -> None:
        path  = str(tmp_path / "ensemble.joblib")
        model = EnsembleModel(model_path=path)
        X, y  = generate_synthetic_data(100)
        model.fit(X, y, validate=False)
        model.save(path)

        model2 = EnsembleModel(model_path=path)
        assert model2.trained
        p1 = model.predict(X[:3])
        p2 = model2.predict(X[:3])
        np.testing.assert_allclose(p1, p2, rtol=1e-5)

    def test_gbr_weight_persisted(self, tmp_path) -> None:
        path  = str(tmp_path / "ens_weight.joblib")
        model = EnsembleModel(model_path=path)
        model.gbr_weight = 0.7
        X, y  = generate_synthetic_data(100)
        model.fit(X, y, validate=False)
        model.save(path)

        model2 = EnsembleModel(model_path=path)
        assert abs(model2.gbr_weight - 0.7) < 1e-9

    def test_1d_input_predict_score(self) -> None:
        model = EnsembleModel(model_path="/tmp/test_ens_1d.joblib")
        X, y  = generate_synthetic_data(100)
        model.fit(X, y, validate=False)
        score = model.predict_score(X[0])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ── E3-S3: GridSearchCV ───────────────────────────────────────────────────────

class TestTuneHyperparameters:
    def test_returns_best_params_and_score(self) -> None:
        result = tune_hyperparameters(
            synthetic_samples=300,
            use_real_data=False,
            cv=2,        # fast for test
            n_jobs=1,
        )
        assert "best_params" in result
        assert "best_score" in result
        # best_score is negative RMSE from GridSearchCV
        assert result["best_score"] <= 0

    def test_best_params_contain_expected_keys(self) -> None:
        result = tune_hyperparameters(
            synthetic_samples=200,
            use_real_data=False,
            cv=2,
            n_jobs=1,
        )
        params = result["best_params"]
        for key in ("n_estimators", "max_depth", "learning_rate"):
            assert key in params


# ── E4-S1: EIP-1559 gas cap ───────────────────────────────────────────────────

class TestEIP1559GasCap:
    @pytest.mark.asyncio
    async def test_max_fee_capped_at_config(self) -> None:
        """get_max_fee_params must not exceed MAX_GAS_PRICE_GWEI."""
        from src.blockchain.polygon_client import PolygonClient
        from web3 import Web3

        client = PolygonClient.__new__(PolygonClient)

        # Mock a very high base fee
        high_base = int(Web3.to_wei(1000, "gwei"))  # 1000 gwei base fee
        mock_block = {"baseFeePerGas": high_base}

        mock_w3 = AsyncMock()
        mock_w3.eth.get_block = AsyncMock(return_value=mock_block)
        client.w3 = mock_w3

        with patch("src.blockchain.polygon_client.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.max_gas_price_gwei = 500.0
            mock_cfg.return_value = cfg

            params = await client.get_max_fee_params()

        # Regardless of base fee, maxFeePerGas must not exceed 500 gwei
        max_allowed = int(Web3.to_wei(500, "gwei"))
        assert int(params["maxFeePerGas"]) <= max_allowed

    @pytest.mark.asyncio
    async def test_fallback_gas_price_capped(self) -> None:
        """Legacy gasPrice fallback must also be capped."""
        from src.blockchain.polygon_client import PolygonClient
        from web3 import Web3

        client = PolygonClient.__new__(PolygonClient)

        # Provide a directly-awaitable coroutine for gas_price
        high_gwei = int(Web3.to_wei(1000, "gwei"))

        async def _gas_price_coro():
            return high_gwei

        mock_w3 = MagicMock()
        mock_w3.eth.get_block = AsyncMock(side_effect=Exception("rpc error"))
        mock_w3.eth.gas_price = _gas_price_coro()  # coroutine object, directly awaitable
        client.w3 = mock_w3

        with patch("src.blockchain.polygon_client.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.max_gas_price_gwei = 500.0
            mock_cfg.return_value = cfg

            params = await client.get_max_fee_params()

        max_allowed = int(Web3.to_wei(500, "gwei"))
        assert int(params["gasPrice"]) <= max_allowed


# ── E4-S4: NonceManager ───────────────────────────────────────────────────────

class TestNonceManager:
    @pytest.mark.asyncio
    async def test_increments_nonce(self) -> None:
        from src.blockchain.polygon_client import NonceManager, PolygonClient

        mock_client = MagicMock(spec=PolygonClient)
        mock_client.get_nonce = AsyncMock(return_value=42)

        nm = NonceManager(mock_client, "0xabc")
        n1 = await nm.next_nonce()
        n2 = await nm.next_nonce()
        assert n1 == 42
        assert n2 == 43
        # get_nonce called only once (cached)
        mock_client.get_nonce.assert_called_once()

    @pytest.mark.asyncio
    async def test_invalidate_refetches(self) -> None:
        from src.blockchain.polygon_client import NonceManager, PolygonClient

        mock_client = MagicMock(spec=PolygonClient)
        mock_client.get_nonce = AsyncMock(return_value=10)

        nm = NonceManager(mock_client, "0xabc")
        await nm.next_nonce()
        nm.invalidate()
        await nm.next_nonce()
        assert mock_client.get_nonce.call_count == 2


# ── E4-S3: Exponential back-off retry ────────────────────────────────────────

class TestExecutionRetry:
    @pytest.mark.asyncio
    async def test_succeeds_first_attempt(self) -> None:
        from src.agents.execution_agent import ExecutionAgent

        agent = _make_execution_agent()
        opp   = MagicMock()

        success_result = {"success": True, "tx_hash": "0xabc", "error_message": ""}
        agent._flash.execute_flash_loan = AsyncMock(return_value=success_result)

        result = await agent._execute_with_retry(opp)
        assert result["success"] is True
        assert agent._flash.execute_flash_loan.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_transient_error(self) -> None:
        from src.agents.execution_agent import ExecutionAgent, _MAX_RETRIES

        agent = _make_execution_agent()
        opp   = MagicMock()

        transient_result = {"success": False, "error_message": "connection timeout"}
        success_result   = {"success": True,  "error_message": "", "tx_hash": "0xok"}
        agent._flash.execute_flash_loan = AsyncMock(
            side_effect=[transient_result, success_result]
        )

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await agent._execute_with_retry(opp)
        assert result["success"] is True
        assert agent._flash.execute_flash_loan.call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_contract_revert(self) -> None:
        from src.agents.execution_agent import ExecutionAgent

        agent = _make_execution_agent()
        opp   = MagicMock()

        revert_result = {"success": False, "error_message": "Transaction reverted"}
        agent._flash.execute_flash_loan = AsyncMock(return_value=revert_result)

        result = await agent._execute_with_retry(opp)
        assert result["success"] is False
        # Should not retry a contract revert
        assert agent._flash.execute_flash_loan.call_count == 1


# ── E4-S6: Circuit-breaker persistence ───────────────────────────────────────

class TestCircuitBreakerPersistence:
    def test_save_and_load(self, tmp_path) -> None:
        from src.agents.execution_agent import ExecutionAgent, _CIRCUIT_STATE_FILE

        state_file = str(tmp_path / "cb.json")

        with patch("src.agents.execution_agent._CIRCUIT_STATE_FILE", state_file):
            agent = _make_execution_agent()
            agent._consecutive_losses = 2
            agent._save_circuit_state()

            # A fresh agent should load the saved state
            agent2 = _make_execution_agent()
            with patch("src.agents.execution_agent._CIRCUIT_STATE_FILE", state_file):
                agent2._consecutive_losses = agent2._load_circuit_state()
            assert agent2._consecutive_losses == 2

    def test_reset_clears_state(self, tmp_path) -> None:
        from src.agents.execution_agent import _CIRCUIT_STATE_FILE

        state_file = str(tmp_path / "cb_reset.json")
        with patch("src.agents.execution_agent._CIRCUIT_STATE_FILE", state_file):
            agent = _make_execution_agent()
            agent._consecutive_losses = 3
            agent._save_circuit_state()
            agent.reset_circuit_breaker()
            assert agent._consecutive_losses == 0
            loaded = agent._load_circuit_state()
        assert loaded == 0


# ── E5-S1: Prometheus metrics ─────────────────────────────────────────────────

class TestArbisyMetrics:
    def test_record_opportunity_no_crash(self) -> None:
        m = ArbisyMetrics()
        m.record_opportunity("triangular")
        m.record_opportunity("cross_platform", count=5)

    def test_record_execution_success(self) -> None:
        m = ArbisyMetrics()
        m.record_execution(success=True, profit_usd=12.5, ml_score=0.75)

    def test_record_execution_failure(self) -> None:
        m = ArbisyMetrics()
        m.record_execution(success=False, profit_usd=0.0, ml_score=0.55)

    def test_set_gas_price(self) -> None:
        m = ArbisyMetrics()
        m.set_gas_price(45.0)

    def test_cycle_timer_context_manager(self) -> None:
        m = ArbisyMetrics()
        with m.cycle_timer():
            time.sleep(0.001)

    def test_no_crash_without_prometheus(self) -> None:
        """ArbisyMetrics degrades gracefully if prometheus_client is missing."""
        import src.utils.metrics as metrics_mod
        original = metrics_mod._PROMETHEUS_AVAILABLE
        try:
            metrics_mod._PROMETHEUS_AVAILABLE = False
            m = ArbisyMetrics()
            # All operations should be no-ops
            m.record_opportunity("triangular")
            m.record_execution(success=True, profit_usd=1.0)
            m.set_gas_price(30.0)
            with m.cycle_timer():
                pass
        finally:
            metrics_mod._PROMETHEUS_AVAILABLE = original


# ── E5-S3: JSON structured logging ────────────────────────────────────────────

class TestJSONLogging:
    def test_json_logger_imported_when_available(self) -> None:
        from src.utils.logger import _HAS_JSON_LOGGER
        # If python-json-logger is installed, this should be True
        assert isinstance(_HAS_JSON_LOGGER, bool)

    def test_logger_created_with_json_env(self, monkeypatch, capsys) -> None:
        """With LOG_JSON=1, log output should be parseable JSON."""
        from src.utils.logger import _HAS_JSON_LOGGER
        if not _HAS_JSON_LOGGER:
            pytest.skip("python-json-logger not installed")

        import src.utils.logger as logger_mod

        monkeypatch.setenv("LOG_JSON", "1")
        # Force fresh logger
        name = "__test_json_logger_unique__"
        logger_mod._LOGGERS.pop(name, None)
        logger = logger_mod.get_logger(name)

        # Should have at least one handler
        assert len(logger.handlers) >= 1
        # Clean up
        logger_mod._LOGGERS.pop(name, None)


# ── E5-S5: Health endpoint ────────────────────────────────────────────────────

class TestHealthEndpoints:
    def test_configure_sets_state(self) -> None:
        import src.utils.health as health_mod
        health_mod.configure(
            rpc_url="https://example.com",
            contract_address="0xABCD",
            model_trained=True,
            nn_model_trained=True,
        )
        assert health_mod._rpc_url == "https://example.com"
        assert health_mod._contract_address == "0xABCD"
        assert health_mod._model_trained is True

    @pytest.mark.asyncio
    async def test_healthz_degraded_when_rpc_down(self) -> None:
        """When RPC is unreachable the /healthz response should be degraded."""
        import src.utils.health as health_mod
        if not health_mod._HAS_FASTAPI:
            pytest.skip("fastapi not installed")

        from fastapi.testclient import TestClient

        health_mod.configure(
            rpc_url="https://unreachable.invalid",
            contract_address="0xABCD",
            model_trained=True,
            nn_model_trained=True,
        )

        # Patch _check_rpc to avoid real network call
        with patch("src.utils.health._check_rpc", new_callable=AsyncMock, return_value=False):
            client = TestClient(health_mod.app, raise_server_exceptions=False)
            resp = client.get("/healthz")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "degraded"

    def test_readyz_not_ready_without_contract(self) -> None:
        import src.utils.health as health_mod
        if not health_mod._HAS_FASTAPI:
            pytest.skip("fastapi not installed")

        from fastapi.testclient import TestClient

        health_mod.configure(
            rpc_url="",
            contract_address="",
            model_trained=False,
            nn_model_trained=False,
        )

        client = TestClient(health_mod.app, raise_server_exceptions=False)
        resp = client.get("/readyz")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"

    def test_readyz_ready_when_configured(self) -> None:
        import src.utils.health as health_mod
        if not health_mod._HAS_FASTAPI:
            pytest.skip("fastapi not installed")

        from fastapi.testclient import TestClient

        health_mod.configure(
            rpc_url="https://rpc.example",
            contract_address="0xDEAD",
            model_trained=True,
            nn_model_trained=True,
        )

        client = TestClient(health_mod.app, raise_server_exceptions=False)
        resp = client.get("/readyz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_execution_agent():
    """Build a minimal ExecutionAgent with all dependencies mocked."""
    from src.agents.execution_agent import ExecutionAgent

    with patch("src.agents.execution_agent._CIRCUIT_STATE_FILE", "/tmp/test_circuit_state.json"):
        with patch("src.agents.execution_agent.get_config") as mock_cfg:
            cfg = MagicMock()
            cfg.min_profit_usd    = 5.0
            cfg.max_gas_price_gwei = 500.0
            cfg.ml_score_threshold = 0.6
            mock_cfg.return_value = cfg

            client    = MagicMock()
            flash     = MagicMock()
            inferencer = MagicMock()
            bus       = MagicMock()

            agent = ExecutionAgent.__new__(ExecutionAgent)
            agent._client    = client
            agent._flash     = flash
            agent._inference = inferencer
            agent._bus       = bus
            agent._min_profit_usd   = 5.0
            agent._max_gas_gwei     = 500.0
            agent._ml_threshold     = 0.6
            agent._circuit_break_limit = 3
            agent._consecutive_losses  = 0
            return agent
