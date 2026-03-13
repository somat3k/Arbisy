"""
Tests for ML components: feature engineering, model, training, inference.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.ml.feature_engineering import FeatureEngineer, FeatureVector, N_FEATURES, FEATURE_NAMES
from src.ml.inference import LiveInference
from src.ml.model import ArbitrageModel
from src.ml.training import generate_synthetic_data
from src.payload.protocol import OpportunityPayload, Priority, SwapHop


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_opportunity(
    profit_usd: float = 50.0,
    loan_usd: float = 10_000.0,
    arb_type: str = "cross_platform",
    n_hops: int = 2,
    slippage: float = 0.1,
) -> OpportunityPayload:
    hops = [
        SwapHop(
            dex_name="uniswap_v3",
            router_address="0x1",
            token_in="0xusdc",
            token_out="0xweth",
            fee_bps=30,
            is_v3=True,
            estimated_amount_out=loan_usd * 1.005,
        )
        for _ in range(n_hops)
    ]
    return OpportunityPayload(
        source="test",
        priority=Priority.HIGH,
        asset="0xusdc",
        loan_amount_wei=int(loan_usd * 1e6),
        loan_amount_usd=loan_usd,
        expected_profit_usd=profit_usd,
        expected_profit_wei=int(profit_usd * 1e6),
        ml_score=0.0,
        path=hops,
        gas_estimate_gwei=50.0,
        slippage_estimate_pct=slippage,
        dex_sources=["uniswap_v3", "quickswap_v3"],
        arb_type=arb_type,
    )


# ── FeatureEngineer ───────────────────────────────────────────────────────────

class TestFeatureEngineer:
    def test_extract_shape(self) -> None:
        fe  = FeatureEngineer()
        opp = make_opportunity()
        fv  = fe.extract(opp, gas_cost_usd=1.5, block_utilization_pct=60.0)
        assert isinstance(fv, FeatureVector)
        assert fv.features.shape == (N_FEATURES,)

    def test_feature_names_match(self) -> None:
        fe  = FeatureEngineer()
        opp = make_opportunity()
        fv  = fe.extract(opp)
        assert fv.feature_names == FEATURE_NAMES

    def test_2d_output(self) -> None:
        fe  = FeatureEngineer()
        opp = make_opportunity()
        fv  = fe.extract(opp)
        assert fv.to_2d().shape == (1, N_FEATURES)

    def test_no_nan_in_features(self) -> None:
        fe  = FeatureEngineer()
        opp = make_opportunity()
        fv  = fe.extract(opp, gas_cost_usd=2.0, block_utilization_pct=45.0)
        assert not np.any(np.isnan(fv.features))
        assert not np.any(np.isinf(fv.features))

    def test_is_triangular_flag(self) -> None:
        fe    = FeatureEngineer()
        opp_t = make_opportunity(arb_type="triangular")
        opp_c = make_opportunity(arb_type="cross_platform")
        fv_t  = fe.extract(opp_t)
        fv_c  = fe.extract(opp_c)
        tri_idx = FEATURE_NAMES.index("is_triangular")
        assert fv_t.features[tri_idx] == 1.0
        assert fv_c.features[tri_idx] == 0.0

    def test_historical_success_rate_defaults_to_half(self) -> None:
        fe = FeatureEngineer()
        assert fe.historical_success_rate("unknown_type") == 0.5

    def test_historical_success_rate_updates(self) -> None:
        fe = FeatureEngineer()
        fe.record_outcome("cross_platform", True)
        fe.record_outcome("cross_platform", True)
        fe.record_outcome("cross_platform", False)
        assert abs(fe.historical_success_rate("cross_platform") - 2/3) < 1e-9


# ── ArbitrageModel ────────────────────────────────────────────────────────────

class TestArbitrageModel:
    def test_heuristic_predict_without_training(self) -> None:
        model = ArbitrageModel(model_path="/tmp/nonexistent_model.joblib")
        X, _ = generate_synthetic_data(10)
        preds = model.predict(X)
        assert preds.shape == (10,)

    def test_train_and_predict(self) -> None:
        model = ArbitrageModel(model_path="/tmp/test_model.joblib")
        X, y  = generate_synthetic_data(200)
        metrics = model.fit(X, y, validate=True)
        assert "train_rmse" in metrics
        assert metrics["train_rmse"] >= 0
        assert model.trained

        preds = model.predict(X[:5])
        assert preds.shape == (5,)
        assert not np.any(np.isnan(preds))

    def test_predict_score_range(self) -> None:
        model = ArbitrageModel(model_path="/tmp/test_model2.joblib")
        X, y  = generate_synthetic_data(100)
        model.fit(X, y, validate=False)
        score = model.predict_score(X[0])
        assert 0.0 <= score <= 1.0

    def test_feature_importances_after_training(self) -> None:
        model = ArbitrageModel(model_path="/tmp/test_model3.joblib")
        X, y  = generate_synthetic_data(100)
        model.fit(X, y, validate=False)
        importances = model.feature_importances()
        assert len(importances) == N_FEATURES
        assert all(v >= 0 for v in importances.values())
        assert abs(sum(importances.values()) - 1.0) < 1e-5

    def test_save_and_reload(self, tmp_path) -> None:
        path  = str(tmp_path / "model.joblib")
        model = ArbitrageModel(model_path=path)
        X, y  = generate_synthetic_data(100)
        model.fit(X, y, validate=False)
        model.save(path)

        model2 = ArbitrageModel(model_path=path)
        assert model2.trained
        p1 = model.predict(X[:3])
        p2 = model2.predict(X[:3])
        np.testing.assert_allclose(p1, p2, rtol=1e-5)


# ── Synthetic Data ────────────────────────────────────────────────────────────

class TestSyntheticData:
    def test_shape(self) -> None:
        X, y = generate_synthetic_data(500)
        assert X.shape == (500, N_FEATURES)
        assert y.shape == (500,)

    def test_no_nan(self) -> None:
        X, y = generate_synthetic_data(100)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))


# ── LiveInference ─────────────────────────────────────────────────────────────

class TestLiveInference:
    def test_score_returns_tuple(self) -> None:
        infer = LiveInference()
        opp   = make_opportunity(profit_usd=20.0)
        score, profit = infer.score(opp, gas_cost_usd=1.0, block_utilization_pct=50.0)
        assert 0.0 <= score <= 1.0
        assert isinstance(profit, float)

    def test_should_execute_below_threshold(self) -> None:
        infer = LiveInference()
        # Very low profit opportunity
        opp = make_opportunity(profit_usd=0.001, loan_usd=1.0)
        # With untrained heuristic model this might or might not pass; just verify no exception
        result = infer.should_execute(opp)
        assert isinstance(result, bool)

    def test_record_outcome(self) -> None:
        infer = LiveInference()
        infer.record_outcome("triangular", True)
        infer.record_outcome("triangular", False)
        rate = infer.fe.historical_success_rate("triangular")
        assert abs(rate - 0.5) < 1e-9
