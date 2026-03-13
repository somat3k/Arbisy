"""
sklearn-based arbitrage opportunity scoring model.

Uses GradientBoostingRegressor to predict net profit (USD) from feature
vectors.  Falls back to a simple heuristic when no trained model is
available.

No torch, tensorflow, onnx, or onnxruntime dependencies.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from src.ml.feature_engineering import FEATURE_NAMES, N_FEATURES
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

DEFAULT_MODEL_PARAMS: Dict[str, Any] = {
    "n_estimators":    200,
    "max_depth":       4,
    "learning_rate":   0.05,
    "subsample":       0.8,
    "min_samples_split": 5,
    "loss":            "squared_error",
    "random_state":    42,
}


class ArbitrageModel:
    """
    Trained GBR model that scores arbitrage opportunities.

    Attributes
    ----------
    model:   The underlying sklearn estimator.
    scaler:  StandardScaler fitted on training data.
    trained: Whether the model has been fitted.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        cfg = get_config()
        self._model_path = model_path or cfg.model_path
        self.model: GradientBoostingRegressor = GradientBoostingRegressor(
            **DEFAULT_MODEL_PARAMS
        )
        self.scaler: StandardScaler = StandardScaler()
        self.trained: bool = False
        self._load_if_exists()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_if_exists(self) -> None:
        if os.path.exists(self._model_path):
            try:
                payload = joblib.load(self._model_path)
                self.model   = payload["model"]
                self.scaler  = payload["scaler"]
                self.trained = True
                log.info("ArbitrageModel loaded from %s", self._model_path)
            except Exception as exc:
                log.warning("Failed to load model from %s: %s", self._model_path, exc)

    def reload(self) -> None:
        """Reload the persisted model from disk, replacing the current weights."""
        self._load_if_exists()

    def save(self, path: Optional[str] = None) -> str:
        """Persist model + scaler to disk; return the saved path."""
        save_path = path or self._model_path
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, save_path)
        log.info("ArbitrageModel saved to %s", save_path)
        return save_path

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validate: bool = True,
    ) -> Dict[str, float]:
        """
        Train the model on feature matrix X and target vector y.

        Parameters
        ----------
        X:        Shape (n_samples, N_FEATURES).
        y:        Net profit in USD for each sample.
        validate: If True, hold out 20% for validation metrics.

        Returns
        -------
        Dict with train_rmse, val_rmse (if validate), r2.
        """
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        assert X.shape[1] == N_FEATURES, f"Expected {N_FEATURES} features, got {X.shape[1]}"

        metrics: Dict[str, float] = {}

        if validate and len(X) >= 10:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        else:
            X_tr, X_val, y_tr, y_val = X, None, y, None

        X_tr_scaled = self.scaler.fit_transform(X_tr)
        self.model.fit(X_tr_scaled, y_tr)
        self.trained = True

        y_pred_tr = self.model.predict(X_tr_scaled)
        metrics["train_rmse"] = float(np.sqrt(mean_squared_error(y_tr, y_pred_tr)))
        metrics["train_r2"]   = float(r2_score(y_tr, y_pred_tr))

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_pred_val   = self.model.predict(X_val_scaled)
            metrics["val_rmse"] = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
            metrics["val_r2"]   = float(r2_score(y_val, y_pred_val))

        log.info("Model trained — %s", metrics)
        return metrics

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict net profit (USD) for each sample.

        Parameters
        ----------
        X: Shape (n_samples, N_FEATURES) or (N_FEATURES,).

        Returns
        -------
        Predicted net profits, shape (n_samples,).
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        assert X.shape[1] == N_FEATURES

        if not self.trained:
            return self._heuristic_predict(X)

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_score(self, X: np.ndarray) -> float:
        """
        Return a 0-1 score for a single sample (normalised sigmoid of predicted profit).
        """
        profit = float(self.predict(X.reshape(1, -1))[0])
        # Sigmoid centred at $5 profit, scale 10
        import math
        score = 1.0 / (1.0 + math.exp(-(profit - 5.0) / 10.0))
        return round(score, 4)

    @staticmethod
    def _heuristic_predict(X: np.ndarray) -> np.ndarray:
        """
        Simple heuristic when no trained model is available.
        Uses the price-spread and gas-cost features directly.
        """
        spread_col = FEATURE_NAMES.index("price_spread_pct")
        gas_col    = FEATURE_NAMES.index("gas_cost_usd")
        loan_col   = FEATURE_NAMES.index("loan_amount_usd_log")

        spread   = X[:, spread_col]
        gas      = X[:, gas_col]
        loan_log = X[:, loan_col]
        loan_usd = 10.0 ** loan_log

        profit = loan_usd * (spread / 100.0) - gas - (loan_usd * 0.0005)
        return profit

    def feature_importances(self) -> Dict[str, float]:
        """Return feature importances if the model is trained."""
        if not self.trained:
            return {}
        imp = self.model.feature_importances_
        return {name: round(float(v), 6) for name, v in zip(FEATURE_NAMES, imp)}
