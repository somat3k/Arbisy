"""
Neural network arbitrage scoring model.

Implements ``NeuralArbModel``: an sklearn ``MLPRegressor`` (multi-layer
perceptron) that predicts net USD profit for an arbitrage opportunity.

Architecture
------------
- 3 hidden layers: 128 → 64 → 32 neurons
- ReLU activation, Adam optimiser
- Early stopping (patience 20 iterations) with 10 % validation split
- ``StandardScaler`` normalisation (fitted on training data)

The model uses the same 15-element feature vector as ``ArbitrageModel``
(GBR) so both models are interchangeable and can be combined in an
ensemble.  Use ``path_to_features()`` from ``array_builder`` to convert
a ``PathArray`` directly to a scoreable feature vector.

No torch, tensorflow, onnx, or onnxruntime — sklearn only.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, Optional

import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.ml.feature_engineering import FEATURE_NAMES, N_FEATURES
from src.utils.logger import get_logger

log = get_logger(__name__)

# Default model file (can be overridden via Config or constructor)
DEFAULT_NN_MODEL_PATH = "models/neural_arb_model.joblib"

# ── MLPRegressor hyper-parameters ─────────────────────────────────────────────
NN_MODEL_PARAMS: Dict[str, Any] = {
    "hidden_layer_sizes":  (128, 64, 32),
    "activation":          "relu",
    "solver":              "adam",
    "learning_rate_init":  0.001,
    "max_iter":            500,
    "early_stopping":      True,
    "validation_fraction": 0.1,
    "n_iter_no_change":    20,
    "random_state":        42,
    "verbose":             False,
}


class NeuralArbModel:
    """
    sklearn MLPRegressor-based neural network for arbitrage profit prediction.

    Usage
    -----
    model = NeuralArbModel()
    X, y  = generate_synthetic_data(2000)
    model.fit(X, y)
    score = model.predict_score(feature_vector)   # float in [0, 1]

    Attributes
    ----------
    model:   The underlying sklearn MLPRegressor.
    scaler:  StandardScaler fitted on training data.
    trained: Whether the model has been fitted.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        self._model_path = model_path or DEFAULT_NN_MODEL_PATH
        self.model: MLPRegressor   = MLPRegressor(**NN_MODEL_PARAMS)
        self.scaler: StandardScaler = StandardScaler()
        self.trained: bool         = False
        self._load_if_exists()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load_if_exists(self) -> None:
        if os.path.exists(self._model_path):
            try:
                payload      = joblib.load(self._model_path)
                self.model   = payload["model"]
                self.scaler  = payload["scaler"]
                self.trained = True
                log.info("NeuralArbModel loaded from %s", self._model_path)
            except Exception as exc:
                log.warning(
                    "Failed to load NN model from %s: %s", self._model_path, exc
                )

    def reload(self) -> None:
        """Reload the persisted model from disk, replacing current weights."""
        self._load_if_exists()

    def save(self, path: Optional[str] = None) -> str:
        """Persist model + scaler to disk; return the saved path."""
        save_path = path or self._model_path
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, save_path)
        log.info("NeuralArbModel saved to %s", save_path)
        return save_path

    # ── Training ──────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validate: bool = True,
    ) -> Dict[str, float]:
        """
        Train the neural network on feature matrix ``X`` and target ``y``.

        Parameters
        ----------
        X:        Shape (n_samples, N_FEATURES).
        y:        Net profit in USD for each sample.
        validate: If True, hold out 20 % for validation metrics.

        Returns
        -------
        Dict with ``train_rmse``, ``val_rmse`` (if validate), ``train_r2``.
        """
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split

        assert X.shape[1] == N_FEATURES, (
            f"Expected {N_FEATURES} features, got {X.shape[1]}"
        )

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
        metrics["train_rmse"] = float(
            np.sqrt(mean_squared_error(y_tr, y_pred_tr))
        )
        metrics["train_r2"] = float(r2_score(y_tr, y_pred_tr))

        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            y_pred_val   = self.model.predict(X_val_scaled)
            metrics["val_rmse"] = float(
                np.sqrt(mean_squared_error(y_val, y_pred_val))
            )
            metrics["val_r2"] = float(r2_score(y_val, y_pred_val))

        log.info("NeuralArbModel trained — %s", metrics)
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
        Return a 0–1 score for a single feature vector.

        Uses a sigmoid centred at $5 net profit (scale 10) — identical to
        ``ArbitrageModel.predict_score`` so both models can be compared
        directly.
        """
        profit = float(self.predict(X.reshape(1, -1))[0])
        score  = 1.0 / (1.0 + math.exp(-(profit - 5.0) / 10.0))
        return round(score, 4)

    @staticmethod
    def _heuristic_predict(X: np.ndarray) -> np.ndarray:
        """Simple heuristic when no trained model is available."""
        spread_col = FEATURE_NAMES.index("price_spread_pct")
        gas_col    = FEATURE_NAMES.index("gas_cost_usd")
        loan_col   = FEATURE_NAMES.index("loan_amount_usd_log")

        spread   = X[:, spread_col]
        gas      = X[:, gas_col]
        loan_usd = 10.0 ** X[:, loan_col]
        return loan_usd * (spread / 100.0) - gas - (loan_usd * 0.0005)
