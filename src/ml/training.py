"""
Model training pipeline.

Generates synthetic training data (for bootstrapping), or loads real
historical trade logs, and retrains the ArbitrageModel.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.ml.feature_engineering import N_FEATURES, FEATURE_NAMES
from src.ml.model import ArbitrageModel
from src.ml.neural_model import NeuralArbModel
from src.utils.logger import get_logger

log = get_logger(__name__)

DATA_DIR = Path("data")
HISTORY_FILE = DATA_DIR / "trade_history.jsonl"


def generate_synthetic_data(n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic training data for bootstrapping the model.

    Each sample represents one arbitrage opportunity.  The target is
    net profit in USD, which is a function of spread, loan size, gas, etc.
    This gives the model a reasonable prior before real data is available.
    """
    rng = np.random.default_rng(42)

    spread_pct         = rng.uniform(0.01, 2.0, n_samples)
    profit_pct         = np.maximum(spread_pct - rng.uniform(0.05, 0.5, n_samples), 0)
    loan_usd_log       = rng.uniform(3.0, 6.0, n_samples)      # $1k–$1M
    gas_cost_usd       = rng.uniform(0.5, 5.0, n_samples)
    liquidity_log      = rng.uniform(3.0, 8.0, n_samples)
    hist_sr            = rng.uniform(0.3, 0.9, n_samples)
    slippage_pct       = rng.uniform(0.01, 0.5, n_samples)
    num_hops           = rng.integers(2, 5, n_samples).astype(float)
    is_triangular      = rng.integers(0, 2, n_samples).astype(float)
    fee_bps_sum        = rng.uniform(10, 100, n_samples)
    spread_fee_ratio   = spread_pct / (fee_bps_sum / 100.0 + 0.05 + 1e-9)
    liq_balance        = rng.uniform(0.3, 1.0, n_samples)
    block_util         = rng.uniform(20, 95, n_samples)
    tod_sin            = np.sin(2 * np.pi * rng.uniform(0, 24, n_samples) / 24.0)
    tod_cos            = np.cos(2 * np.pi * rng.uniform(0, 24, n_samples) / 24.0)

    X = np.column_stack([
        spread_pct, profit_pct, loan_usd_log, gas_cost_usd,
        liquidity_log, hist_sr, slippage_pct, num_hops,
        is_triangular, fee_bps_sum, spread_fee_ratio,
        liq_balance, block_util, tod_sin, tod_cos,
    ])

    assert X.shape == (n_samples, N_FEATURES), f"Shape mismatch: {X.shape}"

    # Ground truth: loan * profit% - gas - flash premium
    loan_usd = 10.0 ** loan_usd_log
    y = loan_usd * (profit_pct / 100.0) - gas_cost_usd - (loan_usd * 0.0005)
    # Add noise
    y += rng.normal(0, 1.0, n_samples)

    return X, y


def load_history(path: Optional[Path] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load real historical trade data from a JSONL file.

    Each line must be a JSON object with keys matching FEATURE_NAMES plus
    a `net_profit_usd` target field.
    """
    fpath = path or HISTORY_FILE
    if not fpath.exists():
        return None, None

    rows: List[List[float]] = []
    targets: List[float] = []

    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                row = [float(record.get(feat, 0.0)) for feat in FEATURE_NAMES]
                rows.append(row)
                targets.append(float(record["net_profit_usd"]))
            except (KeyError, ValueError) as exc:
                log.warning("Skipping malformed history record: %s", exc)

    if not rows:
        return None, None

    return np.array(rows), np.array(targets)


def append_to_history(features: np.ndarray, net_profit_usd: float) -> None:
    """Append one trade result to the history file."""
    DATA_DIR.mkdir(exist_ok=True)
    record = {name: float(features[i]) for i, name in enumerate(FEATURE_NAMES)}
    record["net_profit_usd"] = net_profit_usd
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def train(
    model_path: Optional[str] = None,
    use_real_data: bool = True,
    synthetic_samples: int = 2000,
) -> Tuple[ArbitrageModel, Dict[str, float]]:
    """
    Full GBR training pipeline.

    1. Load real trade history (if available and `use_real_data=True`).
    2. Supplement / fallback to synthetic data.
    3. Fit the ArbitrageModel.
    4. Save to disk.
    5. Return (model, metrics).
    """
    X_real, y_real = load_history() if use_real_data else (None, None)
    X_syn, y_syn   = generate_synthetic_data(synthetic_samples)

    if X_real is not None and y_real is not None:
        X = np.vstack([X_real, X_syn])
        y = np.concatenate([y_real, y_syn])
        log.info("Training on %d real + %d synthetic samples", len(X_real), len(X_syn))
    else:
        X, y = X_syn, y_syn
        log.info("No real data found — training on %d synthetic samples", len(X_syn))

    model = ArbitrageModel(model_path=model_path)
    metrics = model.fit(X, y, validate=True)
    saved = model.save()

    log.info("Training complete. Metrics: %s → saved to %s", metrics, saved)
    return model, metrics


def train_nn(
    model_path: Optional[str] = None,
    use_real_data: bool = True,
    synthetic_samples: int = 2000,
) -> Tuple[NeuralArbModel, Dict[str, float]]:
    """
    Full neural network training pipeline.

    Uses the same data as ``train()`` so the GBR and NN models are trained
    on identical samples and can be fairly compared or combined in an
    ensemble.

    1. Load real trade history (if available and ``use_real_data=True``).
    2. Supplement / fallback to synthetic data.
    3. Fit the ``NeuralArbModel`` (sklearn MLPRegressor).
    4. Save to disk.
    5. Return (model, metrics).
    """
    X_real, y_real = load_history() if use_real_data else (None, None)
    X_syn, y_syn   = generate_synthetic_data(synthetic_samples)

    if X_real is not None and y_real is not None:
        X = np.vstack([X_real, X_syn])
        y = np.concatenate([y_real, y_syn])
        log.info(
            "NeuralArbModel: training on %d real + %d synthetic samples",
            len(X_real), len(X_syn),
        )
    else:
        X, y = X_syn, y_syn
        log.info(
            "NeuralArbModel: no real data — training on %d synthetic samples",
            len(X_syn),
        )

    nn_model = NeuralArbModel(model_path=model_path)
    metrics  = nn_model.fit(X, y, validate=True)
    saved    = nn_model.save()

    log.info("NeuralArbModel training complete. Metrics: %s → saved to %s", metrics, saved)
    return nn_model, metrics


if __name__ == "__main__":
    train()
    train_nn()
