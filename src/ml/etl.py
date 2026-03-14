"""
ETL pipeline — extracts features from the PostgreSQL execution log (E3-S2).

Converts rows from the ``executions`` table into a (X, y) NumPy dataset
ready for training the GradientBoostingRegressor model.

The database stores the seven core observable ML features.  When building
training arrays the ETL maps each DB column to its position within the full
FEATURE_NAMES vector (15 features); unmapped positions are left as zero so
the trained model always receives a correctly shaped input.

Usage
-----
    X, y = await load_training_data()
    model.fit(X, y)

    # Or run synchronously in a script:
    X, y = asyncio.run(load_training_data())
"""

from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ml.feature_engineering import FEATURE_NAMES, N_FEATURES
from src.utils.db import ExecutionDB, get_db
from src.utils.logger import get_logger

log = get_logger(__name__)

# Mapping from DB column name → position in FEATURE_NAMES
# DB stores the seven core observable features; derived features default to 0
_DB_TO_FEATURE_IDX = {
    "price_spread_pct":      FEATURE_NAMES.index("price_spread_pct"),
    "gas_cost_usd":          FEATURE_NAMES.index("gas_cost_usd"),
    "historical_success_rate": FEATURE_NAMES.index("historical_success_rate"),
    "slippage_estimate":     FEATURE_NAMES.index("slippage_estimate_pct"),
    "block_utilization":     FEATURE_NAMES.index("block_utilization_pct"),
}

_LABEL_COLUMN = "net_profit_usd"

# All DB feature columns stored in the executions table
_DB_FEATURE_COLUMNS = list(_DB_TO_FEATURE_IDX.keys())


async def load_training_data(
    db: Optional[ExecutionDB] = None,
    min_rows: int = 10,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load real execution history from PostgreSQL and return (X, y) arrays.

    Falls back to an empty (0, N_FEATURES) array when the DB is unavailable
    or contains fewer than ``min_rows`` executed trades.

    Parameters
    ----------
    db:       ExecutionDB instance; uses the global singleton if None.
    min_rows: Minimum number of rows required before returning real data.
              If fewer rows are found, returns empty arrays.
    limit:    Maximum number of rows to fetch (newest first).

    Returns
    -------
    X: np.ndarray of shape (n_samples, N_FEATURES)
    y: np.ndarray of shape (n_samples,)
    """
    db = db or get_db()

    rows = await db.fetch_training_rows(limit=limit)
    if len(rows) < min_rows:
        log.debug(
            "ETL: only %d rows in DB (min=%d) — returning empty dataset",
            len(rows), min_rows,
        )
        return np.empty((0, N_FEATURES)), np.empty(0)

    df = pd.DataFrame(rows)

    # Ensure all expected DB columns are present
    for col in _DB_FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    if _LABEL_COLUMN not in df.columns:
        log.warning("ETL: '%s' column missing — returning empty dataset", _LABEL_COLUMN)
        return np.empty((0, N_FEATURES)), np.empty(0)

    # Drop rows with NaN labels
    df = df.dropna(subset=[_LABEL_COLUMN])
    if df.empty:
        return np.empty((0, N_FEATURES)), np.empty(0)

    n = len(df)
    # Build full-width feature matrix, mapping DB columns to their FEATURE_NAMES positions
    X = np.zeros((n, N_FEATURES), dtype=float)
    for col, idx in _DB_TO_FEATURE_IDX.items():
        X[:, idx] = df[col].fillna(0.0).to_numpy(dtype=float)

    y = df[_LABEL_COLUMN].to_numpy(dtype=float)

    log.info("ETL: loaded %d training rows from DB → shape (%d, %d)", n, n, N_FEATURES)
    return X, y


def load_training_data_sync(
    db: Optional[ExecutionDB] = None,
    min_rows: int = 10,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synchronous wrapper around ``load_training_data`` for use in scripts.
    """
    return asyncio.run(load_training_data(db=db, min_rows=min_rows, limit=limit))


def merge_with_synthetic(
    real_X: np.ndarray,
    real_y: np.ndarray,
    synthetic_samples: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Merge real execution data with synthetic bootstrap data.

    When real data is scarce the synthetic samples pad the dataset so the
    model can still be initialised with reasonable weights.  The real rows
    are kept as-is; synthetic rows augment (not replace) them.

    Parameters
    ----------
    real_X:            Feature matrix from DB (may be empty).
    real_y:            Label vector from DB (may be empty).
    synthetic_samples: Number of synthetic rows to generate.

    Returns
    -------
    Combined (X, y) arrays with real rows first.
    """
    from src.ml.training import generate_synthetic_data  # local import avoids circularity

    syn_X, syn_y = generate_synthetic_data(n_samples=synthetic_samples)

    if real_X.shape[0] == 0:
        return syn_X, syn_y

    X_merged = np.vstack([real_X, syn_X])
    y_merged = np.concatenate([real_y, syn_y])
    log.info(
        "ETL: merged %d real + %d synthetic rows → %d total",
        len(real_X), len(syn_X), len(X_merged),
    )
    return X_merged, y_merged
