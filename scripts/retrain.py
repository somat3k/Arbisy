#!/usr/bin/env python
"""
Weekly model retraining script (E3-S4).

Loads real execution data via the ETL pipeline, merges it with synthetic
bootstrap data when real rows are scarce, retrains the GBR and NN models,
and replaces the persisted joblib files atomically.

Intended to be run weekly via a cron job or systemd timer:

    # crontab entry — runs every Sunday at 02:00
    0 2 * * 0 /usr/bin/python /path/to/scripts/retrain.py >> /var/log/arbisy_retrain.log 2>&1

Usage
-----
    python scripts/retrain.py [--synthetic-only] [--samples N]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

# Ensure the project root is on sys.path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.ml.etl import load_training_data, merge_with_synthetic
from src.ml.model import EnsembleModel
from src.ml.neural_model import NeuralArbModel
from src.ml.training import train as train_gbr, train_nn as train_nn_model
from src.utils.config import get_config
from src.utils.db import ExecutionDB
from src.utils.logger import get_logger

log = get_logger("retrain")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain Arbisy ML models")
    parser.add_argument(
        "--synthetic-only",
        action="store_true",
        help="Skip PostgreSQL fetch and use only synthetic data.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of synthetic samples when real data is insufficient (default: 2000).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Override path for GBR model joblib.",
    )
    parser.add_argument(
        "--nn-model-path",
        type=str,
        default="",
        help="Override path for NN model joblib.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    cfg = get_config()
    gbr_path = args.model_path or cfg.model_path
    nn_path  = args.nn_model_path or cfg.nn_model_path

    os.makedirs(os.path.dirname(gbr_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(nn_path)  or ".", exist_ok=True)

    t_start = time.time()

    # ── Load training data ────────────────────────────────────────────────────
    if args.synthetic_only:
        log.info("Synthetic-only mode — skipping database fetch")
        from src.ml.training import generate_synthetic_data
        X, y = generate_synthetic_data(n_samples=args.samples)
    else:
        db = ExecutionDB()
        await db.connect()
        try:
            real_X, real_y = await load_training_data(db=db, min_rows=10)
        finally:
            await db.close()

        if real_X.shape[0] == 0:
            log.warning(
                "No real data available — falling back to %d synthetic samples",
                args.samples,
            )
            from src.ml.training import generate_synthetic_data
            X, y = generate_synthetic_data(n_samples=args.samples)
        else:
            X, y = merge_with_synthetic(
                real_X, real_y, synthetic_samples=args.samples // 2
            )

    log.info("Training dataset: %d rows, %d features", X.shape[0], X.shape[1])

    # ── Train GBR EnsembleModel ───────────────────────────────────────────────
    log.info("Training GBR EnsembleModel → %s", gbr_path)
    ensemble = EnsembleModel(model_path=gbr_path)
    metrics = ensemble.fit(X, y, validate=True)
    ensemble.save(gbr_path)
    log.info(
        "GBR done — train_rmse=%.4f  val_rmse=%.4f  R²=%.4f",
        metrics.get("train_rmse", 0),
        metrics.get("val_rmse", 0),
        metrics.get("val_r2", 0),
    )

    # ── Train Neural Network model ────────────────────────────────────────────
    log.info("Training NeuralArbModel → %s", nn_path)
    nn = NeuralArbModel(model_path=nn_path)
    nn_metrics = nn.fit(X, y, validate=True)
    nn.save(nn_path)
    log.info(
        "NN done  — train_rmse=%.4f  val_rmse=%.4f",
        nn_metrics.get("train_rmse", 0),
        nn_metrics.get("val_rmse", 0),
    )

    elapsed = time.time() - t_start
    log.info("Retraining complete in %.1fs", elapsed)


def main() -> None:
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
