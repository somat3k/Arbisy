"""
PostgreSQL persistence layer for execution logs (E3-S1).

Stores every opportunity evaluation and its execution outcome so that the ML
model can be retrained on real historical data.

Schema
------
The ``executions`` table captures all seven ML feature columns together with
the outcome label ``net_profit_usd``:

    CREATE TABLE IF NOT EXISTS executions (
        id                    BIGSERIAL PRIMARY KEY,
        recorded_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        arb_type              TEXT,
        price_spread_pct      DOUBLE PRECISION,
        gas_cost_usd          DOUBLE PRECISION,
        liquidity_depth       DOUBLE PRECISION,
        historical_success_rate DOUBLE PRECISION,
        slippage_estimate     DOUBLE PRECISION,
        block_utilization     DOUBLE PRECISION,
        time_since_last_trade DOUBLE PRECISION,
        ml_score              DOUBLE PRECISION,
        executed              BOOLEAN,
        success               BOOLEAN,
        net_profit_usd        DOUBLE PRECISION,
        tx_hash               TEXT
    );

Requires the ``asyncpg`` package:  pip install asyncpg

When ``asyncpg`` is not installed or DATABASE_URL is empty the module
degrades gracefully — all public functions become silent no-ops so the
rest of the system continues to work without a database.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Optional asyncpg dependency ───────────────────────────────────────────────
try:
    import asyncpg  # type: ignore
    _HAS_ASYNCPG = True
except ImportError:
    asyncpg = None  # type: ignore
    _HAS_ASYNCPG = False

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS executions (
    id                      BIGSERIAL PRIMARY KEY,
    recorded_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    arb_type                TEXT,
    price_spread_pct        DOUBLE PRECISION,
    gas_cost_usd            DOUBLE PRECISION,
    liquidity_depth         DOUBLE PRECISION,
    historical_success_rate DOUBLE PRECISION,
    slippage_estimate       DOUBLE PRECISION,
    block_utilization       DOUBLE PRECISION,
    time_since_last_trade   DOUBLE PRECISION,
    ml_score                DOUBLE PRECISION,
    executed                BOOLEAN,
    success                 BOOLEAN,
    net_profit_usd          DOUBLE PRECISION,
    tx_hash                 TEXT
);
"""

_INSERT_SQL = """
INSERT INTO executions (
    arb_type,
    price_spread_pct,
    gas_cost_usd,
    liquidity_depth,
    historical_success_rate,
    slippage_estimate,
    block_utilization,
    time_since_last_trade,
    ml_score,
    executed,
    success,
    net_profit_usd,
    tx_hash
) VALUES (
    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
);
"""

_SELECT_TRAINING_SQL = """
SELECT
    price_spread_pct,
    gas_cost_usd,
    liquidity_depth,
    historical_success_rate,
    slippage_estimate,
    block_utilization,
    time_since_last_trade,
    net_profit_usd
FROM executions
WHERE executed = TRUE
  AND recorded_at >= NOW() - INTERVAL '90 days'
ORDER BY recorded_at DESC;
"""


class ExecutionDB:
    """
    Async PostgreSQL client for persisting execution logs.

    Parameters
    ----------
    database_url:  PostgreSQL DSN.  If empty or None, operates as a no-op.
    """

    def __init__(self, database_url: Optional[str] = None) -> None:
        cfg = get_config()
        self._url = database_url or cfg.database_url
        self._pool: Optional[Any] = None
        self._enabled = bool(self._url and _HAS_ASYNCPG)
        if not _HAS_ASYNCPG:
            log.debug("asyncpg not installed — DB persistence disabled")
        elif not self._url:
            log.debug("DATABASE_URL not set — DB persistence disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def connect(self) -> None:
        """Create the connection pool and ensure the schema exists."""
        if not self._enabled:
            return
        try:
            self._pool = await asyncpg.create_pool(self._url, min_size=1, max_size=5)
            async with self._pool.acquire() as conn:
                await conn.execute(_CREATE_TABLE_SQL)
            log.info("ExecutionDB: connected to PostgreSQL")
        except Exception as exc:
            log.error("ExecutionDB: connection failed — %s", exc)
            self._enabled = False

    async def close(self) -> None:
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def log_execution(
        self,
        arb_type: str,
        price_spread_pct: float,
        gas_cost_usd: float,
        liquidity_depth: float,
        historical_success_rate: float,
        slippage_estimate: float,
        block_utilization: float,
        time_since_last_trade: float,
        ml_score: float,
        executed: bool,
        success: bool,
        net_profit_usd: float,
        tx_hash: str = "",
    ) -> None:
        """
        Persist one execution record.

        All seven ML feature columns plus outcome and metadata are stored.
        Missing pool or gas data should be passed as 0.0.
        """
        if not self._enabled or self._pool is None:
            return
        try:
            async with self._pool.acquire() as conn:
                await conn.execute(
                    _INSERT_SQL,
                    arb_type,
                    price_spread_pct,
                    gas_cost_usd,
                    liquidity_depth,
                    historical_success_rate,
                    slippage_estimate,
                    block_utilization,
                    time_since_last_trade,
                    ml_score,
                    executed,
                    success,
                    net_profit_usd,
                    tx_hash,
                )
        except Exception as exc:
            log.warning("ExecutionDB.log_execution failed: %s", exc)

    async def fetch_training_rows(
        self,
        limit: Optional[int] = None,
    ) -> List[Dict[str, float]]:
        """
        Fetch the last 90 days of executed trades as a list of feature dicts.

        Returns an empty list when the DB is unavailable.
        """
        if not self._enabled or self._pool is None:
            return []
        sql = _SELECT_TRAINING_SQL
        if limit:
            sql = sql.rstrip(";") + f"\nLIMIT {int(limit)};"
        try:
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql)
            return [dict(r) for r in rows]
        except Exception as exc:
            log.warning("ExecutionDB.fetch_training_rows failed: %s", exc)
            return []


# ── Module-level singleton ────────────────────────────────────────────────────
_db: Optional[ExecutionDB] = None


def get_db() -> ExecutionDB:
    """Return (or create) the global ExecutionDB singleton."""
    global _db
    if _db is None:
        _db = ExecutionDB()
    return _db
