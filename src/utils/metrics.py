"""
Prometheus metrics for Arbisy (E5-S1).

Exposes the following metrics on a configurable HTTP port:

  arbisy_cycle_duration_ms       Histogram  Duration of each Find→Simulate→Release cycle
  arbisy_opportunities_found     Counter    Opportunities found per arb_type
  arbisy_executions_total        Counter    Flash-loan executions (labelled success/failure)
  arbisy_profit_usd_total        Counter    Cumulative net profit in USD
  arbisy_ml_score                Histogram  ML score distribution at execution time
  arbisy_gas_price_gwei          Gauge      Current gas price in gwei

Usage
-----
Start the metrics server once at orchestrator startup::

    from src.utils.metrics import ArbisyMetrics
    metrics = ArbisyMetrics()
    metrics.start_server(port=8000)          # serves /metrics on port 8000

Then record observations in the hot path::

    with metrics.cycle_timer():
        ...
    metrics.record_opportunity("triangular")
    metrics.record_execution(success=True, profit_usd=12.5, ml_score=0.82)
    metrics.set_gas_price(45.0)
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator, Optional

from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        start_http_server,
    )
    _PROMETHEUS_AVAILABLE = True
    _CounterType  = Counter
    _GaugeType    = Gauge
    _HistogramType = Histogram
except ImportError:  # pragma: no cover
    _PROMETHEUS_AVAILABLE = False
    log.warning("prometheus_client not installed — metrics disabled")
    _CounterType  = Any  # type: ignore[assignment,misc]
    _GaugeType    = Any  # type: ignore[assignment,misc]
    _HistogramType = Any  # type: ignore[assignment,misc]


class ArbisyMetrics:
    """
    Central metrics registry for the Arbisy arbitrage system.

    All metric objects are created once at class initialisation time (module
    level) and shared across all instances.  This avoids
    ``ValueError: Duplicated timeseries in CollectorRegistry`` when the class
    is instantiated more than once in the same process (e.g. in tests).

    If ``prometheus_client`` is not installed the class still works but
    records nothing (graceful degradation).
    """

    # ── Class-level metric singletons (created once) ──────────────────────────
    _metrics_initialised: bool = False
    _cycle_duration: Optional["_HistogramType"] = None
    _opportunities: Optional["_CounterType"] = None
    _executions: Optional["_CounterType"] = None
    _profit: Optional["_CounterType"] = None
    _ml_score: Optional["_HistogramType"] = None
    _gas_price: Optional["_GaugeType"] = None

    def __init__(self) -> None:
        if not _PROMETHEUS_AVAILABLE:
            return
        if not ArbisyMetrics._metrics_initialised:
            ArbisyMetrics._cycle_duration = Histogram(
                "arbisy_cycle_duration_ms",
                "Duration of one Find→Simulate→Release cycle in milliseconds",
                buckets=[50, 100, 250, 500, 1000, 2500, 5000, 10000],
            )
            ArbisyMetrics._opportunities = Counter(
                "arbisy_opportunities_found_total",
                "Number of arbitrage opportunities detected",
                labelnames=["arb_type"],
            )
            ArbisyMetrics._executions = Counter(
                "arbisy_executions_total",
                "Number of flash-loan execution attempts",
                labelnames=["status"],
            )
            ArbisyMetrics._profit = Counter(
                "arbisy_profit_usd_total",
                "Cumulative net profit in USD",
            )
            ArbisyMetrics._ml_score = Histogram(
                "arbisy_ml_score",
                "ML score distribution at execution decision time",
                buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            )
            ArbisyMetrics._gas_price = Gauge(
                "arbisy_gas_price_gwei",
                "Current network gas price in gwei",
            )
            ArbisyMetrics._metrics_initialised = True

    # ── Server ────────────────────────────────────────────────────────────────

    def start_server(self, port: int = 8000) -> None:
        """Start the Prometheus HTTP server on the given port."""
        if not _PROMETHEUS_AVAILABLE:
            log.warning("prometheus_client not installed — /metrics server not started")
            return
        start_http_server(port)
        log.info("Prometheus /metrics available on :%d", port)

    # ── Cycle timer ───────────────────────────────────────────────────────────

    @contextmanager
    def cycle_timer(self) -> Iterator[None]:
        """Context manager that records cycle duration in milliseconds."""
        if not _PROMETHEUS_AVAILABLE:
            yield
            return
        start = time.monotonic()
        try:
            yield
        finally:
            elapsed_ms = (time.monotonic() - start) * 1000.0
            self._cycle_duration.observe(elapsed_ms)

    # ── Individual metric helpers ─────────────────────────────────────────────

    def record_opportunity(self, arb_type: str, count: int = 1) -> None:
        """Increment the opportunities counter for a given arb type."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self._opportunities.labels(arb_type=arb_type).inc(count)

    def record_execution(
        self,
        success: bool,
        profit_usd: float = 0.0,
        ml_score: float = 0.0,
    ) -> None:
        """Record one execution attempt."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self._executions.labels(status="success" if success else "failure").inc()
        if success and profit_usd > 0:
            self._profit.inc(profit_usd)
        if ml_score > 0:
            self._ml_score.observe(ml_score)

    def set_gas_price(self, gwei: float) -> None:
        """Update the current gas price gauge."""
        if not _PROMETHEUS_AVAILABLE:
            return
        self._gas_price.set(gwei)
