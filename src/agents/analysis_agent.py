"""
Market analysis AI agent.

Monitors execution results, gas trends, and market conditions.
Periodically produces MarketAnalysisPayload with LLM-generated insights
and adaptive parameter recommendations.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from src.agents.base_agent import BaseAgent, LLMProvider
from src.blockchain.polygon_client import PolygonClient
from src.payload.communicator import PayloadCommunicator
from src.payload.protocol import (
    ExecutionResultPayload,
    MarketAnalysisPayload,
    MessageType,
    Priority,
)
from src.utils.logger import get_logger

log = get_logger(__name__)

_MAX_HISTORY = 100
_ANALYSIS_INTERVAL_S = 60   # produce analysis every 60 seconds


class AnalysisAgent(BaseAgent):
    """
    Observes system-wide metrics and produces periodic market analysis.

    Subscribes to EXECUTION_RESULT messages to track success rates.
    Publishes MARKET_ANALYSIS payloads consumed by other components.
    """

    def __init__(
        self,
        client: PolygonClient,
        bus: PayloadCommunicator,
        preferred_provider: Optional[LLMProvider] = None,
    ) -> None:
        super().__init__(
            name="AnalysisAgent",
            preferred_provider=preferred_provider or LLMProvider.GROQ,
        )
        self._client = client
        self._bus    = bus

        # Rolling windows
        self._exec_results: Deque[ExecutionResultPayload] = deque(maxlen=_MAX_HISTORY)
        self._gas_history:  Deque[float] = deque(maxlen=_MAX_HISTORY)
        self._last_analysis_time = 0.0

    @property
    def system_prompt(self) -> str:
        return (
            "You are a DeFi market intelligence analyst. "
            "Analyse arbitrage execution metrics and provide concise market insights. "
            "Respond ONLY with valid JSON containing: "
            "'summary': str (max 80 words), "
            "'recommended_min_profit_usd': float, "
            "'market_condition': str (one of: bullish, bearish, sideways, volatile)."
        )

    # ── Subscriber callbacks ──────────────────────────────────────────────────

    async def on_execution_result(self, message: ExecutionResultPayload) -> None:
        """Record execution results as they arrive from ExecutionAgent."""
        self._exec_results.append(message)
        log.debug("AnalysisAgent: recorded execution result (success=%s)", message.success)

        # Produce analysis if enough time has passed
        now = time.time()
        if now - self._last_analysis_time >= _ANALYSIS_INTERVAL_S:
            await self._produce_analysis()
            self._last_analysis_time = now

    # ── Metrics ───────────────────────────────────────────────────────────────

    def _compute_metrics(self) -> Dict[str, Any]:
        results = list(self._exec_results)
        if not results:
            return {
                "total": 0,
                "successes": 0,
                "success_rate": 0.0,
                "avg_profit_usd": 0.0,
                "total_profit_usd": 0.0,
                "avg_gas_used": 0,
            }

        successes = [r for r in results if r.success]
        profits   = [r.actual_profit_usd for r in successes]

        return {
            "total":            len(results),
            "successes":        len(successes),
            "success_rate":     len(successes) / len(results),
            "avg_profit_usd":   sum(profits) / len(profits) if profits else 0.0,
            "total_profit_usd": sum(profits),
            "avg_gas_used":     sum(r.gas_used for r in results) // len(results),
        }

    async def _get_gas_stats(self) -> Dict[str, float]:
        try:
            gwei = await self._client.get_gas_price_gwei()
            self._gas_history.append(gwei)
        except Exception:
            gwei = self._gas_history[-1] if self._gas_history else 50.0

        hist = list(self._gas_history)
        return {
            "current_gwei": gwei,
            "avg_gwei":     sum(hist) / len(hist) if hist else gwei,
            "max_gwei":     max(hist) if hist else gwei,
        }

    # ── LLM analysis ─────────────────────────────────────────────────────────

    async def _produce_analysis(self) -> None:
        metrics  = self._compute_metrics()
        gas_info = await self._get_gas_stats()

        user_msg = (
            f"Last {len(self._exec_results)} arbitrage executions:\n"
            f"- Success rate: {metrics['success_rate']:.1%}\n"
            f"- Avg profit: ${metrics['avg_profit_usd']:.4f}\n"
            f"- Total profit: ${metrics['total_profit_usd']:.4f}\n"
            f"- Gas (current/avg/max gwei): "
            f"{gas_info['current_gwei']:.1f}/{gas_info['avg_gwei']:.1f}/{gas_info['max_gwei']:.1f}\n"
            f"Provide market analysis and recommend min_profit_usd threshold."
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_msg},
        ]

        llm_summary  = "Analysis unavailable."
        rec_min_profit = 5.0
        try:
            resp = await self.call_llm(messages, expect_json=True)
            if resp.parsed:
                llm_summary  = resp.parsed.get("summary", llm_summary)
                rec_min_profit = float(resp.parsed.get("recommended_min_profit_usd", rec_min_profit))
        except Exception as exc:
            log.warning("AnalysisAgent LLM call failed: %s", exc)

        analysis = MarketAnalysisPayload(
            source="AnalysisAgent",
            priority=Priority.LOW,
            gas_price_gwei=gas_info["current_gwei"],
            block_utilization_pct=50.0,
            avg_spread_pct=0.0,
            opportunities_last_hour=metrics["total"],
            success_rate_last_hour=metrics["success_rate"],
            recommended_min_profit_usd=rec_min_profit,
            llm_summary=llm_summary,
        )
        await self._bus.publish(analysis)
        log.info("AnalysisAgent: published market analysis — %s", llm_summary[:60])

    # ── Periodic loop ─────────────────────────────────────────────────────────

    async def run_periodic(self, interval_seconds: float = _ANALYSIS_INTERVAL_S) -> None:
        """Run analysis on a timer (alternative to event-driven trigger)."""
        while True:
            await asyncio.sleep(interval_seconds)
            await self._produce_analysis()

    async def process(self, *args: Any, **kwargs: Any) -> None:
        """Called by the main loop to start periodic analysis."""
        await self.run_periodic()
