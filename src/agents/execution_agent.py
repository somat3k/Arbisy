"""
Trade execution AI agent.

Manages the execution of flash loan arbitrage transactions.
Uses LLM reasoning to make final go/no-go decisions before signing.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent, LLMProvider
from src.blockchain.flash_loan import FlashLoan
from src.blockchain.polygon_client import PolygonClient
from src.ml.inference import LiveInference
from src.payload.communicator import PayloadCommunicator
from src.payload.protocol import (
    ExecutionResultPayload,
    MessageType,
    OpportunityPayload,
    Priority,
    SystemEventPayload,
)
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# Approximate ETH price in USD used for gas cost estimation.
# Update this constant or replace with a live price feed for production use.
_ETH_PRICE_USD: float = 1800.0


class ExecutionAgent(BaseAgent):
    """
    Executes approved arbitrage opportunities as flash loan transactions.

    Flow
    ----
    1. Receive OpportunityPayload from the bus
    2. Run final ML score check
    3. Ask LLM for final go/no-go reasoning
    4. If approved, call FlashLoan.execute_flash_loan()
    5. Publish ExecutionResultPayload
    """

    def __init__(
        self,
        client: PolygonClient,
        flash_loan: FlashLoan,
        inferencer: LiveInference,
        bus: PayloadCommunicator,
        preferred_provider: Optional[LLMProvider] = None,
    ) -> None:
        super().__init__(
            name="ExecutionAgent",
            preferred_provider=preferred_provider or LLMProvider.GROQ,
        )
        self._client    = client
        self._flash     = flash_loan
        self._inference = inferencer
        self._bus       = bus
        cfg             = get_config()
        self._min_profit_usd   = cfg.min_profit_usd
        self._max_gas_gwei     = cfg.max_gas_price_gwei
        self._ml_threshold     = cfg.ml_score_threshold
        self._consecutive_losses = 0
        self._circuit_break_limit = 3

    @property
    def system_prompt(self) -> str:
        return (
            "You are an automated DeFi trade execution system. "
            "You make final go/no-go decisions on flash loan arbitrage trades. "
            "Respond ONLY with valid JSON containing 'execute': bool and 'reason': str."
        )

    async def _final_llm_check(
        self,
        opportunity: OpportunityPayload,
        ml_score: float,
        predicted_profit_usd: float,
        gas_price_gwei: float,
    ) -> bool:
        """Ask the LLM for final execution approval."""
        user_msg = (
            f"Flash loan arbitrage opportunity:\n"
            f"- Asset: {opportunity.asset}\n"
            f"- Loan: ${opportunity.loan_amount_usd:,.2f}\n"
            f"- Expected profit: ${opportunity.expected_profit_usd:.4f} USD\n"
            f"- ML predicted profit: ${predicted_profit_usd:.4f} USD\n"
            f"- ML score: {ml_score:.4f}\n"
            f"- Gas price: {gas_price_gwei:.2f} gwei\n"
            f"- Arb type: {opportunity.arb_type}\n"
            f"- Hops: {len(opportunity.path)}\n"
            f"- Slippage estimate: {opportunity.slippage_estimate_pct:.3f}%\n"
            f"Should we execute? Respond with JSON: {{\"execute\": bool, \"reason\": str}}"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_msg},
        ]

        try:
            resp = await self.call_llm(messages, expect_json=True)
            if resp.parsed:
                execute = resp.parsed.get("execute", False)
                reason  = resp.parsed.get("reason", "")
                log.info("LLM decision: execute=%s | %s", execute, reason)
                return bool(execute)
        except Exception as exc:
            log.warning("LLM final check failed: %s — defaulting to ML score", exc)

        # Fallback to ML score
        return ml_score >= self._ml_threshold

    async def handle_opportunity(self, opportunity: OpportunityPayload) -> None:
        """
        Process a single opportunity end-to-end.
        Called as a subscriber callback from the bus.
        """
        opp_id = opportunity.message_id[:8]
        log.info("ExecutionAgent: evaluating opportunity %s", opp_id)

        # ── Circuit breaker ────────────────────────────────────────────────
        if self._consecutive_losses >= self._circuit_break_limit:
            log.warning("Circuit breaker active — skipping execution (losses=%d)", self._consecutive_losses)
            return

        # ── Gas check ─────────────────────────────────────────────────────
        try:
            gas_gwei = await self._client.get_gas_price_gwei()
        except Exception:
            gas_gwei = 0.0

        if gas_gwei > self._max_gas_gwei:
            log.info("Gas too high (%.2f > %.2f gwei) — skipping %s", gas_gwei, self._max_gas_gwei, opp_id)
            return

        # ── ML score ──────────────────────────────────────────────────────
        ml_score, predicted_profit = self._inference.score(
            opportunity,
            gas_cost_usd=gas_gwei * 150_000 * 1e-9 * _ETH_PRICE_USD,  # rough USD estimate
            block_utilization_pct=50.0,
        )
        if ml_score < self._ml_threshold:
            log.info("ML score %.4f < threshold %.2f — skipping %s", ml_score, self._ml_threshold, opp_id)
            return

        # ── Minimum profit check ──────────────────────────────────────────
        if predicted_profit < self._min_profit_usd:
            log.info("Predicted profit $%.4f < min $%.2f — skipping %s", predicted_profit, self._min_profit_usd, opp_id)
            return

        # ── LLM final approval ────────────────────────────────────────────
        approved = await self._final_llm_check(opportunity, ml_score, predicted_profit, gas_gwei)
        if not approved:
            log.info("LLM rejected opportunity %s", opp_id)
            return

        # ── Execute ───────────────────────────────────────────────────────
        log.info("Executing flash loan for opportunity %s", opp_id)
        result = await self._flash.execute_flash_loan(opportunity)

        # ── Publish result ────────────────────────────────────────────────
        exec_result = ExecutionResultPayload(
            source="ExecutionAgent",
            priority=Priority.HIGH,
            opportunity_id=opportunity.message_id,
            success=result.get("success", False),
            tx_hash=result.get("tx_hash", ""),
            actual_profit_usd=predicted_profit if result.get("success") else 0.0,
            gas_used=result.get("gas_used", 0),
            gas_price_gwei=gas_gwei,
            error_message=result.get("error_message", ""),
            block_number=result.get("block_number", 0),
        )
        await self._bus.publish(exec_result)

        # ── Update circuit breaker and ML feedback ────────────────────────
        if result.get("success"):
            self._consecutive_losses = 0
            self._inference.record_outcome(opportunity.arb_type, True)
            log.info("✓ Execution succeeded: tx=%s profit=$%.4f", exec_result.tx_hash[:10], exec_result.actual_profit_usd)
        else:
            self._consecutive_losses += 1
            self._inference.record_outcome(opportunity.arb_type, False)
            log.warning("✗ Execution failed: %s (losses=%d)", exec_result.error_message, self._consecutive_losses)

    async def process(self, opportunity: OpportunityPayload) -> None:
        """Entry point called by the main loop."""
        await self.handle_opportunity(opportunity)
