"""
Trade execution AI agent.

Manages the execution of flash loan arbitrage transactions.
Uses LLM reasoning to make final go/no-go decisions before signing.
"""

from __future__ import annotations

import asyncio
import json
import os
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

# File where the circuit-breaker state is persisted across restarts (E4-S6)
_CIRCUIT_STATE_FILE = "data/circuit_breaker.json"

# Retry parameters for transient RPC errors (E4-S3)
_MAX_RETRIES = 3
_MAX_RETRY_DELAY = 8.0  # seconds


class ExecutionAgent(BaseAgent):
    """
    Executes approved arbitrage opportunities as flash loan transactions.

    Flow
    ----
    1. Receive OpportunityPayload from the bus
    2. Run final ML score check
    3. Ask LLM for final go/no-go reasoning
    4. If approved, call FlashLoan.execute_flash_loan() with exponential back-off retry
    5. Publish ExecutionResultPayload

    Circuit-breaker (E4-S6)
    -----------------------
    Consecutive-loss count is persisted to ``data/circuit_breaker.json`` so
    that a halted circuit breaker survives process restarts.
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
        self._circuit_break_limit = 3
        # Load persisted circuit-breaker state (E4-S6)
        self._consecutive_losses = self._load_circuit_state()

    @property
    def system_prompt(self) -> str:
        return (
            "You are an automated DeFi trade execution system. "
            "You make final go/no-go decisions on flash loan arbitrage trades. "
            "Respond ONLY with valid JSON containing 'execute': bool and 'reason': str."
        )

    # ── Circuit-breaker persistence (E4-S6) ───────────────────────────────────

    @staticmethod
    def _load_circuit_state() -> int:
        """Load persisted consecutive-loss count from disk."""
        try:
            if os.path.exists(_CIRCUIT_STATE_FILE):
                with open(_CIRCUIT_STATE_FILE) as f:
                    data = json.load(f)
                    return int(data.get("consecutive_losses", 0))
        except Exception as exc:
            log.warning("Could not load circuit-breaker state: %s", exc)
        return 0

    def _save_circuit_state(self) -> None:
        """Persist consecutive-loss count to disk so it survives restarts."""
        try:
            os.makedirs(os.path.dirname(_CIRCUIT_STATE_FILE) or ".", exist_ok=True)
            with open(_CIRCUIT_STATE_FILE, "w") as f:
                json.dump({"consecutive_losses": self._consecutive_losses}, f)
        except Exception as exc:
            log.warning("Could not persist circuit-breaker state: %s", exc)

    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker (e.g. after operator review)."""
        self._consecutive_losses = 0
        self._save_circuit_state()
        log.info("Circuit breaker manually reset")

    # ── Retry logic (E4-S3) ───────────────────────────────────────────────────

    async def _execute_with_retry(
        self,
        opportunity: OpportunityPayload,
    ) -> Dict[str, Any]:
        """
        Execute the flash loan with exponential back-off retry on transient
        RPC errors (up to _MAX_RETRIES attempts, max delay _MAX_RETRY_DELAY s).

        Transient errors are identified both by exception type (asyncio.TimeoutError,
        ConnectionError, OSError) at the call site and by keywords in the returned
        error_message string.  Contract reverts and non-transient failures are not
        retried so they fail fast without wasting gas budget.
        """
        # Keywords that indicate a transient RPC/network error in the error string
        _TRANSIENT_KEYWORDS = ("timeout", "connection", "rpc", "network", "429", "503")

        last_result: Dict[str, Any] = {}
        for attempt in range(_MAX_RETRIES):
            try:
                last_result = await self._flash.execute_flash_loan(opportunity)
            except (asyncio.TimeoutError, ConnectionError, OSError) as exc:
                # Transient network/OS exception — treat as retryable
                last_result = {"success": False, "error_message": str(exc)}
            if last_result.get("success"):
                return last_result
            # If not a transient error (e.g. contract reverted), don't retry
            err = last_result.get("error_message", "")
            transient = any(kw in err.lower() for kw in _TRANSIENT_KEYWORDS)
            if not transient:
                return last_result
            if attempt < _MAX_RETRIES - 1:
                delay = min(2.0 ** attempt, _MAX_RETRY_DELAY)
                log.warning(
                    "Transient RPC error (attempt %d/%d), retrying in %.1fs: %s",
                    attempt + 1, _MAX_RETRIES, delay, err,
                )
                await asyncio.sleep(delay)
        return last_result

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

        # ── Execute (with exponential back-off retry) ─────────────────────
        log.info("Executing flash loan for opportunity %s", opp_id)
        result = await self._execute_with_retry(opportunity)

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
            self._save_circuit_state()
            self._inference.record_outcome(opportunity.arb_type, True)
            log.info("✓ Execution succeeded: tx=%s profit=$%.4f", exec_result.tx_hash[:10], exec_result.actual_profit_usd)
        else:
            self._consecutive_losses += 1
            self._save_circuit_state()
            self._inference.record_outcome(opportunity.arb_type, False)
            log.warning("✗ Execution failed: %s (losses=%d)", exec_result.error_message, self._consecutive_losses)

    async def process(self, opportunity: OpportunityPayload) -> None:
        """Entry point called by the main loop."""
        await self.handle_opportunity(opportunity)
