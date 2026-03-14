"""
Arbitrage analysis AI agent.

Uses an LLM to evaluate arbitrage opportunities, provide reasoning about
market conditions, and decide whether to proceed with execution.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from src.agents.base_agent import BaseAgent, LLMProvider
from src.blockchain.dex_price_feed import PoolPrice
from src.payload.protocol import (
    MessageType,
    OpportunityPayload,
    Priority,
    SwapHop,
    SystemEventPayload,
)
from src.utils.logger import get_logger

log = get_logger(__name__)


class ArbitrageAgent(BaseAgent):
    """
    Analyses raw arbitrage candidates and enriches them with LLM reasoning.

    Responsibilities
    ----------------
    - Receives lists of raw candidates (from triangular/cross-platform detectors)
    - Calls LLM to rank and filter opportunities
    - Attaches `dex_sources` and `arb_type` metadata
    - Publishes OpportunityPayload to the bus
    """

    def __init__(self, preferred_provider: Optional[LLMProvider] = None) -> None:
        super().__init__(
            name="ArbitrageAgent",
            preferred_provider=preferred_provider or LLMProvider.GROQ,
        )

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert DeFi arbitrage analyst. "
            "You evaluate arbitrage opportunities on Polygon zkEVM, considering "
            "price spreads, gas costs, liquidity, slippage risk, and flash loan premiums. "
            "Respond ONLY with valid JSON. Never include markdown or explanations outside JSON."
        )

    async def evaluate_opportunities(
        self,
        candidates: List[Dict[str, Any]],
        gas_price_gwei: float,
        block_number: int,
    ) -> List[Dict[str, Any]]:
        """
        Ask the LLM to rank and annotate a list of raw arbitrage candidates.

        Returns a filtered, ranked list with LLM reasoning attached.
        """
        if not candidates:
            return []

        # Limit to top-10 candidates to keep prompt size reasonable
        top = sorted(candidates, key=lambda c: c.get("estimated_profit_pct", 0), reverse=True)[:10]

        user_msg = (
            f"Current gas price: {gas_price_gwei:.2f} gwei | block: {block_number}\n"
            f"Evaluate these arbitrage candidates and return a JSON array of the top 3 "
            f"ranked by risk-adjusted expected profit. For each, include:\n"
            f"- 'rank': int (1 = best)\n"
            f"- 'reasoning': str (max 50 words)\n"
            f"- 'risk_score': float 0-1 (1 = highest risk)\n"
            f"- 'proceed': bool\n"
            f"- 'estimated_profit_pct': float\n\n"
            f"Candidates:\n{json.dumps(top, indent=2)}"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": user_msg},
        ]

        try:
            response = await self.call_llm(messages, expect_json=True)
            if response.parsed and isinstance(response.parsed, list):
                return response.parsed
            if response.parsed and "opportunities" in response.parsed:
                return response.parsed["opportunities"]
        except Exception as exc:
            log.warning("LLM evaluation failed, using raw ranking: %s", exc)

        # Fallback: return top 3 as-is
        return top[:3]

    async def build_opportunity_payload(
        self,
        candidate: Dict[str, Any],
        loan_amount_wei: int,
        loan_amount_usd: float,
        source: str = "ArbitrageAgent",
    ) -> OpportunityPayload:
        """Convert a raw candidate dict into a typed OpportunityPayload."""
        hops_raw = candidate.get("hops", [])
        path: List[SwapHop] = [
            SwapHop(
                dex_name=h.get("dex_name", "unknown"),
                router_address=h.get("router_address", "0x0"),
                token_in=h.get("token_in", "0x0"),
                token_out=h.get("token_out", "0x0"),
                fee_bps=h.get("fee_bps", 30),
                is_v3=h.get("is_v3", True),
                estimated_amount_out=h.get("estimated_amount_out", 0.0),
            )
            for h in hops_raw
        ]

        profit_pct = candidate.get("estimated_profit_pct", 0.0)
        expected_profit_usd = loan_amount_usd * profit_pct / 100.0

        return OpportunityPayload(
            source=source,
            priority=Priority.HIGH if profit_pct > 0.5 else Priority.MEDIUM,
            asset=candidate.get("asset", ""),
            loan_amount_wei=loan_amount_wei,
            loan_amount_usd=loan_amount_usd,
            expected_profit_usd=expected_profit_usd,
            expected_profit_wei=int(loan_amount_wei * profit_pct / 100.0),
            ml_score=0.0,  # Will be filled by ML inference
            path=path,
            gas_estimate_gwei=candidate.get("gas_estimate_gwei", 0.0),
            slippage_estimate_pct=candidate.get("slippage_estimate_pct", 0.1),
            dex_sources=candidate.get("dex_sources", []),
            arb_type=candidate.get("arb_type", "cross_platform"),
        )

    async def process(
        self,
        pool_prices: List[PoolPrice],
        raw_candidates: List[Dict[str, Any]],
        gas_price_gwei: float,
        block_number: int,
    ) -> List[OpportunityPayload]:
        """
        Main processing: evaluate candidates and return ranked OpportunityPayloads.
        """
        if not raw_candidates:
            return []

        ranked = await self.evaluate_opportunities(
            raw_candidates, gas_price_gwei, block_number
        )

        payloads: List[OpportunityPayload] = []
        for candidate in ranked:
            if not candidate.get("proceed", True):
                continue
            payload = await self.build_opportunity_payload(
                candidate,
                loan_amount_wei=int(candidate.get("loan_amount_wei", 10_000 * 10**6)),
                loan_amount_usd=candidate.get("loan_amount_usd", 10_000.0),
            )
            payloads.append(payload)

        log.info(
            "ArbitrageAgent produced %d payloads from %d candidates",
            len(payloads), len(raw_candidates),
        )
        return payloads
