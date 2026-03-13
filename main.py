"""
Main orchestrator — Find → Simulate → Release cycle.

Entry point for the Arbisy atomic arbitrage system.

Cycle
-----
FIND:     DEXPriceFeed polls all configured pools
SIMULATE: ML model scores each candidate; on-chain simulateProfit called
RELEASE:  ExecutionAgent fires the flash loan transaction
"""

from __future__ import annotations

import asyncio
import os
import signal
import sys
from typing import Any, Dict, List, Optional

# Ensure src is on the path when running as `python main.py`
sys.path.insert(0, os.path.dirname(__file__))

from src.agents.analysis_agent import AnalysisAgent
from src.agents.arbitrage_agent import ArbitrageAgent
from src.agents.execution_agent import ExecutionAgent
from src.arbitrage.cross_platform import CrossPlatformArbitrage
from src.arbitrage.matrix import ArbitrageMatrix
from src.arbitrage.triangular import TriangularArbitrage
from src.blockchain.dex_price_feed import DEXPriceFeed, PoolPrice
from src.blockchain.flash_loan import FlashLoan
from src.blockchain.polygon_client import PolygonClient
from src.ml.inference import LiveInference
from src.ml.training import train as train_model
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


class ArbisyOrchestrator:
    """
    Wires all components together and drives the main event loop.

    Architecture
    ------------
    - PayloadCommunicator  — async pub/sub bus
    - PolygonClient        — Web3 connection
    - DEXPriceFeed         — price polling
    - TriangularArbitrage  — Bellman-Ford cycle detection
    - CrossPlatformArbitrage — spread detection
    - ArbitrageMatrix      — Floyd-Warshall matrix solver
    - LiveInference        — ML scoring
    - ArbitrageAgent       — LLM-enriched opportunity ranking
    - ExecutionAgent       — flash loan execution
    - AnalysisAgent        — market monitoring
    """

    def __init__(self) -> None:
        self._cfg       = get_config()
        self._running   = False

        # Core infrastructure
        self._bus       = PayloadCommunicator()
        self._client    = PolygonClient()
        self._price_feed = DEXPriceFeed(self._client)
        self._flash     = FlashLoan(self._client)

        # Detectors
        self._tri_arb   = TriangularArbitrage(max_path_length=4)
        self._cross_arb = CrossPlatformArbitrage()
        self._matrix    = ArbitrageMatrix()

        # ML
        self._inference = LiveInference()

        # AI Agents
        self._arb_agent  = ArbitrageAgent()
        self._exec_agent = ExecutionAgent(
            client=self._client,
            flash_loan=self._flash,
            inferencer=self._inference,
            bus=self._bus,
        )
        self._anal_agent = AnalysisAgent(
            client=self._client,
            bus=self._bus,
        )

        # Wire subscriptions
        self._bus.subscribe(MessageType.OPPORTUNITY,      self._exec_agent.process)
        self._bus.subscribe(MessageType.EXECUTION_RESULT, self._anal_agent.on_execution_result)

    # ── Startup ───────────────────────────────────────────────────────────────

    async def _ensure_model(self) -> None:
        """Train model if no persisted version exists."""
        if not self._inference.model.trained:
            log.info("No trained model found — training on synthetic data...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: train_model(
                model_path=self._cfg.model_path,
                use_real_data=False,
                synthetic_samples=2000,
            ))
            # Reload the trained model into the *existing* LiveInference instance.
            # ExecutionAgent holds a reference to the same object, so it will
            # automatically start using the fresh model without re-wiring.
            self._inference.reload()

    async def _verify_connection(self) -> None:
        connected = await self._client.is_connected()
        if not connected:
            log.warning("RPC connection failed — continuing in offline mode")
        else:
            block = await self._client.get_block_number()
            log.info("Connected to Polygon zkEVM — block #%d", block)

    # ── Find phase ────────────────────────────────────────────────────────────

    async def _find(self) -> List[Dict[str, Any]]:
        """Poll DEX prices and run all detectors. Return raw candidate list."""
        pool_prices: List[PoolPrice] = await self._price_feed.fetch_all_prices()

        if not pool_prices:
            log.debug("No pool prices fetched this cycle")
            return []

        # Update all detectors
        self._tri_arb.update_prices(pool_prices)
        self._cross_arb.update_prices(pool_prices)
        self._matrix.build_from_prices(pool_prices)

        candidates: List[Dict[str, Any]] = []

        # Triangular arbitrage
        tri_opps = self._tri_arb.find_opportunities(min_profit_pct=0.05)
        for opp in tri_opps:
            # Build hop structure from the path
            hops = []
            for i in range(len(opp.tokens) - 1):
                hops.append({
                    "token_in":  opp.tokens[i],
                    "token_out": opp.tokens[i + 1],
                    "dex_name":  opp.dexes[i],
                    "router_address": "0x0",  # resolved from dex_name at execution time
                    "fee_bps":   30,
                    "is_v3":     True,
                    "estimated_amount_out": 0.0,
                })
            candidates.append({
                "arb_type":              "triangular",
                "estimated_profit_pct":  opp.estimated_profit_pct,
                "asset":                 opp.tokens[0],
                "loan_amount_usd":       10_000.0,
                "loan_amount_wei":       10_000 * 10**6,  # 10k USDC
                "dex_sources":           list(set(opp.dexes)),
                "hops":                  hops,
                "slippage_estimate_pct": 0.1,
                "gas_estimate_gwei":     0.0,
            })

        # Cross-platform arbitrage
        cross_opps = self._cross_arb.find_opportunities(min_spread_pct=0.1)
        for opp in cross_opps:
            candidates.append({
                "arb_type":              "cross_platform",
                "estimated_profit_pct":  opp.estimated_profit_pct,
                "asset":                 opp.token_a,
                "loan_amount_usd":       10_000.0,
                "loan_amount_wei":       10_000 * 10**6,
                "dex_sources":           [opp.buy_dex, opp.sell_dex],
                "hops": [
                    {
                        "token_in":  opp.token_a,
                        "token_out": opp.token_b,
                        "dex_name":  opp.buy_dex,
                        "router_address": "0x0",
                        "fee_bps":   opp.buy_fee_bps,
                        "is_v3":     True,
                        "estimated_amount_out": 0.0,
                    },
                    {
                        "token_in":  opp.token_b,
                        "token_out": opp.token_a,
                        "dex_name":  opp.sell_dex,
                        "router_address": "0x0",
                        "fee_bps":   opp.sell_fee_bps,
                        "is_v3":     True,
                        "estimated_amount_out": 0.0,
                    },
                ],
                "slippage_estimate_pct": 0.1,
                "gas_estimate_gwei":     0.0,
            })

        # Matrix arbitrage
        matrix_opps = self._matrix.find_opportunities(min_profit_pct=0.05)
        for opp in matrix_opps:
            candidates.append({
                "arb_type":              "matrix",
                "estimated_profit_pct":  opp.profit_pct,
                "asset":                 opp.token_names[0] if opp.token_names else "0x0",
                "loan_amount_usd":       10_000.0,
                "loan_amount_wei":       10_000 * 10**6,
                "dex_sources":           ["matrix"],
                "hops":                  [],
                "slippage_estimate_pct": 0.15,
                "gas_estimate_gwei":     0.0,
            })

        log.info(
            "FIND: %d triangular, %d cross-platform, %d matrix candidates",
            len(tri_opps), len(cross_opps), len(matrix_opps),
        )
        return candidates

    # ── Simulate phase ────────────────────────────────────────────────────────

    async def _simulate(
        self, candidates: List[Dict[str, Any]]
    ) -> List[OpportunityPayload]:
        """Score candidates with ML + LLM; return approved payloads."""
        if not candidates:
            return []

        try:
            block_num = await self._client.get_block_number()
            gas_gwei  = await self._client.get_gas_price_gwei()
        except Exception:
            block_num, gas_gwei = 0, 50.0

        pool_prices = list(self._price_feed.get_cached_prices().values())

        payloads = await self._arb_agent.process(
            pool_prices=pool_prices,
            raw_candidates=candidates,
            gas_price_gwei=gas_gwei,
            block_number=block_num,
        )

        # Attach ML scores
        scored: List[OpportunityPayload] = []
        for payload in payloads:
            ml_score, pred_profit = self._inference.score(
                payload,
                gas_cost_usd=gas_gwei * 150_000 * 1e-9 * 1800,
                block_utilization_pct=50.0,
            )
            payload.ml_score = ml_score
            if ml_score >= self._cfg.ml_score_threshold and pred_profit >= self._cfg.min_profit_usd:
                scored.append(payload)
                log.info(
                    "SIMULATE ✓ opp=%s score=%.4f profit=$%.4f",
                    payload.message_id[:8], ml_score, pred_profit,
                )
            else:
                log.debug(
                    "SIMULATE ✗ opp=%s score=%.4f profit=$%.4f",
                    payload.message_id[:8], ml_score, pred_profit,
                )

        return scored

    # ── Release phase ─────────────────────────────────────────────────────────

    async def _release(self, payloads: List[OpportunityPayload]) -> None:
        """Publish approved opportunities to the execution bus."""
        for payload in payloads:
            payload.source = "Orchestrator"
            await self._bus.publish(payload)
            log.info("RELEASE → published opp %s to bus", payload.message_id[:8])

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def _cycle(self) -> None:
        """One complete Find → Simulate → Release iteration."""
        candidates = await self._find()
        if not candidates:
            return
        payloads   = await self._simulate(candidates)
        await self._release(payloads)

    async def run(self) -> None:
        """Start the orchestrator: pre-flight checks, then continuous loop."""
        log.info("=" * 60)
        log.info("  Arbisy Atomic Arbitrage System — starting up")
        log.info("=" * 60)

        await self._verify_connection()
        await self._ensure_model()

        self._running = True

        # Start background tasks
        bus_task     = asyncio.create_task(self._bus.run(), name="bus")
        anal_task    = asyncio.create_task(self._anal_agent.process(), name="analysis")

        log.info("Entering Find → Simulate → Release loop (interval=%.1fs)", self._cfg.poll_interval_seconds)

        try:
            while self._running:
                try:
                    await self._cycle()
                except Exception as exc:
                    log.error("Cycle error: %s", exc, exc_info=True)
                await asyncio.sleep(self._cfg.poll_interval_seconds)
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False
            self._bus.stop()
            bus_task.cancel()
            anal_task.cancel()
            await self._arb_agent.close()
            await self._exec_agent.close()
            await self._anal_agent.close()
            log.info("Arbisy shut down cleanly.")

    def stop(self) -> None:
        self._running = False
        self._bus.stop()


def main() -> None:
    orchestrator = ArbisyOrchestrator()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    def _shutdown(sig: signal.Signals) -> None:
        log.info("Received %s — shutting down...", sig.name)
        orchestrator.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: _shutdown(s))

    try:
        loop.run_until_complete(orchestrator.run())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
