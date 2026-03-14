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
from src.arbitrage.array_builder import ArrayBuilder, PathArray
from src.arbitrage.cross_platform import CrossPlatformArbitrage
from src.arbitrage.matrix import ArbitrageMatrix
from src.arbitrage.triangular import TriangularArbitrage
from src.blockchain.dex_price_feed import DEXPriceFeed, PoolPrice
from src.blockchain.factory_scanner import DEXFactoryScanner
from src.blockchain.flash_loan import FlashLoan
from src.blockchain.polygon_client import PolygonClient
from src.blockchain.swap_event_feed import SwapEventFeed
from src.ml.inference import LiveInference
from src.ml.neural_model import NeuralArbModel
from src.ml.training import train as train_model, train_nn as train_nn_model
from src.payload.communicator import PayloadCommunicator
from src.payload.protocol import (
    ExecutionResultPayload,
    MessageType,
    OpportunityPayload,
    Priority,
    SystemEventPayload,
)
from src.utils.config import get_config
from src.utils.dashboard import dashboard_manager, record_execution, record_opportunity, set_running
from src.utils.db import ExecutionDB, get_db
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
        self._db        = get_db()

        # Detectors
        self._tri_arb     = TriangularArbitrage(max_path_length=4)
        self._cross_arb   = CrossPlatformArbitrage()
        self._matrix      = ArbitrageMatrix()
        self._array_builder = ArrayBuilder(max_hops=4, min_profit_pct=0.05)

        # Pool discovery components (E2-S2, E2-S3)
        self._swap_feed      = SwapEventFeed(
            self._client,
            on_price_update=self._on_swap_price_update,
        )
        self._factory_scanner = DEXFactoryScanner(self._client)

        # ML: GBR model + Neural network model
        self._inference = LiveInference()
        self._nn_model  = NeuralArbModel(model_path=self._cfg.nn_model_path)

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

        # Wire subscriptions — also hook into dashboard (E5-S4)
        self._bus.subscribe(MessageType.OPPORTUNITY,      self._exec_agent.process)
        self._bus.subscribe(MessageType.OPPORTUNITY,      self._on_opportunity_for_dashboard)
        self._bus.subscribe(MessageType.EXECUTION_RESULT, self._anal_agent.on_execution_result)
        self._bus.subscribe(MessageType.EXECUTION_RESULT, self._on_execution_for_dashboard)
        self._bus.subscribe(MessageType.EXECUTION_RESULT, self._on_execution_for_db)

    # ── Startup ───────────────────────────────────────────────────────────────

    async def _on_swap_price_update(self, pool_price: "PoolPrice") -> None:
        """Callback from SwapEventFeed — ingest a real-time price update."""
        self._tri_arb.update_prices([pool_price])
        self._cross_arb.update_prices([pool_price])
        self._matrix.build_from_prices(list(self._price_feed.get_cached_prices().values()) + [pool_price])
        log.debug(
            "Swap price update: %s/%s = %.6f on %s",
            pool_price.token0_symbol,
            pool_price.token1_symbol,
            pool_price.price_token1_per_token0,
            pool_price.dex_name,
        )

    async def _on_opportunity_for_dashboard(self, payload: "OpportunityPayload") -> None:
        """Forward opportunities to the live dashboard (E5-S4)."""
        record_opportunity(payload.to_dict())

    async def _on_execution_for_dashboard(self, payload: "ExecutionResultPayload") -> None:
        """Forward execution results to the live dashboard (E5-S4)."""
        record_execution(payload.to_dict())

    async def _on_execution_for_db(self, payload: "ExecutionResultPayload") -> None:
        """Persist execution result to PostgreSQL (E3-S1)."""
        if not self._db.enabled:
            return
        # Retrieve the matching opportunity details from the analysis agent if available,
        # otherwise use sensible defaults.
        await self._db.log_execution(
            arb_type="unknown",
            price_spread_pct=0.0,
            gas_cost_usd=payload.gas_price_gwei * 150_000 * 1e-9 * 1800,
            liquidity_depth=0.0,
            historical_success_rate=0.0,
            slippage_estimate=0.0,
            block_utilization=50.0,
            time_since_last_trade=0.0,
            ml_score=0.0,
            executed=True,
            success=payload.success,
            net_profit_usd=payload.actual_profit_usd if payload.success else 0.0,
            tx_hash=payload.tx_hash,
        )

    async def _ensure_model(self) -> None:
        """Train models if no persisted versions exist."""
        if not self._inference.model.trained:
            log.info("No trained GBR model found — training on synthetic data...")
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

        if not self._nn_model.trained:
            log.info("No trained NN model found — training on synthetic data...")
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: train_nn_model(
                model_path=self._cfg.nn_model_path,
                use_real_data=False,
                synthetic_samples=2000,
            ))
            self._nn_model.reload()

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
        self._array_builder.update_prices(pool_prices)

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
                    "router_address": "0x" + "0" * 40,  # resolved from dex_name at execution time
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
                        "router_address": "0x" + "0" * 40,
                        "fee_bps":   opp.buy_fee_bps,
                        "is_v3":     True,
                        "estimated_amount_out": 0.0,
                    },
                    {
                        "token_in":  opp.token_b,
                        "token_out": opp.token_a,
                        "dex_name":  opp.sell_dex,
                        "router_address": "0x" + "0" * 40,
                        "fee_bps":   opp.sell_fee_bps,
                        "is_v3":     True,
                        "estimated_amount_out": 0.0,
                    },
                ],
                "slippage_estimate_pct": 0.1,
                "gas_estimate_gwei":     0.0,
            })

        # Matrix arbitrage — build proper hops via build_hop_list() (E6-S1)
        matrix_opps = self._matrix.find_opportunities(min_profit_pct=0.05)
        for opp in matrix_opps:
            # Reconstruct actual swap hops from the matrix path (E6-S1)
            matrix_hops = self._matrix.build_hop_list(opp)
            n_hops = len(matrix_hops)
            # Scale slippage estimate with number of hops (multi-hop = more risk)
            slip_pct = max(0.1, n_hops * 0.05)
            dex_sources = list({h["dex_name"] for h in matrix_hops if h["dex_name"] != "unknown"}) or ["matrix"]
            candidates.append({
                "arb_type":              "matrix",
                "estimated_profit_pct":  opp.profit_pct,
                "asset":                 opp.token_names[0] if opp.token_names else "0x" + "0" * 40,
                "loan_amount_usd":       10_000.0,
                "loan_amount_wei":       10_000 * 10**6,
                "dex_sources":           dex_sources,
                "hops":                  matrix_hops,
                "slippage_estimate_pct": slip_pct,
                "gas_estimate_gwei":     0.0,
            })

        # NN-scored ArrayBuilder arbitrage
        # Collect all unique loan assets from current pool prices
        loan_assets = list({
            pp.token0.lower() for pp in pool_prices
        } | {pp.token1.lower() for pp in pool_prices})

        array_opps: List[PathArray] = self._array_builder.build_profitable_arrays(
            loan_assets=loan_assets,
            loan_amount_usd=10_000.0,
            nn_model=self._nn_model if self._nn_model.trained else None,
        )
        for pa in array_opps:
            hops = [
                {
                    "token_in":             pa.token_path[i],
                    "token_out":            pa.token_path[i + 1],
                    "dex_name":             pa.dex_path[i],
                    "router_address":       "0x" + "0" * 40,
                    "fee_bps":              pa.fee_path[i] if i < len(pa.fee_path) else 30,
                    "is_v3":                True,
                    "estimated_amount_out": pa.final_usd if i == len(pa.dex_path) - 1 else 0.0,
                }
                for i in range(len(pa.dex_path))
            ]
            candidates.append({
                "arb_type":              "array_builder",
                "estimated_profit_pct":  pa.profit_pct,
                "asset":                 pa.loan_asset,
                "loan_amount_usd":       pa.initial_usd,
                # NOTE: wei conversion uses 6-decimal assumption (USDC/USDT).
                # In production, look up pa.loan_asset token decimals from
                # the chain and compute wei = int(pa.initial_usd * 10**decimals).
                "loan_amount_wei":       int(pa.initial_usd * 10**6),
                "dex_sources":           list(set(pa.dex_path)),
                "hops":                  hops,
                "slippage_estimate_pct": pa.num_hops * 0.1,
                "gas_estimate_gwei":     0.0,
                "nn_score":              pa.nn_score,
            })

        log.info(
            "FIND: %d triangular, %d cross-platform, %d matrix, %d array-builder candidates",
            len(tri_opps), len(cross_opps), len(matrix_opps), len(array_opps),
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

        # Connect to PostgreSQL if configured (E3-S1)
        await self._db.connect()

        self._running = True
        set_running(True)

        # Start background tasks
        bus_task       = asyncio.create_task(self._bus.run(), name="bus")
        anal_task      = asyncio.create_task(self._anal_agent.process(), name="analysis")

        # Start live dashboard server (E5-S4)
        await dashboard_manager.start(port=8080)

        # Start WebSocket swap-event feed if pool configs are available (E2-S2)
        pool_configs = self._price_feed.pool_configs
        swap_feed_task: Optional[asyncio.Task] = None
        if pool_configs:
            swap_feed_task = asyncio.create_task(
                self._swap_feed.subscribe(pool_configs),
                name="swap_event_feed",
            )
            log.info("SwapEventFeed started for %d pools", len(pool_configs))

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
            set_running(False)
            self._swap_feed.stop()
            if swap_feed_task:
                swap_feed_task.cancel()
            self._bus.stop()
            bus_task.cancel()
            anal_task.cancel()
            dashboard_manager.stop()
            await self._db.close()
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
