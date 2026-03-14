# Arbisy — Agent Ruleset & TODO

## System Overview

Arbisy is an Atomic Arbitrage System built on Polygon zkEVM. It detects and
executes flash-loan-funded arbitrage opportunities across Uniswap V3,
QuickSwap V3, and SushiSwap using a Find → Simulate → Release execution
cycle backed by a sklearn-based ML scoring model and multiple AI agents.

---

## Components

| Component | Responsibility |
|---|---|
| `ArbitrageAgent` | Identifies and ranks arbitrage opportunities |
| `ExecutionAgent` | Signs and broadcasts flash-loan transactions |
| `AnalysisAgent` | Evaluates market conditions, gas, and slippage |
| `PolygonClient` | Web3 connection to Polygon zkEVM |
| `DEXPriceFeed` | Real-time price polling from DEX pool contracts |
| `TriangularArbitrage` | Bellman-Ford negative-cycle detection on token graphs |
| `CrossPlatformArbitrage` | Cross-DEX spread detection |
| `ArbitrageMatrix` | NumPy log-matrix arbitrage path finder |
| `MLModel` | sklearn GBR opportunity scorer |
| `PayloadCommunicator` | asyncio priority-queue inter-component bus |
| `FlashLoan` | Aave V3 flash loan trigger and management |

---

## Agent Roles

### ArbitrageAgent
- Polls `DEXPriceFeed` every cycle
- Invokes `TriangularArbitrage` and `CrossPlatformArbitrage` detectors
- Scores candidates through `MLModel`
- Publishes `OpportunityPayload` to the payload bus

### ExecutionAgent
- Subscribes to `OpportunityPayload` messages
- Calls `FlashLoan.initiate()` for approved candidates
- Reports `ExecutionResultPayload` back to the bus
- Implements retry with exponential back-off on RPC failures

### AnalysisAgent
- Subscribes to `ExecutionResultPayload`
- Updates historical success-rate features used by the ML model
- Produces `MarketAnalysisPayload` with gas/liquidity context
- Uses LLM providers (Groq / OpenRouter / Gemini / OpenAI) for reasoning

---

## Find → Simulate → Release Cycle

```
FIND       DEXPriceFeed scans all configured pools
             └── ArbitrageDetectors identify candidate paths

SIMULATE   FlashLoan.simulateProfit() called on-chain (view)
             └── MLModel.predict() scores expected net profit
             └── AnalysisAgent validates gas/slippage

RELEASE    ExecutionAgent calls FlashLoanArbitrage.initiateFlashLoan()
             └── Aave V3 Pool issues loan → executeOperation callback
             └── Swaps executed on DEXes
             └── Loan + 0.05% premium repaid atomically
```

---

## TODO List

### HIGH Priority
- [ ] Deploy `FlashLoanArbitrage.sol` to Polygon zkEVM mainnet
- [ ] Configure live RPC endpoints and private key in `.env`
- [ ] Seed `MLModel` with real historical trade data and retrain
- [ ] Add WebSocket subscription to Uniswap V3 `Swap` events for sub-block detection
- [ ] Implement proper gas estimation with EIP-1559 support
- [x] Add circuit-breaker: halt execution if 3 consecutive losses occur *(implemented in `ExecutionAgent._consecutive_losses`)*
- [x] Add slippage protection (max 0.5% slippage per hop) *(implemented in `FlashLoanArbitrage.sol` via `maxSlippageBps` + per-hop `amountOutMinimum`)*

### MEDIUM Priority
- [ ] Expand token universe (add MATIC, WBTC, LINK, AAVE pools)
- [ ] Add SushiSwap V3 pool ABI and price-feed integration
- [ ] Tune `GradientBoostingRegressor` hyperparameters via GridSearchCV
- [ ] Persist opportunity and execution logs to PostgreSQL
- [ ] Add Prometheus metrics endpoint for monitoring
- [ ] Implement multi-hop arbitrage (4+ legs) via matrix solver

### LOW Priority
- [ ] Build a simple dashboard (FastAPI + WebSocket) for live monitoring
- [ ] Add support for Balancer V2 pools
- [ ] Explore reinforcement-learning upgrade for the ML model
- [ ] Write Hardhat/Foundry integration tests for the Solidity contract

---

## Security Rules

1. **Private key never logged** — use env vars only; never commit `.env`
2. **Reentrancy guard** on all contract entry points
3. **Owner-only admin** functions (`withdraw`, `updatePool`)
4. **Profit check** inside `executeOperation`; revert if net profit ≤ 0
5. **Whitelist** allowed DEX router addresses in the contract
6. **Max loan cap** — enforce `maxFlashLoanAmount` per asset
7. **Replay protection** — nonce validation in encoded `params`

---

## Execution Rules

1. Minimum expected net profit threshold: **$5 USD equivalent**
2. Maximum gas price: **500 gwei** — skip opportunity otherwise
3. ML model score threshold: **0.60** — only release above this
4. Simulation must succeed before any real execution
5. All amounts must be validated as > 0 before contract call
6. Execution timeout: **15 seconds** — abandon if RPC unresponsive

---

## Model Training Guidelines

- Use `sklearn` only — no torch, tensorflow, onnx, or onnxruntime
- Retrain weekly on accumulated execution history
- Feature set: `price_spread_pct`, `gas_cost_usd`, `liquidity_depth`,
  `historical_success_rate`, `slippage_estimate`, `block_utilization`,
  `time_since_last_trade`
- Target: `net_profit_usd` (after gas and flash-loan premium)
- Persist model with `joblib` to `models/arbitrage_model.joblib`
- Validate with 20% holdout; log RMSE and R² after each training run

---

## LLM Provider Priority

1. **Groq** — fastest inference for time-sensitive reasoning
2. **OpenRouter** — fallback with model routing
3. **Gemini** — secondary fallback
4. **OpenAI** — tertiary fallback

Agents rotate providers on rate-limit (HTTP 429) errors.

---

## EPICS

The following epics capture the major work streams required to take Arbisy
from prototype to production.  Each epic contains a set of user stories that
together deliver a releasable capability.

---

### EPIC 1 — On-Chain Infrastructure

> **Goal:** Deploy and verify the `FlashLoanArbitrage` smart contract on
> Polygon zkEVM and connect all off-chain components to the live contract.

**Stories:**
- [ ] **E1-S1** Set up Hardhat project with `FlashLoanArbitrage.sol` compilation and local fork test.
- [ ] **E1-S2** Write Hardhat/Foundry integration tests: mock Aave Pool, mock DEX routers, verify `executeOperation` reverts on unprofitable trade.
- [ ] **E1-S3** Deploy `FlashLoanArbitrage.sol` to Polygon zkEVM testnet (Goerli / Cardona); verify DEX whitelist and `simulateProfit()`.
- [ ] **E1-S4** Deploy to Polygon zkEVM mainnet; record contract address in `.env` and `AGENTS.md`.
- [ ] **E1-S5** Implement an upgrade/proxy pattern (OpenZeppelin TransparentProxy) to allow parameter changes post-deployment.

---

### EPIC 2 — Real-Time Price Feed

> **Goal:** Replace the offline/mock price feed with live on-chain data and
> WebSocket event subscriptions for sub-block latency.

**Stories:**
- [ ] **E2-S1** Populate `POOL_CONFIGS` with verified QuickSwap V3, Uniswap V3, and SushiSwap pool addresses on Polygon zkEVM mainnet.
- [ ] **E2-S2** Implement WebSocket subscription to Uniswap V3 `Swap` events via `AsyncWeb3` for sub-block opportunity detection.
- [ ] **E2-S3** Add a `DEXFactory` scanner that auto-discovers new pools for configured token pairs using the V3 `PoolCreated` event.
- [ ] **E2-S4** Expand the token universe: MATIC, WBTC, LINK, AAVE, DAI, USDT pools.
- [ ] **E2-S5** Add SushiSwap V3 pool ABI and integrate into `DEXPriceFeed`.

---

### EPIC 3 — ML Model Production Quality

> **Goal:** Replace the synthetic-data bootstrap model with one trained on
> real historical execution data and delivered with continuous retraining.

**Stories:**
- [ ] **E3-S1** Persist opportunity and execution logs to PostgreSQL (`executions` table with all feature columns and `net_profit_usd` label).
- [ ] **E3-S2** Build an ETL pipeline that extracts features from the PostgreSQL log into a training dataset.
- [x] **E3-S3** Tune `GradientBoostingRegressor` hyperparameters via `GridSearchCV` on 90-day lookback window. *(implemented in `src/ml/training.tune_hyperparameters()`)*
- [ ] **E3-S4** Set up a weekly cron job that retrains the model and replaces `models/arbitrage_model.joblib`.
- [ ] **E3-S5** Add a model evaluation dashboard: RMSE, R², feature importances (Prometheus + Grafana or FastAPI endpoint).
- [x] **E3-S6** Explore a secondary `RandomForestRegressor` ensemble for variance reduction. *(implemented in `src/ml/model.EnsembleModel`)*

---

### EPIC 4 — Execution Reliability

> **Goal:** Harden the execution pipeline so that no money is lost to gas
> waste, RPC failures, or adverse market conditions.

**Stories:**
- [x] **E4-S1** Implement EIP-1559 gas estimation: cap `maxFeePerGas` at `MAX_GAS_PRICE_GWEI`; skip execution if base fee exceeds threshold. *(implemented in `PolygonClient.get_max_fee_params()`)*
- [ ] **E4-S2** Add per-hop slippage protection: compute `amountOutMinimum` from expected output minus 0.5% and pass to DEX swap functions.
- [x] **E4-S3** Implement exponential back-off retry in `ExecutionAgent` for RPC transient errors (up to 3 retries, max 8 s delay). *(implemented in `ExecutionAgent._execute_with_retry()`)*
- [x] **E4-S4** Add a nonce management service that serialises concurrent transaction submissions and handles stuck transactions. *(implemented in `polygon_client.NonceManager`)*
- [ ] **E4-S5** Implement MEV protection: submit transactions via Flashbots-compatible private mempool on zkEVM (or a bundler service).
- [x] **E4-S6** Extend the circuit-breaker: persist loss streaks across restarts; alert on Slack/Telegram after circuit trips. *(persistence implemented in `ExecutionAgent`; persists to `data/circuit_breaker.json`)*

---

### EPIC 5 — Observability & Operations

> **Goal:** Give operators full visibility into system health, opportunity
> quality, and P&L in real time.

**Stories:**
- [x] **E5-S1** Add a Prometheus metrics endpoint (`/metrics`) exposing: `cycle_duration_ms`, `opportunities_found_total`, `executions_total`, `profit_usd_total`, `ml_score_histogram`. *(implemented in `src/utils/metrics.ArbisyMetrics`)*
- [ ] **E5-S2** Build a Grafana dashboard with panels for: rolling P&L, ML score distribution, gas cost trend, circuit-breaker status.
- [x] **E5-S3** Implement structured JSON logging (via `python-json-logger`) and ship logs to a centralised store (e.g., Loki, CloudWatch). *(implemented in `src/utils/logger.py`; enable with `LOG_JSON=1`)*
- [ ] **E5-S4** Build a minimal FastAPI + WebSocket dashboard: live opportunity feed, execution log, current model metrics.
- [x] **E5-S5** Add health-check endpoint (`/healthz`) that verifies RPC connectivity and contract reachability; integrate with a watchdog (systemd / Docker). *(implemented in `src/utils/health.py`)*

---

### EPIC 6 — Advanced Arbitrage Strategies

> **Goal:** Expand the opportunity search space beyond simple triangular and
> cross-platform pairs.

**Stories:**
- [ ] **E6-S1** Implement 4-leg (quad-hop) arbitrage paths in the matrix solver and wire them through the execution pipeline.
- [ ] **E6-S2** Add Balancer V2 pool support: weighted pool price calculation and swap path encoding.
- [ ] **E6-S3** Add Curve Finance stable-swap pool support: invariant-based price calculation.
- [ ] **E6-S4** Explore cross-chain arbitrage via bridges (e.g., Polygon mainnet &lt;-&gt; zkEVM) as a future research item.
- [ ] **E6-S5** Evaluate a reinforcement-learning agent (using `stable-baselines3` sklearn-compatible wrapper) as an upgrade path for the ML model.

---

### EPIC 7 — Security Hardening

> **Goal:** Ensure the system is robust against adversarial on-chain
> conditions and off-chain configuration errors.

**Stories:**
- [ ] **E7-S1** Commission an independent security audit of `FlashLoanArbitrage.sol`.
- [x] **E7-S2** Implement replay protection: include a nonce in ABI-encoded `params`; reject calls with an already-used nonce. *(implemented: `usedNonces` mapping + `decodeSwapPathWithNonce` in `ArbitrageLib`; Python side generates fresh nonce via `secrets.token_bytes(32)` in `FlashLoan.encode_swap_path()`)*
- [x] **E7-S3** Add a pause mechanism (`Pausable`) to the contract: owner can halt all flash-loan execution instantly. *(implemented: `paused` bool + `whenNotPaused` modifier + `pause()`/`unpause()` admin functions)*
- [x] **E7-S4** Restrict `withdrawToken` to a configurable allow-list of destination addresses. *(implemented: `allowedWithdrawTo` mapping + `setAllowedWithdrawTo()` admin; both `withdrawToken` and `withdrawEth` check the list)*
- [x] **E7-S5** Add a maxSlippage parameter to the contract; revert inside `executeOperation` if slippage exceeds the on-chain limit. *(implemented: `maxSlippageBps` (default 50 = 0.5%) + per-hop `amountOutMinimum` in V3 swaps + `SlippageExceeded` revert)*
