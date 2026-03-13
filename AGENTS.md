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
- [ ] Add circuit-breaker: halt execution if 3 consecutive losses occur
- [ ] Add slippage protection (max 0.5% slippage per hop)

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
