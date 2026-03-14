# Arbisy — Atomic Arbitrage System

An end-to-end flash-loan arbitrage bot operating on **Polygon zkEVM**, using:

- **Aave V3** flash loans (Pool: `0xb47673b7a73D78743AFF1487AF69dBB5763F00cA`)
- **Uniswap V3**, **QuickSwap V3**, **SushiSwap** price feeds
- **Triangular + cross-platform** arbitrage detection
- **sklearn** ML opportunity scoring (no torch/onnx/tensorflow)
- **Multi-provider AI agents** (Groq, OpenRouter, Gemini, OpenAI)
- **Find → Simulate → Release** execution cycle

---

## Quick Start

```bash
cp .env.example .env          # fill in RPC URL and private key
pip install -r requirements.txt
python main.py
```

## Architecture

```
main.py  ──►  PayloadCommunicator  ──►  ArbitrageAgent
                                   ──►  ExecutionAgent
                                   ──►  AnalysisAgent
                    │
                    ▼
          DEXPriceFeed (Uniswap V3, QuickSwap, SushiSwap)
          TriangularArbitrage / CrossPlatformArbitrage
          MLModel (sklearn GradientBoostingRegressor)
          FlashLoan (Aave V3 Pool)
```

## Directory Structure

```
contracts/          Solidity — FlashLoanArbitrage + interfaces + libs
src/agents/         AI agent classes (base, arbitrage, execution, analysis)
src/arbitrage/      Opportunity detectors (triangular, cross-platform, matrix)
src/blockchain/     Web3 client, DEX price feeds, flash-loan trigger
src/ml/             Model, training, inference, feature engineering
src/payload/        Async inter-component messaging bus
src/utils/          Config and logging utilities
tests/              Pytest test suite
```

## Security

- Private key is **never** logged or committed
- Reentrancy guard on every contract entry point
- Profit check inside `executeOperation` — reverts if unprofitable
- ML score threshold: 0.60; gas cap: 500 gwei

## License

MIT
