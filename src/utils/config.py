"""
Configuration management.

Reads all settings from environment variables (via .env file).
Provides a single `Config` singleton used throughout the system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Required environment variable '{key}' is not set.")
    return val


def _get(key: str, default: str = "") -> str:
    return os.getenv(key, default)


@dataclass(frozen=True)
class Config:
    # ── Blockchain ────────────────────────────────────────────────────────────
    rpc_url: str = field(default_factory=lambda: _get(
        "POLYGON_ZKEVM_RPC_URL", "https://zkevm-rpc.com"
    ))
    ws_url: str = field(default_factory=lambda: _get(
        "POLYGON_ZKEVM_WS_URL", "wss://zkevm-rpc.com/ws"
    ))
    private_key: str = field(default_factory=lambda: _get("PRIVATE_KEY", ""))
    wallet_address: str = field(default_factory=lambda: _get("WALLET_ADDRESS", ""))

    # ── Aave V3 ───────────────────────────────────────────────────────────────
    aave_pool_address: str = field(default_factory=lambda: _get(
        "AAVE_POOL_ADDRESS", "0xb47673b7a73D78743AFF1487AF69dBB5763F00cA"
    ))
    flash_loan_contract: str = field(default_factory=lambda: _get(
        "FLASH_LOAN_ARBITRAGE_CONTRACT", ""
    ))
    flash_loan_premium_bps: int = field(default_factory=lambda: int(
        _get("FLASH_LOAN_PREMIUM_BPS", "5")
    ))

    # ── DEX factories ─────────────────────────────────────────────────────────
    uniswap_v3_factory: str = field(default_factory=lambda: _get(
        "UNISWAP_V3_FACTORY", "0x4B9f4d2435Ef65559567e5DbFC1BbB37abC43B57"
    ))
    quickswap_v3_factory: str = field(default_factory=lambda: _get(
        "QUICKSWAP_V3_FACTORY", "0x411b0fAcC3489691f28ad58c47006AF5E3Ab3A28"
    ))
    sushiswap_factory: str = field(default_factory=lambda: _get(
        "SUSHISWAP_FACTORY", "0xc35DADB65012eC5796536bD9864eD8773aBc74C4"
    ))

    # ── AI providers ─────────────────────────────────────────────────────────
    groq_api_key: str = field(default_factory=lambda: _get("GROQ_API_KEY", ""))
    openrouter_api_key: str = field(default_factory=lambda: _get("OPENROUTER_API_KEY", ""))
    gemini_api_key: str = field(default_factory=lambda: _get("GEMINI_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: _get("OPENAI_API_KEY", ""))

    # ── Bot parameters ────────────────────────────────────────────────────────
    min_profit_usd: float = field(default_factory=lambda: float(
        _get("MIN_PROFIT_USD", "5.0")
    ))
    max_gas_price_gwei: float = field(default_factory=lambda: float(
        _get("MAX_GAS_PRICE_GWEI", "500")
    ))
    ml_score_threshold: float = field(default_factory=lambda: float(
        _get("ML_SCORE_THRESHOLD", "0.60")
    ))
    poll_interval_seconds: float = field(default_factory=lambda: float(
        _get("POLL_INTERVAL_SECONDS", "2")
    ))

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = field(default_factory=lambda: _get("LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: _get("LOG_FILE", "logs/arbisy.log"))

    # ── Model ─────────────────────────────────────────────────────────────────
    model_path: str = field(default_factory=lambda: _get(
        "MODEL_PATH", "models/arbitrage_model.joblib"
    ))
    nn_model_path: str = field(default_factory=lambda: _get(
        "NN_MODEL_PATH", "models/neural_arb_model.joblib"
    ))


# Module-level singleton
_config: Optional[Config] = None


def get_config() -> Config:
    """Return the global Config singleton."""
    global _config
    if _config is None:
        _config = Config()
    return _config
