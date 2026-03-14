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

    # ── Database (E3-S1) ─────────────────────────────────────────────────────
    database_url: str = field(default_factory=lambda: _get(
        "DATABASE_URL", ""
    ))

    # ── Token universe (E2-S4) — Polygon zkEVM mainnet addresses ─────────────
    # WETH (Wrapped Ether)
    token_weth: str = field(default_factory=lambda: _get(
        "TOKEN_WETH", "0x4F9A0e7FD2Bf6067db6994CF12E4495Df938E6e9"
    ))
    # USDC (USD Coin — bridged)
    token_usdc: str = field(default_factory=lambda: _get(
        "TOKEN_USDC", "0xA8CE8aee21bC2A48a5EF670afCc9274C7bbbC035"
    ))
    # USDT (Tether USD — bridged)
    token_usdt: str = field(default_factory=lambda: _get(
        "TOKEN_USDT", "0x1E4a5963aBFD975d8c9021ce480b42188849D41d"
    ))
    # WMATIC (Wrapped MATIC)
    token_wmatic: str = field(default_factory=lambda: _get(
        "TOKEN_WMATIC", "0xa2036f0538221a77A3937F1379699f44945018d0"
    ))
    # WBTC (Wrapped BTC)
    token_wbtc: str = field(default_factory=lambda: _get(
        "TOKEN_WBTC", "0xEA034fb02eB1808C2cc3adbC15f447B93CbE08e1"
    ))
    # LINK (Chainlink)
    token_link: str = field(default_factory=lambda: _get(
        "TOKEN_LINK", "0x4B16e4752711A7ABEc32799C976F3CeFc0111f2B"
    ))
    # AAVE
    token_aave: str = field(default_factory=lambda: _get(
        "TOKEN_AAVE", "0x68791cfe079814c46e0808bad312641223e8ca0e"
    ))
    # DAI (Multi-Collateral DAI)
    token_dai: str = field(default_factory=lambda: _get(
        "TOKEN_DAI", "0xC5015b9d9161Dca7e18e32f6f25C4aD850731Fd4"
    ))

    # ── Balancer V2 (E6-S2) ───────────────────────────────────────────────────
    balancer_vault: str = field(default_factory=lambda: _get(
        "BALANCER_VAULT", "0xBA12222222228d8Ba445958a75a0704d566BF2C8"
    ))


# Module-level singleton
_config: Optional[Config] = None


def get_config() -> Config:
    """Return the global Config singleton."""
    global _config
    if _config is None:
        _config = Config()
    return _config
