"""
DEX price feed for Polygon zkEVM.

Reads Uniswap V3-compatible pool state (sqrtPriceX96, tick, liquidity)
from on-chain contracts and converts to human-readable prices.

Supported DEXes
---------------
- Uniswap V3
- QuickSwap V3  (same pool interface)
- SushiSwap V3  (same pool interface)
"""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from web3 import Web3

from src.blockchain.polygon_client import PolygonClient
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Pool ABI (minimal — slot0 + liquidity) ────────────────────────────────────
POOL_ABI = [
    {
        "inputs": [],
        "name": "slot0",
        "outputs": [
            {"name": "sqrtPriceX96",             "type": "uint160"},
            {"name": "tick",                     "type": "int24"},
            {"name": "observationIndex",         "type": "uint16"},
            {"name": "observationCardinality",   "type": "uint16"},
            {"name": "observationCardinalityNext","type": "uint16"},
            {"name": "feeProtocol",              "type": "uint8"},
            {"name": "unlocked",                 "type": "bool"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "liquidity",
        "outputs": [{"name": "", "type": "uint128"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "token0",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "token1",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "fee",
        "outputs": [{"name": "", "type": "uint24"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# ── Factory ABI (getPool) ─────────────────────────────────────────────────────
FACTORY_ABI = [
    {
        "inputs": [
            {"name": "tokenA", "type": "address"},
            {"name": "tokenB", "type": "address"},
            {"name": "fee",    "type": "uint24"},
        ],
        "name": "getPool",
        "outputs": [{"name": "pool", "type": "address"}],
        "stateMutability": "view",
        "type": "function",
    }
]

# Common fee tiers (bps * 100 = fee in pool units)
FEE_TIERS = [500, 3000, 10000]


@dataclass
class PoolPrice:
    """Price snapshot from a single DEX pool."""
    dex_name: str
    pool_address: str
    token0: str
    token1: str
    token0_symbol: str
    token1_symbol: str
    price_token1_per_token0: float   # How many token1 per 1 token0
    price_token0_per_token1: float
    sqrt_price_x96: int
    tick: int
    liquidity: int
    fee_bps: int
    block_number: int
    timestamp: float = field(default_factory=time.time)

    @property
    def spread_vs(self) -> str:
        return f"{self.token0_symbol}/{self.token1_symbol}"


def sqrt_price_x96_to_price(
    sqrt_price_x96: int,
    decimals0: int,
    decimals1: int,
) -> Tuple[float, float]:
    """
    Convert Uniswap V3 sqrtPriceX96 to human-readable prices.

    Returns (price_token1_per_token0, price_token0_per_token1).
    """
    Q96 = 2 ** 96
    sqrt_price = sqrt_price_x96 / Q96
    raw_price = sqrt_price ** 2  # token1 per token0 in raw units

    # Adjust for decimals
    decimal_adj = 10 ** decimals0 / 10 ** decimals1
    price_1_per_0 = raw_price * decimal_adj
    price_0_per_1 = 1 / price_1_per_0 if price_1_per_0 > 0 else 0.0
    return price_1_per_0, price_0_per_1


class DEXPriceFeed:
    """
    Polls multiple Uniswap V3-compatible DEX pools for current prices.

    Parameters
    ----------
    client:        PolygonClient instance.
    pool_configs:  List of (dex_name, pool_address) tuples to monitor.
    """

    def __init__(
        self,
        client: PolygonClient,
        pool_configs: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self._client = client
        self._pool_configs = pool_configs or self._default_pools()
        self._prices: Dict[str, PoolPrice] = {}
        self._token_decimals: Dict[str, int] = {}
        self._token_symbols: Dict[str, str] = {}

    @staticmethod
    def _default_pools() -> List[Tuple[str, str]]:
        """
        Return pool configs from the POOL_CONFIGS env var (comma-separated
        ``dex_name:pool_address`` pairs), or an empty list when none are set.

        Configure pools via the environment before starting:

            POOL_CONFIGS=uniswap_v3:0x<addr1>,quickswap_v3:0x<addr2>

        Known Polygon zkEVM pool addresses (examples — verify on-chain):
          - QuickSwap V3 WETH/USDC:  0x2f9DCc...  (look up on QuickSwap explorer)
          - Uniswap V3 WETH/USDC:    deploy varies (check Uniswap app)

        Without a valid POOL_CONFIGS the feed runs in offline/mock mode.
        """
        import os
        raw = os.getenv("POOL_CONFIGS", "")
        if not raw:
            return []
        pools: List[Tuple[str, str]] = []
        for entry in raw.split(","):
            entry = entry.strip()
            if ":" not in entry:
                continue
            dex_name, pool_addr = entry.split(":", 1)
            pools.append((dex_name.strip(), pool_addr.strip()))
        return pools

    def _get_pool_contract(self, pool_address: str):
        checksum = Web3.to_checksum_address(pool_address)
        return self._client.w3s.eth.contract(address=checksum, abi=POOL_ABI)

    async def _get_token_meta(self, token_address: str) -> Tuple[int, str]:
        """Return (decimals, symbol) for a token, with caching."""
        if token_address not in self._token_decimals:
            try:
                dec = await self._client.get_token_decimals(token_address)
                sym = await self._client.get_token_symbol(token_address)
            except Exception:
                dec, sym = 18, "???"
            self._token_decimals[token_address] = dec
            self._token_symbols[token_address] = sym
        return self._token_decimals[token_address], self._token_symbols[token_address]

    async def fetch_pool_price(
        self,
        dex_name: str,
        pool_address: str,
    ) -> Optional[PoolPrice]:
        """Fetch current price snapshot from one pool."""
        try:
            contract = self._get_pool_contract(pool_address)
            loop = asyncio.get_event_loop()

            # Run synchronous calls in executor to avoid blocking
            slot0     = await loop.run_in_executor(None, contract.functions.slot0().call)
            liq       = await loop.run_in_executor(None, contract.functions.liquidity().call)
            token0    = await loop.run_in_executor(None, contract.functions.token0().call)
            token1    = await loop.run_in_executor(None, contract.functions.token1().call)
            fee       = await loop.run_in_executor(None, contract.functions.fee().call)
            block_num = await self._client.get_block_number()

            dec0, sym0 = await self._get_token_meta(token0)
            dec1, sym1 = await self._get_token_meta(token1)

            sqrt_price_x96 = slot0[0]
            tick           = slot0[1]

            if sqrt_price_x96 == 0:
                return None

            p1_per_0, p0_per_1 = sqrt_price_x96_to_price(sqrt_price_x96, dec0, dec1)

            snapshot = PoolPrice(
                dex_name=dex_name,
                pool_address=pool_address,
                token0=token0,
                token1=token1,
                token0_symbol=sym0,
                token1_symbol=sym1,
                price_token1_per_token0=p1_per_0,
                price_token0_per_token1=p0_per_1,
                sqrt_price_x96=sqrt_price_x96,
                tick=tick,
                liquidity=liq,
                fee_bps=fee // 100,  # convert from pool units to bps
                block_number=block_num,
            )
            self._prices[pool_address] = snapshot
            return snapshot

        except Exception as exc:
            log.warning("Failed to fetch pool %s (%s): %s", pool_address, dex_name, exc)
            return None

    async def fetch_all_prices(self) -> List[PoolPrice]:
        """Fetch prices from all configured pools concurrently."""
        tasks = [
            self.fetch_pool_price(dex_name, pool_addr)
            for dex_name, pool_addr in self._pool_configs
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        prices = [r for r in results if isinstance(r, PoolPrice)]
        log.debug("Fetched %d/%d pool prices", len(prices), len(self._pool_configs))
        return prices

    async def get_price_for_pair(
        self,
        dex_name: str,
        token_a: str,
        token_b: str,
        fee: int = 3000,
    ) -> Optional[float]:
        """
        Return price of token_b per token_a from a factory-derived pool.
        `fee` is the Uniswap pool fee tier (e.g. 500, 3000, 10000).
        """
        cfg = get_config()
        factory_map = {
            "uniswap_v3":  cfg.uniswap_v3_factory,
            "quickswap_v3": cfg.quickswap_v3_factory,
            "sushiswap":   cfg.sushiswap_factory,
        }
        factory_addr = factory_map.get(dex_name)
        if not factory_addr:
            return None

        factory = self._client.w3s.eth.contract(
            address=Web3.to_checksum_address(factory_addr),
            abi=FACTORY_ABI,
        )
        loop = asyncio.get_event_loop()
        try:
            pool_addr = await loop.run_in_executor(
                None,
                factory.functions.getPool(
                    Web3.to_checksum_address(token_a),
                    Web3.to_checksum_address(token_b),
                    fee,
                ).call,
            )
        except Exception as exc:
            log.warning("getPool failed for %s/%s on %s: %s", token_a, token_b, dex_name, exc)
            return None

        if pool_addr == "0x" + "0" * 40:
            return None

        price_snap = await self.fetch_pool_price(dex_name, pool_addr)
        if price_snap is None:
            return None

        # Normalise direction: always return price_b_per_a
        if price_snap.token0.lower() == token_a.lower():
            return price_snap.price_token1_per_token0
        else:
            return price_snap.price_token0_per_token1

    def get_cached_prices(self) -> Dict[str, PoolPrice]:
        return dict(self._prices)

    def tick_to_price(self, tick: int, decimals0: int = 18, decimals1: int = 18) -> float:
        """Convert a Uniswap V3 tick to a price ratio."""
        raw = 1.0001 ** tick
        return raw * (10 ** decimals0) / (10 ** decimals1)
