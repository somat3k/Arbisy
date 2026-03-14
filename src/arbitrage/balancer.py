"""
Balancer V2 weighted-pool arbitrage detector (E6-S2).

Fetches on-chain pool data from the Balancer V2 Vault contract and computes
spot prices for weighted pools.  Detected price discrepancies between Balancer
pools and Uniswap V3-style pools are expressed as PoolPrice snapshots
compatible with DEXPriceFeed, TriangularArbitrage, and ArbitrageMatrix.

Balancer V2 Architecture
------------------------
A single Vault contract (0xBA12222...) manages all pool liquidity.
Each pool has a unique ``poolId`` (bytes32) and stores token balances inside
the Vault.

Spot price for a two-token weighted pool:
    spot_price(token_in → token_out) = (balance_out / weight_out) /
                                       (balance_in  / weight_in )

See: https://dev.balancer.fi/resources/pool-math/weighted-math

Usage
-----
    detector = BalancerArbitrage(client)
    await detector.load_pools(pool_ids=["0x..."])
    prices = await detector.fetch_prices()
    # Prices can be fed into DEXPriceFeed, TriangularArbitrage, etc.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from web3 import Web3

from src.blockchain.dex_price_feed import PoolPrice
from src.blockchain.polygon_client import PolygonClient
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Balancer V2 Vault ABI (minimal) ──────────────────────────────────────────
VAULT_ABI = [
    {
        "inputs": [{"name": "poolId", "type": "bytes32"}],
        "name": "getPoolTokens",
        "outputs": [
            {"name": "tokens",          "type": "address[]"},
            {"name": "balances",        "type": "uint256[]"},
            {"name": "lastChangeBlock", "type": "uint256"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [{"name": "poolId", "type": "bytes32"}],
        "name": "getPool",
        "outputs": [
            {"name": "pool",            "type": "address"},
            {"name": "specialization",  "type": "uint8"},
        ],
        "stateMutability": "view",
        "type": "function",
    },
]

# Weighted pool ABI (normalised weights)
WEIGHTED_POOL_ABI = [
    {
        "inputs": [],
        "name": "getNormalizedWeights",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "getSwapFeePercentage",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
]

_WAD = 10 ** 18  # Balancer uses 18-decimal fixed-point for weights


@dataclass
class BalancerPool:
    """Metadata for a Balancer V2 weighted pool."""
    pool_id: str
    pool_address: str
    tokens: List[str]
    balances: List[int]            # raw on-chain balances
    weights: List[float]           # normalised weights (sum = 1.0)
    swap_fee: float                # e.g. 0.003 for 0.3%
    token_symbols: List[str] = field(default_factory=list)
    token_decimals: List[int] = field(default_factory=list)
    last_update: float = field(default_factory=time.time)

    def spot_price(self, token_in_idx: int, token_out_idx: int) -> float:
        """
        Compute the spot price token_out per token_in using weighted-pool math.

            spot = (balance_out / weight_out) / (balance_in / weight_in)

        Balances are adjusted for token decimals before computing the price.
        """
        if (
            token_in_idx < 0 or token_in_idx >= len(self.tokens)
            or token_out_idx < 0 or token_out_idx >= len(self.tokens)
            or self.balances[token_in_idx] == 0
            or self.balances[token_out_idx] == 0
            or self.weights[token_in_idx] == 0
            or self.weights[token_out_idx] == 0
        ):
            return 0.0

        dec_in  = self.token_decimals[token_in_idx]  if self.token_decimals else 18
        dec_out = self.token_decimals[token_out_idx] if self.token_decimals else 18

        # Human-readable balance = raw / 10^decimals
        bal_in  = self.balances[token_in_idx]  / 10 ** dec_in
        bal_out = self.balances[token_out_idx] / 10 ** dec_out

        w_in  = self.weights[token_in_idx]
        w_out = self.weights[token_out_idx]

        return (bal_out / w_out) / (bal_in / w_in)


class BalancerArbitrage:
    """
    Fetches Balancer V2 pool data and exposes prices as PoolPrice snapshots.

    Parameters
    ----------
    client:  PolygonClient instance.
    """

    DEX_NAME = "balancer_v2"

    def __init__(self, client: PolygonClient) -> None:
        self._client = client
        cfg = get_config()
        vault_addr = cfg.balancer_vault
        self._vault_address = Web3.to_checksum_address(vault_addr) if vault_addr else None
        self._vault = None
        if self._vault_address:
            self._vault = self._client.w3s.eth.contract(
                address=self._vault_address,
                abi=VAULT_ABI,
            )
        self._pools: Dict[str, BalancerPool] = {}
        self._token_meta: Dict[str, Tuple[int, str]] = {}  # address → (decimals, symbol)

    @property
    def enabled(self) -> bool:
        return self._vault is not None

    async def _get_token_meta(self, address: str) -> Tuple[int, str]:
        if address not in self._token_meta:
            try:
                dec = await self._client.get_token_decimals(address)
                sym = await self._client.get_token_symbol(address)
            except Exception:
                dec, sym = 18, "???"
            self._token_meta[address] = (dec, sym)
        return self._token_meta[address]

    async def load_pools(self, pool_ids: List[str]) -> List[BalancerPool]:
        """
        Load pool metadata from the Vault for the given pool IDs.

        Parameters
        ----------
        pool_ids: List of Balancer V2 pool IDs (bytes32 hex strings).

        Returns
        -------
        List of BalancerPool objects with populated weights and balances.
        """
        if not self.enabled:
            log.debug("BalancerArbitrage: Vault not configured — skipping")
            return []

        loop = asyncio.get_event_loop()
        results: List[BalancerPool] = []

        for pool_id in pool_ids:
            try:
                # 1. Get pool address
                pool_addr_raw, _ = await loop.run_in_executor(
                    None, self._vault.functions.getPool(pool_id).call
                )
                pool_addr = pool_addr_raw

                # 2. Get token balances from Vault
                tokens_raw, balances_raw, _ = await loop.run_in_executor(
                    None, self._vault.functions.getPoolTokens(pool_id).call
                )
                tokens   = [t.lower() for t in tokens_raw]
                balances = list(balances_raw)

                # 3. Get normalised weights from pool contract
                pool_contract = self._client.w3s.eth.contract(
                    address=Web3.to_checksum_address(pool_addr),
                    abi=WEIGHTED_POOL_ABI,
                )
                try:
                    raw_weights = await loop.run_in_executor(
                        None, pool_contract.functions.getNormalizedWeights().call
                    )
                    weights = [w / _WAD for w in raw_weights]
                except Exception:
                    # Fallback: equal weights
                    n = len(tokens)
                    weights = [1.0 / n] * n

                # 4. Get swap fee
                try:
                    raw_fee = await loop.run_in_executor(
                        None, pool_contract.functions.getSwapFeePercentage().call
                    )
                    swap_fee = raw_fee / _WAD
                except Exception:
                    swap_fee = 0.003

                # 5. Fetch token metadata
                decimals = []
                symbols  = []
                for tok in tokens:
                    dec, sym = await self._get_token_meta(tok)
                    decimals.append(dec)
                    symbols.append(sym)

                pool = BalancerPool(
                    pool_id=pool_id,
                    pool_address=pool_addr.lower(),
                    tokens=tokens,
                    balances=balances,
                    weights=weights,
                    swap_fee=swap_fee,
                    token_symbols=symbols,
                    token_decimals=decimals,
                )
                self._pools[pool_id] = pool
                results.append(pool)
                log.info(
                    "Loaded Balancer pool %s — %d tokens: %s",
                    pool_id[:16], len(tokens), " / ".join(symbols),
                )

            except Exception as exc:
                log.warning("Failed to load Balancer pool %s: %s", pool_id[:16], exc)

        return results

    async def fetch_prices(self) -> List[PoolPrice]:
        """
        Refresh balances from the Vault for all loaded pools and
        return PoolPrice snapshots for all token pairs.

        Returns
        -------
        List of PoolPrice objects (one per ordered token pair per pool).
        """
        if not self.enabled or not self._pools:
            return []

        loop = asyncio.get_event_loop()
        prices: List[PoolPrice] = []
        block_num = await self._client.get_block_number()

        for pool_id, pool in self._pools.items():
            try:
                # Refresh balances
                _, fresh_balances, _ = await loop.run_in_executor(
                    None, self._vault.functions.getPoolTokens(pool_id).call
                )
                pool.balances = list(fresh_balances)
                pool.last_update = time.time()
            except Exception as exc:
                log.warning("Failed to refresh Balancer pool %s balances: %s", pool_id[:16], exc)
                continue

            n = len(pool.tokens)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    p_ij = pool.spot_price(i, j)
                    if p_ij <= 0:
                        continue
                    p_ji = pool.spot_price(j, i)

                    prices.append(PoolPrice(
                        dex_name=self.DEX_NAME,
                        pool_address=pool.pool_address,
                        token0=pool.tokens[i],
                        token1=pool.tokens[j],
                        token0_symbol=pool.token_symbols[i] if pool.token_symbols else "???",
                        token1_symbol=pool.token_symbols[j] if pool.token_symbols else "???",
                        price_token1_per_token0=p_ij,
                        price_token0_per_token1=p_ji if p_ji > 0 else (1.0 / p_ij),
                        sqrt_price_x96=0,    # not applicable for Balancer
                        tick=0,
                        liquidity=sum(pool.balances),
                        fee_bps=int(pool.swap_fee * 10_000),
                        block_number=block_num,
                    ))

        log.debug("BalancerArbitrage: %d price pairs from %d pools", len(prices), len(self._pools))
        return prices

    def get_loaded_pool_ids(self) -> List[str]:
        return list(self._pools.keys())
