"""
DEX Factory scanner — auto-discovers new V3 pools (E2-S3).

Queries Uniswap V3-compatible factory contracts for ``PoolCreated`` events
to automatically find pools for any combination of configured token pairs
and fee tiers, without requiring manual configuration of pool addresses.

Usage
-----
    scanner = DEXFactoryScanner(client)
    new_pools = await scanner.discover_pools(token_pairs=[("0xWETH", "0xUSDC")])
    # Returns [(dex_name, pool_address), ...]
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from web3 import Web3

from src.blockchain.polygon_client import PolygonClient
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── PoolCreated event ABI ─────────────────────────────────────────────────────
POOL_CREATED_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True,  "name": "token0",      "type": "address"},
        {"indexed": True,  "name": "token1",      "type": "address"},
        {"indexed": True,  "name": "fee",         "type": "uint24"},
        {"indexed": False, "name": "tickSpacing", "type": "int24"},
        {"indexed": False, "name": "pool",        "type": "address"},
    ],
    "name": "PoolCreated",
    "type": "event",
}

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
    },
    POOL_CREATED_ABI,
]

# Fee tiers common across Uniswap V3, QuickSwap V3, SushiSwap V3
FEE_TIERS: List[int] = [100, 500, 3000, 10000]

_ZERO_ADDR = "0x" + "0" * 40


@dataclass
class DiscoveredPool:
    """A pool discovered via the factory's getPool or PoolCreated events."""
    dex_name: str
    pool_address: str
    token0: str
    token1: str
    fee: int           # Uniswap fee units (e.g. 3000 = 0.30%)
    fee_bps: int       # fee / 100


class DEXFactoryScanner:
    """
    Queries Uniswap V3-compatible factory contracts for pools.

    Supports two discovery modes:

    1. ``discover_pools`` — calls ``getPool(tokenA, tokenB, fee)`` for each
       combination of the supplied token pairs and fee tiers.  Fast, targeted.

    2. ``scan_pool_created_events`` — fetches historical ``PoolCreated`` events
       in a block range to build a complete on-chain pool registry.  Slower
       but exhaustive.

    Parameters
    ----------
    client:  PolygonClient instance.
    """

    def __init__(self, client: PolygonClient) -> None:
        self._client = client
        cfg = get_config()
        # Map from dex_name → factory address
        self._factories: Dict[str, str] = {
            "uniswap_v3":   cfg.uniswap_v3_factory,
            "quickswap_v3": cfg.quickswap_v3_factory,
            "sushiswap":    cfg.sushiswap_factory,
        }
        self._known_pools: Set[str] = set()   # lowercased pool addresses

    def _get_factory(self, factory_address: str):
        checksum = Web3.to_checksum_address(factory_address)
        return self._client.w3s.eth.contract(address=checksum, abi=FACTORY_ABI)

    # ── Targeted discovery ────────────────────────────────────────────────────

    async def discover_pools(
        self,
        token_pairs: List[Tuple[str, str]],
        fee_tiers: Optional[List[int]] = None,
        dex_names: Optional[List[str]] = None,
    ) -> List[DiscoveredPool]:
        """
        For each (tokenA, tokenB) pair and fee tier, call ``getPool`` on all
        configured DEX factories and return non-zero pool addresses.

        Parameters
        ----------
        token_pairs: List of (tokenA_address, tokenB_address) tuples.
        fee_tiers:   Fee tiers to probe.  Defaults to all of FEE_TIERS.
        dex_names:   Subset of DEXes to query; defaults to all configured.
        """
        tiers = fee_tiers or FEE_TIERS
        names = dex_names or list(self._factories.keys())
        results: List[DiscoveredPool] = []
        loop = asyncio.get_event_loop()

        for dex_name in names:
            factory_addr = self._factories.get(dex_name)
            if not factory_addr:
                continue
            factory = self._get_factory(factory_addr)

            for token_a, token_b in token_pairs:
                # UniswapV3-style factories require token0 < token1 (lexicographic address order)
                ta_cs = Web3.to_checksum_address(token_a)
                tb_cs = Web3.to_checksum_address(token_b)
                # Sort so the getPool call always matches how the factory indexes the pair
                t0_cs, t1_cs = (ta_cs, tb_cs) if ta_cs.lower() < tb_cs.lower() else (tb_cs, ta_cs)
                for fee in tiers:
                    try:
                        pool_addr = await loop.run_in_executor(
                            None,
                            factory.functions.getPool(
                                t0_cs,
                                t1_cs,
                                fee,
                            ).call,
                        )
                        if pool_addr.lower() == _ZERO_ADDR.lower():
                            continue
                        if pool_addr.lower() in self._known_pools:
                            continue
                        self._known_pools.add(pool_addr.lower())
                        # Record the canonical (sorted) token order
                        t0 = t0_cs.lower()
                        t1 = t1_cs.lower()
                        pool = DiscoveredPool(
                            dex_name=dex_name,
                            pool_address=pool_addr,
                            token0=t0,
                            token1=t1,
                            fee=fee,
                            fee_bps=fee // 100,
                        )
                        results.append(pool)
                        log.info(
                            "Discovered pool %s on %s (fee=%d): %s/%s",
                            pool_addr[:10], dex_name, fee, token_a[:10], token_b[:10],
                        )
                    except Exception as exc:
                        log.debug(
                            "getPool failed for %s/%s fee=%d on %s: %s",
                            token_a[:10], token_b[:10], fee, dex_name, exc,
                        )

        return results

    # ── Historical event scan ─────────────────────────────────────────────────

    async def scan_pool_created_events(
        self,
        from_block: int,
        to_block: Optional[int] = None,
        dex_names: Optional[List[str]] = None,
        chunk_size: int = 10_000,
    ) -> List[DiscoveredPool]:
        """
        Fetch historical ``PoolCreated`` events to discover all pools deployed
        in a block range.

        Parameters
        ----------
        from_block:  Starting block number.
        to_block:    Ending block number; defaults to the latest block.
        dex_names:   DEX factories to query; defaults to all.
        chunk_size:  Number of blocks per ``eth_getLogs`` request.
        """
        if to_block is None:
            to_block = await self._client.get_block_number()

        names = dex_names or list(self._factories.keys())
        results: List[DiscoveredPool] = []

        pool_created_topic = Web3.keccak(
            text="PoolCreated(address,address,uint24,int24,address)"
        ).hex()

        for dex_name in names:
            factory_addr = self._factories.get(dex_name)
            if not factory_addr:
                continue
            factory = self._get_factory(factory_addr)
            checksum = Web3.to_checksum_address(factory_addr)

            # Process in chunks to avoid RPC size limits
            start = from_block
            while start <= to_block:
                end = min(start + chunk_size - 1, to_block)
                try:
                    raw_logs = await self._client.w3.eth.get_logs({
                        "address":   checksum,
                        "topics":    [pool_created_topic],
                        "fromBlock": start,
                        "toBlock":   end,
                    })
                    for raw_log in raw_logs:
                        try:
                            event = factory.events.PoolCreated().process_log(raw_log)
                            pool_addr = event["args"]["pool"]
                            if pool_addr.lower() in self._known_pools:
                                continue
                            self._known_pools.add(pool_addr.lower())
                            pool = DiscoveredPool(
                                dex_name=dex_name,
                                pool_address=pool_addr,
                                token0=event["args"]["token0"].lower(),
                                token1=event["args"]["token1"].lower(),
                                fee=event["args"]["fee"],
                                fee_bps=event["args"]["fee"] // 100,
                            )
                            results.append(pool)
                        except Exception as exc:
                            log.debug("PoolCreated decode error: %s", exc)

                    log.debug(
                        "Scanned %s blocks %d-%d — %d pools so far",
                        dex_name, start, end, len(results),
                    )
                except Exception as exc:
                    log.warning(
                        "eth_getLogs failed for %s blocks %d-%d: %s",
                        dex_name, start, end, exc,
                    )
                start = end + 1

        return results

    def as_pool_configs(
        self,
        discovered: List[DiscoveredPool],
    ) -> List[Tuple[str, str]]:
        """
        Convert a list of DiscoveredPool into the (dex_name, pool_address)
        tuples expected by DEXPriceFeed.
        """
        return [(p.dex_name, p.pool_address) for p in discovered]
