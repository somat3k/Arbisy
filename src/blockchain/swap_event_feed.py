"""
Real-time Uniswap V3 Swap event polling (E2-S2).

Monitors ``Swap`` events emitted by V3-compatible pool contracts by
periodically calling ``eth_getLogs`` on an HTTP RPC endpoint.  Each
new swap triggers a price-update callback, enabling sub-block opportunity
detection at configurable poll intervals (default 0.5 s).

Note: The implementation uses ``eth_getLogs`` polling rather than a true
WebSocket ``eth_subscribe`` because most commercial RPC providers do not
support ``eth_subscribe`` for zkEVM endpoints, or charge extra for it.
To switch to a genuine WS subscription, replace ``_poll_swap_logs`` with
``w3.eth.subscribe("logs", filter)`` and wire ``client.ws_url`` instead of
``client.rpc_url`` in ``subscribe()``.

Usage
-----
    feed = SwapEventFeed(client, on_price_update=my_callback)
    await feed.subscribe([("uniswap_v3", "0xPoolAddress")])
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, List, Optional, Set

from web3 import AsyncWeb3, Web3
from web3.types import EventData

from src.blockchain.dex_price_feed import PoolPrice, sqrt_price_x96_to_price
from src.blockchain.polygon_client import PolygonClient
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── Uniswap V3 Swap event ABI ─────────────────────────────────────────────────
SWAP_EVENT_ABI = {
    "anonymous": False,
    "inputs": [
        {"indexed": True,  "name": "sender",       "type": "address"},
        {"indexed": True,  "name": "recipient",     "type": "address"},
        {"indexed": False, "name": "amount0",       "type": "int256"},
        {"indexed": False, "name": "amount1",       "type": "int256"},
        {"indexed": False, "name": "sqrtPriceX96",  "type": "uint160"},
        {"indexed": False, "name": "liquidity",     "type": "uint128"},
        {"indexed": False, "name": "tick",          "type": "int24"},
    ],
    "name": "Swap",
    "type": "event",
}

# Pool ABI (minimal — same as DEXPriceFeed.POOL_ABI)
_POOL_ABI_MIN = [
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
    SWAP_EVENT_ABI,
]

PriceCallback = Callable[[PoolPrice], Awaitable[None]]


@dataclass
class SwapEvent:
    """Decoded Uniswap V3 Swap event."""
    pool_address: str
    sender: str
    recipient: str
    amount0: int
    amount1: int
    sqrt_price_x96: int
    liquidity: int
    tick: int
    block_number: int
    tx_hash: str


class SwapEventFeed:
    """
    Subscribes to Uniswap V3 ``Swap`` events for a set of pool addresses.

    On each swap the feed:
    1. Decodes the new sqrtPriceX96 from the event.
    2. Converts it to a human-readable ``PoolPrice`` snapshot.
    3. Calls the optional ``on_price_update`` async callback.

    Parameters
    ----------
    client:           PolygonClient — provides ws_url and token metadata.
    on_price_update:  Async callback invoked with each new PoolPrice.
    dex_name_map:     Optional dict mapping pool_address → dex_name.
                      If omitted, dex_name defaults to ``"unknown"``.
    """

    def __init__(
        self,
        client: PolygonClient,
        on_price_update: Optional[PriceCallback] = None,
        dex_name_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self._client = client
        self._on_price_update = on_price_update
        # Normalise all keys to lower-case so lookups via pool_address.lower() always hit
        self._dex_name_map: Dict[str, str] = (
            {k.lower(): v for k, v in dex_name_map.items()} if dex_name_map else {}
        )
        self._token_meta: Dict[str, tuple] = {}   # address → (decimals, symbol)
        self._pool_meta: Dict[str, dict] = {}     # pool → {token0, token1, fee}
        self._subscriptions: Set[str] = set()
        self._running = False
        self._tasks: List[asyncio.Task] = []

    # ── Pool metadata cache ───────────────────────────────────────────────────

    async def _load_pool_meta(self, pool_address: str) -> None:
        """Fetch and cache token0, token1, fee for a pool."""
        if pool_address in self._pool_meta:
            return
        checksum = Web3.to_checksum_address(pool_address)
        contract = self._client.w3s.eth.contract(address=checksum, abi=_POOL_ABI_MIN)
        loop = asyncio.get_event_loop()
        try:
            token0 = await loop.run_in_executor(None, contract.functions.token0().call)
            token1 = await loop.run_in_executor(None, contract.functions.token1().call)
            fee    = await loop.run_in_executor(None, contract.functions.fee().call)
            self._pool_meta[pool_address] = {
                "token0": token0.lower(),
                "token1": token1.lower(),
                "fee":    fee,
            }
        except Exception as exc:
            log.warning("Could not load pool metadata for %s: %s", pool_address, exc)
            self._pool_meta[pool_address] = {"token0": "", "token1": "", "fee": 3000}

    async def _get_token_meta(self, address: str):
        if address not in self._token_meta:
            try:
                dec = await self._client.get_token_decimals(address)
                sym = await self._client.get_token_symbol(address)
            except Exception:
                dec, sym = 18, "???"
            self._token_meta[address] = (dec, sym)
        return self._token_meta[address]

    # ── Swap event → PoolPrice ────────────────────────────────────────────────

    async def _swap_to_pool_price(
        self,
        pool_address: str,
        swap: SwapEvent,
    ) -> Optional[PoolPrice]:
        """Convert a raw SwapEvent into a PoolPrice snapshot."""
        await self._load_pool_meta(pool_address)
        meta = self._pool_meta[pool_address]
        token0 = meta["token0"]
        token1 = meta["token1"]
        fee    = meta["fee"]

        if not token0 or not token1 or swap.sqrt_price_x96 == 0:
            return None

        dec0, sym0 = await self._get_token_meta(token0)
        dec1, sym1 = await self._get_token_meta(token1)

        p1_per_0, p0_per_1 = sqrt_price_x96_to_price(
            swap.sqrt_price_x96, dec0, dec1
        )

        dex_name = self._dex_name_map.get(pool_address.lower(), "unknown")

        return PoolPrice(
            dex_name=dex_name,
            pool_address=pool_address,
            token0=token0,
            token1=token1,
            token0_symbol=sym0,
            token1_symbol=sym1,
            price_token1_per_token0=p1_per_0,
            price_token0_per_token1=p0_per_1,
            sqrt_price_x96=swap.sqrt_price_x96,
            tick=swap.tick,
            liquidity=swap.liquidity,
            fee_bps=fee // 100,
            block_number=swap.block_number,
        )

    # ── Subscription loop ─────────────────────────────────────────────────────

    async def _poll_swap_logs(
        self,
        pool_address: str,
        ws_w3: AsyncWeb3,
        poll_interval: float = 0.5,
    ) -> None:
        """
        Poll for new Swap events on a single pool using eth_getLogs.

        This approach is compatible with providers that don't support
        ``eth_subscribe`` (most HTTP and many WS RPC endpoints).
        """
        checksum = Web3.to_checksum_address(pool_address)
        contract = ws_w3.eth.contract(address=checksum, abi=_POOL_ABI_MIN)

        # Build the Swap event topic0
        swap_topic = Web3.keccak(
            text="Swap(address,address,int256,int256,uint160,uint128,int24)"
        ).hex()

        last_block = await ws_w3.eth.block_number
        log.debug("SwapEventFeed: watching pool %s from block %d", pool_address, last_block)

        while self._running:
            try:
                current_block = await ws_w3.eth.block_number
                if current_block <= last_block:
                    await asyncio.sleep(poll_interval)
                    continue

                logs = await ws_w3.eth.get_logs({
                    "address":   checksum,
                    "topics":    [swap_topic],
                    "fromBlock": last_block + 1,
                    "toBlock":   current_block,
                })

                for raw_log in logs:
                    try:
                        event = contract.events.Swap().process_log(raw_log)
                        swap = SwapEvent(
                            pool_address=pool_address,
                            sender=event["args"]["sender"],
                            recipient=event["args"]["recipient"],
                            amount0=event["args"]["amount0"],
                            amount1=event["args"]["amount1"],
                            sqrt_price_x96=event["args"]["sqrtPriceX96"],
                            liquidity=event["args"]["liquidity"],
                            tick=event["args"]["tick"],
                            block_number=event["blockNumber"],
                            tx_hash=event["transactionHash"].hex(),
                        )
                        price = await self._swap_to_pool_price(pool_address, swap)
                        if price and self._on_price_update:
                            await self._on_price_update(price)
                            log.debug(
                                "SwapEvent: pool=%s block=%d price=%.6f",
                                pool_address[:10],
                                swap.block_number,
                                price.price_token1_per_token0,
                            )
                    except Exception as exc:
                        log.warning("Failed to decode swap log for %s: %s", pool_address, exc)

                last_block = current_block

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.error("SwapEventFeed error on %s: %s", pool_address, exc, exc_info=True)

            await asyncio.sleep(poll_interval)

    async def subscribe(
        self,
        pool_configs: List[tuple],
        poll_interval: float = 0.5,
    ) -> None:
        """
        Start event polling for a list of (dex_name, pool_address) tuples.

        Updates ``dex_name_map`` automatically from the provided configs.
        This coroutine runs until ``stop()`` is called or the task is
        cancelled.

        Parameters
        ----------
        pool_configs:   List of (dex_name, pool_address) pairs to watch.
        poll_interval:  Seconds between ``eth_getLogs`` scans per pool.
        """
        for dex_name, pool_address in pool_configs:
            self._dex_name_map[pool_address.lower()] = dex_name
            self._subscriptions.add(pool_address.lower())

        cfg_rpc = self._client.rpc_url
        ws_w3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(cfg_rpc))

        self._running = True
        log.info(
            "SwapEventFeed: subscribing to %d pools",
            len(pool_configs),
        )

        tasks = [
            asyncio.create_task(
                self._poll_swap_logs(pool_address, ws_w3, poll_interval),
                name=f"swap_feed_{pool_address[:10]}",
            )
            for _, pool_address in pool_configs
        ]
        self._tasks = tasks

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            for task in tasks:
                task.cancel()

    def stop(self) -> None:
        """Signal the feed to stop all polling tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        log.info("SwapEventFeed: stopped")
