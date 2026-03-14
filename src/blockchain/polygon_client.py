"""
Polygon zkEVM Web3 client.

Manages HTTP and optional WebSocket connections to the Polygon zkEVM RPC.
Provides helper methods for gas estimation, block polling, and contract calls.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from web3 import AsyncWeb3, Web3
from web3.exceptions import TransactionNotFound
from web3.middleware import ExtraDataToPOAMiddleware
from web3.types import TxParams, Wei

from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# Minimal ERC-20 ABI for balance/approve checks
ERC20_ABI = [
    {
        "inputs": [{"name": "account", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [],
        "name": "symbol",
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount",  "type": "uint256"},
        ],
        "name": "approve",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "nonpayable",
        "type": "function",
    },
]


class PolygonClient:
    """
    Async Web3 client for Polygon zkEVM.

    Attributes
    ----------
    w3 : AsyncWeb3   Async provider for contract calls.
    w3s: Web3        Synchronous provider (fallback / gas estimation).
    """

    def __init__(self) -> None:
        cfg = get_config()
        self._rpc_url = cfg.rpc_url
        self._ws_url  = cfg.ws_url
        self.w3: AsyncWeb3 = AsyncWeb3(AsyncWeb3.AsyncHTTPProvider(self._rpc_url))
        self.w3s: Web3     = Web3(Web3.HTTPProvider(self._rpc_url))
        # zkEVM uses PoA consensus — inject middleware for extra-data field
        self.w3s.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        log.info("PolygonClient initialised → %s", self._rpc_url)

    # ── Connectivity ──────────────────────────────────────────────────────────

    async def is_connected(self) -> bool:
        try:
            return await self.w3.is_connected()
        except Exception:
            return False

    async def get_block_number(self) -> int:
        return await self.w3.eth.block_number

    async def get_block(self, block_number: int) -> Dict[str, Any]:
        return dict(await self.w3.eth.get_block(block_number))

    # ── Gas ───────────────────────────────────────────────────────────────────

    async def get_gas_price_gwei(self) -> float:
        """Return current gas price in gwei."""
        wei = await self.w3.eth.gas_price
        return float(Web3.from_wei(wei, "gwei"))

    async def estimate_gas(self, tx: TxParams) -> int:
        return await self.w3.eth.estimate_gas(tx)

    async def get_max_fee_params(self) -> Dict[str, Wei]:
        """
        Return EIP-1559 fee parameters suitable for a TxParams dict.
        Caps maxFeePerGas at MAX_GAS_PRICE_GWEI to enforce the execution rule.
        Falls back to legacy gasPrice if baseFee is unavailable.
        """
        cfg = get_config()
        max_gas_wei = Web3.to_wei(cfg.max_gas_price_gwei, "gwei")
        try:
            latest = await self.w3.eth.get_block("latest")
            base_fee: int = latest.get("baseFeePerGas", 0)  # type: ignore[assignment]
            max_priority = int(Web3.to_wei(30, "gwei"))
            max_fee = base_fee * 2 + max_priority
            # Enforce the configured gas cap (E4-S1)
            max_fee = min(max_fee, int(max_gas_wei))
            # EIP-1559 requires maxFeePerGas >= baseFeePerGas + maxPriorityFeePerGas.
            # After capping, ensure max_priority doesn't exceed what max_fee can cover.
            max_priority = min(max_priority, max(0, max_fee - base_fee))
            return {
                "maxFeePerGas":         Wei(max_fee),
                "maxPriorityFeePerGas": Wei(max_priority),
            }
        except Exception:
            gas_price = await self.w3.eth.gas_price
            gas_price = Wei(min(int(gas_price), int(max_gas_wei)))
            return {"gasPrice": gas_price}

    # ── ERC-20 helpers ────────────────────────────────────────────────────────

    def get_erc20_contract(self, address: str):
        checksum = Web3.to_checksum_address(address)
        return self.w3s.eth.contract(address=checksum, abi=ERC20_ABI)

    async def get_token_balance(self, token_address: str, wallet: str) -> int:
        """Return raw token balance (in wei-equivalent smallest unit)."""
        contract = self.get_erc20_contract(token_address)
        checksum_wallet = Web3.to_checksum_address(wallet)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, contract.functions.balanceOf(checksum_wallet).call
        )

    async def get_token_decimals(self, token_address: str) -> int:
        contract = self.get_erc20_contract(token_address)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, contract.functions.decimals().call)

    async def get_token_symbol(self, token_address: str) -> str:
        contract = self.get_erc20_contract(token_address)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, contract.functions.symbol().call)

    # ── Transaction ───────────────────────────────────────────────────────────

    async def send_transaction(
        self,
        tx: TxParams,
        private_key: str,
    ) -> str:
        """Sign and broadcast a transaction; return the tx hash hex."""
        signed = self.w3s.eth.account.sign_transaction(tx, private_key)
        raw = signed.raw_transaction
        tx_hash = await self.w3.eth.send_raw_transaction(raw)
        return tx_hash.hex()

    async def wait_for_receipt(
        self,
        tx_hash: str,
        timeout: int = 120,
    ) -> Dict[str, Any]:
        """Poll for a transaction receipt; raises TimeoutError if too slow."""
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                receipt = await self.w3.eth.get_transaction_receipt(tx_hash)  # type: ignore[arg-type]
                if receipt is not None:
                    return dict(receipt)
            except TransactionNotFound:
                # Not yet mined — keep polling until timeout.
                pass
            except Exception as exc:
                log.warning("Receipt poll error for %s: %s", tx_hash, exc)
            await asyncio.sleep(2)
        raise TimeoutError(f"Transaction {tx_hash} not mined within {timeout}s")

    # ── Nonce management ──────────────────────────────────────────────────────

    async def get_nonce(self, address: str) -> int:
        checksum = Web3.to_checksum_address(address)
        return await self.w3.eth.get_transaction_count(checksum, "pending")


class NonceManager:
    """
    Serialises transaction nonce allocation to prevent nonce collisions when
    multiple coroutines submit transactions concurrently (E4-S4).

    Usage
    -----
    nm = NonceManager(client, wallet_address)
    async with nm.reserve() as nonce:
        tx = build_tx(nonce=nonce, ...)
        await client.send_transaction(tx, private_key)

    The manager fetches the on-chain pending nonce on first use and then
    increments it locally for each reservation.  On any send error the
    cached nonce is invalidated so the next call re-fetches from the chain.
    """

    def __init__(self, client: "PolygonClient", address: str) -> None:
        self._client  = client
        self._address = address
        self._nonce:  int = -1
        self._lock    = asyncio.Lock()

    async def _refresh(self) -> None:
        self._nonce = await self._client.get_nonce(self._address)

    async def next_nonce(self) -> int:
        """Return the next nonce and increment the local counter."""
        async with self._lock:
            if self._nonce < 0:
                await self._refresh()
            nonce = self._nonce
            self._nonce += 1
            return nonce

    def invalidate(self) -> None:
        """Reset the cached nonce; the next call will re-fetch from chain."""
        self._nonce = -1
