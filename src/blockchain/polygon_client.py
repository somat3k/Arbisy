"""
Polygon zkEVM Web3 client.

Manages HTTP and optional WebSocket connections to the Polygon zkEVM RPC.
Provides helper methods for gas estimation, block polling, and contract calls.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from web3 import AsyncWeb3, Web3
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
        Falls back to legacy gasPrice if baseFee is unavailable.
        """
        try:
            latest = await self.w3.eth.get_block("latest")
            base_fee: int = latest.get("baseFeePerGas", 0)  # type: ignore[assignment]
            max_priority = Web3.to_wei(30, "gwei")
            max_fee = base_fee * 2 + max_priority
            return {
                "maxFeePerGas":         Wei(max_fee),
                "maxPriorityFeePerGas": Wei(max_priority),
            }
        except Exception:
            gas_price = await self.w3.eth.gas_price
            return {"gasPrice": gas_price}

    # ── ERC-20 helpers ────────────────────────────────────────────────────────

    def get_erc20_contract(self, address: str):
        checksum = Web3.to_checksum_address(address)
        return self.w3s.eth.contract(address=checksum, abi=ERC20_ABI)

    async def get_token_balance(self, token_address: str, wallet: str) -> int:
        """Return raw token balance (in wei-equivalent smallest unit)."""
        contract = self.get_erc20_contract(token_address)
        checksum_wallet = Web3.to_checksum_address(wallet)
        return contract.functions.balanceOf(checksum_wallet).call()

    async def get_token_decimals(self, token_address: str) -> int:
        contract = self.get_erc20_contract(token_address)
        return contract.functions.decimals().call()

    async def get_token_symbol(self, token_address: str) -> str:
        contract = self.get_erc20_contract(token_address)
        return contract.functions.symbol().call()

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
            receipt = await self.w3.eth.get_transaction_receipt(tx_hash)  # type: ignore[arg-type]
            if receipt is not None:
                return dict(receipt)
            await asyncio.sleep(2)
        raise TimeoutError(f"Transaction {tx_hash} not mined within {timeout}s")

    # ── Nonce management ──────────────────────────────────────────────────────

    async def get_nonce(self, address: str) -> int:
        checksum = Web3.to_checksum_address(address)
        return await self.w3.eth.get_transaction_count(checksum, "pending")
