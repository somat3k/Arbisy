"""
Flash loan trigger and management.

Encodes swap paths, submits flash loan transactions to the deployed
FlashLoanArbitrage contract, and monitors execution results.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from eth_abi import encode
from web3 import Web3

from src.blockchain.polygon_client import PolygonClient
from src.payload.protocol import OpportunityPayload, SwapHop
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)

# ── FlashLoanArbitrage contract ABI (minimal) ─────────────────────────────────
FLASH_LOAN_ABI = [
    {
        "inputs": [
            {"name": "assets",            "type": "address[]"},
            {"name": "amounts",           "type": "uint256[]"},
            {"name": "interestRateModes", "type": "uint256[]"},
            {"name": "params",            "type": "bytes"},
        ],
        "name": "initiateFlashLoan",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "asset",  "type": "address"},
            {"name": "amount", "type": "uint256"},
            {"name": "params", "type": "bytes"},
        ],
        "name": "initiateFlashLoanSimple",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"name": "loanAmount",   "type": "uint256"},
            {"name": "outputAmount", "type": "uint256"},
        ],
        "name": "simulateProfit",
        "outputs": [
            {"name": "profit",     "type": "uint256"},
            {"name": "profitable", "type": "bool"},
        ],
        "stateMutability": "pure",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True,  "name": "asset",     "type": "address"},
            {"indexed": False, "name": "loanAmount", "type": "uint256"},
            {"indexed": False, "name": "repayAmount","type": "uint256"},
            {"indexed": False, "name": "netProfit",  "type": "uint256"},
            {"indexed": True,  "name": "initiator",  "type": "address"},
        ],
        "name": "ArbitrageExecuted",
        "type": "event",
    },
]


@dataclass
class SimulationResult:
    profitable: bool
    loan_amount_wei: int
    expected_output_wei: int
    profit_wei: int
    profit_usd: float
    repay_amount_wei: int


class FlashLoan:
    """
    High-level interface to the FlashLoanArbitrage Solidity contract.

    Parameters
    ----------
    client:            PolygonClient instance.
    contract_address:  Address of deployed FlashLoanArbitrage contract.
    """

    FLASH_PREMIUM_BPS = 5  # 0.05%

    def __init__(
        self,
        client: PolygonClient,
        contract_address: Optional[str] = None,
    ) -> None:
        self._client = client
        cfg = get_config()
        addr = contract_address or cfg.flash_loan_contract
        self._contract_address = Web3.to_checksum_address(addr) if addr else None
        self._private_key = cfg.private_key
        self._wallet = cfg.wallet_address
        if self._contract_address:
            self._contract = self._client.w3s.eth.contract(
                address=self._contract_address,
                abi=FLASH_LOAN_ABI,
            )
        else:
            self._contract = None
        log.info("FlashLoan handler ready — contract: %s", self._contract_address)

    # ── Encoding ──────────────────────────────────────────────────────────────

    @staticmethod
    def encode_swap_path(
        hops: List[SwapHop],
        expected_amounts_out: Optional[List[int]] = None,
    ) -> bytes:
        """
        ABI-encode a list of SwapHop objects into the bytes `params` expected
        by FlashLoanArbitrage.executeOperation.

        Encoding (E7-S2, E7-S5):
            (bytes32 nonce, address[] routers, address[] tokens,
             uint24[] fees, bool[] isV3, uint256[] expectedAmountsOut)

        A fresh cryptographic nonce is generated for each call to ensure
        replay protection in the contract.
        tokens has len(hops)+1 entries (token_in of hop0 .. token_out of last hop).
        expected_amounts_out: per-hop expected output amounts from off-chain
            simulation (in tokenOut smallest units).  Pass None or an empty list
            to skip per-hop slippage checks (profit guard is the backstop).
        """
        nonce = secrets.token_bytes(32)

        routers = [Web3.to_checksum_address(h.router_address) for h in hops]
        tokens: List[str] = []
        fees: List[int] = []
        is_v3: List[bool] = []

        for i, hop in enumerate(hops):
            if i == 0:
                tokens.append(Web3.to_checksum_address(hop.token_in))
            tokens.append(Web3.to_checksum_address(hop.token_out))
            fees.append(hop.fee_bps * 100)  # bps → Uniswap V3 fee units (hundredths of a bps; e.g. 30 bps → 3000)
            is_v3.append(hop.is_v3)

        # Pad or truncate expected_amounts_out to match hop count
        n_hops = len(hops)
        amounts_out: List[int] = list(expected_amounts_out or [])
        if len(amounts_out) < n_hops:
            amounts_out.extend([0] * (n_hops - len(amounts_out)))
        else:
            amounts_out = amounts_out[:n_hops]

        encoded = encode(
            ["bytes32", "address[]", "address[]", "uint24[]", "bool[]", "uint256[]"],
            [nonce, routers, tokens, fees, is_v3, amounts_out],
        )
        return encoded

    # ── Simulation ────────────────────────────────────────────────────────────

    def simulate_profit_onchain(
        self,
        loan_amount_wei: int,
        estimated_output_wei: int,
    ) -> SimulationResult:
        """
        Call the pure `simulateProfit` view function on the contract.
        No gas consumed.
        """
        if self._contract is None:
            raise RuntimeError("FlashLoan contract address not configured.")

        profit_wei, profitable = self._contract.functions.simulateProfit(
            loan_amount_wei, estimated_output_wei
        ).call()

        premium = (loan_amount_wei * self.FLASH_PREMIUM_BPS) // 10_000
        repay   = loan_amount_wei + premium

        return SimulationResult(
            profitable=profitable,
            loan_amount_wei=loan_amount_wei,
            expected_output_wei=estimated_output_wei,
            profit_wei=profit_wei,
            profit_usd=0.0,  # Caller should convert using current USD price
            repay_amount_wei=repay,
        )

    # ── Execution ─────────────────────────────────────────────────────────────

    async def execute_flash_loan(
        self,
        opportunity: OpportunityPayload,
    ) -> Dict[str, Any]:
        """
        Build and send the flash loan transaction for a given opportunity.

        Returns a dict with tx_hash, gas_used, success, error_message.
        """
        if self._contract is None:
            return {"success": False, "error_message": "Contract not configured"}
        if not self._private_key:
            return {"success": False, "error_message": "Private key not set"}

        try:
            params = self.encode_swap_path(opportunity.path)
            nonce  = await self._client.get_nonce(self._wallet)
            fee_params = await self._client.get_max_fee_params()

            # Always use initiateFlashLoanSimple for single-asset borrows.
            # The "simple" variant is cheaper and is the correct choice whenever
            # we borrow exactly one asset (regardless of the number of swap hops).
            # initiateFlashLoan (multi-asset) is reserved for borrowing multiple
            # *different* tokens simultaneously — not needed for arbitrage.
            tx = self._contract.functions.initiateFlashLoanSimple(
                Web3.to_checksum_address(opportunity.asset),
                opportunity.loan_amount_wei,
                params,
            ).build_transaction({
                "from":  Web3.to_checksum_address(self._wallet),
                "nonce": nonce,
                "gas":   800_000,
                **fee_params,
            })

            tx_hash = await self._client.send_transaction(tx, self._private_key)
            log.info("Flash loan tx submitted: %s", tx_hash)

            receipt = await self._client.wait_for_receipt(tx_hash, timeout=60)
            success = receipt.get("status") == 1

            log.info(
                "Flash loan %s — block=%s gas=%s",
                "SUCCESS" if success else "FAILED",
                receipt.get("blockNumber"),
                receipt.get("gasUsed"),
            )
            return {
                "success":      success,
                "tx_hash":      tx_hash,
                "gas_used":     receipt.get("gasUsed", 0),
                "block_number": receipt.get("blockNumber", 0),
                "error_message": "" if success else "Transaction reverted",
            }

        except Exception as exc:
            log.error("Flash loan execution error: %s", exc, exc_info=True)
            return {"success": False, "error_message": str(exc), "tx_hash": ""}

    # ── Premium helpers ───────────────────────────────────────────────────────

    @staticmethod
    def compute_repay_amount(loan_amount_wei: int) -> int:
        premium = (loan_amount_wei * FlashLoan.FLASH_PREMIUM_BPS) // 10_000
        return loan_amount_wei + premium

    @staticmethod
    def compute_min_profitable_output(loan_amount_wei: int) -> int:
        """Minimum output amount that yields positive net profit."""
        return FlashLoan.compute_repay_amount(loan_amount_wei) + 1
