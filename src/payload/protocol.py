"""
Message protocol definitions for the Arbisy payload bus.

All inter-component messages are typed dataclasses that serialize to / from
plain dicts for easy logging and transmission.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class MessageType(str, Enum):
    OPPORTUNITY    = "OPPORTUNITY"
    EXECUTION_RESULT = "EXECUTION_RESULT"
    MARKET_ANALYSIS  = "MARKET_ANALYSIS"
    PRICE_UPDATE     = "PRICE_UPDATE"
    SYSTEM_EVENT     = "SYSTEM_EVENT"
    ERROR            = "ERROR"


class Priority(int, Enum):
    CRITICAL = 0   # Execute immediately
    HIGH     = 1
    MEDIUM   = 5
    LOW      = 10


@dataclass
class BaseMessage:
    """Root message type — every payload extends this."""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.SYSTEM_EVENT
    priority: Priority = Priority.MEDIUM
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["message_type"] = self.message_type.value
        d["priority"] = int(self.priority)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BaseMessage":
        d = dict(d)
        d["message_type"] = MessageType(d["message_type"])
        d["priority"] = Priority(d["priority"])
        return cls(**d)


@dataclass
class SwapHop:
    """Represents a single DEX swap within an arbitrage path."""
    dex_name: str
    router_address: str
    token_in: str
    token_out: str
    fee_bps: int
    is_v3: bool
    estimated_amount_out: float = 0.0


@dataclass
class OpportunityPayload(BaseMessage):
    """Emitted by ArbitrageAgent when a profitable path is detected."""
    message_type: MessageType = MessageType.OPPORTUNITY
    priority: Priority = Priority.HIGH

    asset: str = ""                     # Flash-loan asset address
    loan_amount_wei: int = 0            # Loan amount in wei
    loan_amount_usd: float = 0.0
    expected_profit_usd: float = 0.0
    expected_profit_wei: int = 0
    ml_score: float = 0.0
    path: List[SwapHop] = field(default_factory=list)
    gas_estimate_gwei: float = 0.0
    slippage_estimate_pct: float = 0.0
    dex_sources: List[str] = field(default_factory=list)
    arb_type: str = "cross_platform"    # "triangular" | "cross_platform"


@dataclass
class ExecutionResultPayload(BaseMessage):
    """Emitted by ExecutionAgent after a flash-loan transaction."""
    message_type: MessageType = MessageType.EXECUTION_RESULT
    priority: Priority = Priority.HIGH

    opportunity_id: str = ""
    success: bool = False
    tx_hash: str = ""
    actual_profit_usd: float = 0.0
    actual_profit_wei: int = 0
    gas_used: int = 0
    gas_price_gwei: float = 0.0
    error_message: str = ""
    block_number: int = 0


@dataclass
class PriceUpdatePayload(BaseMessage):
    """Emitted by DEXPriceFeed for each pool price refresh."""
    message_type: MessageType = MessageType.PRICE_UPDATE
    priority: Priority = Priority.LOW

    dex_name: str = ""
    pool_address: str = ""
    token0: str = ""
    token1: str = ""
    price_token1_per_token0: float = 0.0
    sqrt_price_x96: int = 0
    liquidity: int = 0
    tick: int = 0
    block_number: int = 0


@dataclass
class MarketAnalysisPayload(BaseMessage):
    """Emitted by AnalysisAgent with broader market context."""
    message_type: MessageType = MessageType.MARKET_ANALYSIS
    priority: Priority = Priority.LOW

    gas_price_gwei: float = 0.0
    block_utilization_pct: float = 0.0
    avg_spread_pct: float = 0.0
    opportunities_last_hour: int = 0
    success_rate_last_hour: float = 0.0
    recommended_min_profit_usd: float = 5.0
    llm_summary: str = ""


@dataclass
class SystemEventPayload(BaseMessage):
    """Generic system lifecycle events."""
    message_type: MessageType = MessageType.SYSTEM_EVENT
    priority: Priority = Priority.MEDIUM

    event_name: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
