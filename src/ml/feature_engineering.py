"""
Feature engineering for the ML arbitrage scoring model.

Converts raw arbitrage opportunity data into a fixed-length numeric
feature vector suitable for sklearn estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from src.payload.protocol import OpportunityPayload
from src.utils.logger import get_logger

log = get_logger(__name__)

FEATURE_NAMES: List[str] = [
    "price_spread_pct",          # Raw price spread between buy/sell DEX
    "estimated_profit_pct",      # Profit % after fees
    "loan_amount_usd_log",       # log10(loan amount in USD)
    "gas_cost_usd",              # Estimated gas cost in USD
    "liquidity_depth_log",       # log10(min liquidity across hops)
    "historical_success_rate",   # Rolling success rate for similar paths
    "slippage_estimate_pct",     # Estimated slippage
    "num_hops",                  # Number of swap hops
    "is_triangular",             # 1 if triangular arb, 0 if cross-platform
    "fee_bps_sum",               # Sum of all hop fees in bps
    "spread_to_fee_ratio",       # spread / total_fees
    "liquidity_balance",         # min/max liquidity ratio across hops
    "block_utilization_pct",     # Current block utilization %
    "time_of_day_sin",           # Sine of hour/24 (cyclic encoding)
    "time_of_day_cos",           # Cosine of hour/24 (cyclic encoding)
]

N_FEATURES = len(FEATURE_NAMES)


@dataclass
class FeatureVector:
    """Named container for a feature array."""
    features: np.ndarray   # shape (N_FEATURES,)
    opportunity_id: str
    feature_names: List[str]

    def to_2d(self) -> np.ndarray:
        """Return shape (1, N_FEATURES) for sklearn predict."""
        return self.features.reshape(1, -1)


class FeatureEngineer:
    """
    Extracts ML features from OpportunityPayload objects.

    Usage
    -----
    fe = FeatureEngineer()
    fv = fe.extract(opportunity, historical_success_rate=0.72,
                    block_utilization_pct=45.0, gas_cost_usd=1.20)
    X = fv.to_2d()
    """

    def __init__(self) -> None:
        self._history: Dict[str, List[bool]] = {}   # arb_type → outcomes

    def record_outcome(self, arb_type: str, success: bool) -> None:
        """Update rolling history for success-rate feature."""
        if arb_type not in self._history:
            self._history[arb_type] = []
        self._history[arb_type].append(success)
        # Keep last 100 outcomes
        self._history[arb_type] = self._history[arb_type][-100:]

    def historical_success_rate(self, arb_type: str) -> float:
        hist = self._history.get(arb_type, [])
        if not hist:
            return 0.5   # prior: 50%
        return sum(hist) / len(hist)

    def extract(
        self,
        opp: OpportunityPayload,
        gas_cost_usd: float = 0.0,
        block_utilization_pct: float = 50.0,
        timestamp: Optional[float] = None,
    ) -> FeatureVector:
        """
        Build a FeatureVector from an OpportunityPayload.

        Parameters
        ----------
        opp:                    The arbitrage opportunity.
        gas_cost_usd:           Estimated gas cost in USD.
        block_utilization_pct:  Current block utilization (0-100).
        timestamp:              Unix timestamp (defaults to opp.timestamp).
        """
        import math
        import time as _time

        ts = timestamp or opp.timestamp
        hour = (_time.gmtime(ts).tm_hour + _time.gmtime(ts).tm_min / 60.0)
        tod_sin = math.sin(2 * math.pi * hour / 24.0)
        tod_cos = math.cos(2 * math.pi * hour / 24.0)

        # Liquidity stats across hops
        liquidities = [h.estimated_amount_out for h in opp.path if h.estimated_amount_out > 0]
        min_liq = min(liquidities) if liquidities else 1.0
        max_liq = max(liquidities) if liquidities else 1.0
        liq_balance = min_liq / max_liq if max_liq > 0 else 0.0
        liq_log = math.log10(max(min_liq, 1.0))

        # Fee sum
        fee_sum = sum(h.fee_bps for h in opp.path)

        # Price spread (from raw spread stored in opportunity)
        price_spread_pct = opp.expected_profit_usd / max(opp.loan_amount_usd, 1.0) * 100.0
        profit_pct_after_fees = max(price_spread_pct - (fee_sum / 100.0) - 0.05, 0.0)

        # spread-to-fee ratio
        total_fee_pct = fee_sum / 100.0 + 0.05
        spread_fee_ratio = price_spread_pct / total_fee_pct if total_fee_pct > 0 else 0.0

        hist_sr = self.historical_success_rate(opp.arb_type)

        vec = np.array([
            price_spread_pct,
            profit_pct_after_fees,
            math.log10(max(opp.loan_amount_usd, 1.0)),
            gas_cost_usd,
            liq_log,
            hist_sr,
            opp.slippage_estimate_pct,
            float(len(opp.path)),
            1.0 if opp.arb_type == "triangular" else 0.0,
            float(fee_sum),
            spread_fee_ratio,
            liq_balance,
            block_utilization_pct,
            tod_sin,
            tod_cos,
        ], dtype=np.float64)

        assert len(vec) == N_FEATURES, f"Feature count mismatch: {len(vec)} vs {N_FEATURES}"

        return FeatureVector(
            features=vec,
            opportunity_id=opp.message_id,
            feature_names=FEATURE_NAMES,
        )

    @staticmethod
    def feature_names() -> List[str]:
        return list(FEATURE_NAMES)
