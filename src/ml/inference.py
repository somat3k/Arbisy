"""
Live inference wrapper.

Provides a high-level interface to score an OpportunityPayload using
the trained ArbitrageModel.  Handles feature extraction internally.
"""

from __future__ import annotations

from typing import Optional, Tuple

from src.ml.feature_engineering import FeatureEngineer, FeatureVector
from src.ml.model import ArbitrageModel
from src.payload.protocol import OpportunityPayload
from src.utils.config import get_config
from src.utils.logger import get_logger

log = get_logger(__name__)


class LiveInference:
    """
    Combines FeatureEngineer + ArbitrageModel for single-call scoring.

    Usage
    -----
    inferencer = LiveInference()
    score, profit_usd = inferencer.score(opportunity,
                                          gas_cost_usd=1.50,
                                          block_utilization_pct=62.0)
    if score >= 0.60:
        execute(opportunity)
    """

    def __init__(
        self,
        model: Optional[ArbitrageModel] = None,
        feature_engineer: Optional[FeatureEngineer] = None,
    ) -> None:
        self.model = model or ArbitrageModel()
        self.fe    = feature_engineer or FeatureEngineer()
        cfg = get_config()
        self._threshold = cfg.ml_score_threshold
        log.info(
            "LiveInference ready — model trained=%s threshold=%.2f",
            self.model.trained,
            self._threshold,
        )

    def score(
        self,
        opportunity: OpportunityPayload,
        gas_cost_usd: float = 0.0,
        block_utilization_pct: float = 50.0,
    ) -> Tuple[float, float]:
        """
        Score an opportunity.

        Returns
        -------
        (ml_score 0-1, predicted_net_profit_usd)
        """
        fv: FeatureVector = self.fe.extract(
            opportunity,
            gas_cost_usd=gas_cost_usd,
            block_utilization_pct=block_utilization_pct,
        )
        predicted_profit = float(self.model.predict(fv.features)[0])
        ml_score         = self.model.predict_score(fv.features)
        log.debug(
            "Score: %.4f | predicted profit: $%.4f | opp=%s",
            ml_score,
            predicted_profit,
            opportunity.message_id[:8],
        )
        return ml_score, predicted_profit

    def should_execute(
        self,
        opportunity: OpportunityPayload,
        gas_cost_usd: float = 0.0,
        block_utilization_pct: float = 50.0,
    ) -> bool:
        """Return True if opportunity passes the ML score threshold."""
        score, _ = self.score(opportunity, gas_cost_usd, block_utilization_pct)
        return score >= self._threshold

    def record_outcome(self, arb_type: str, success: bool) -> None:
        """Feed execution result back to the feature engineer for the success-rate feature."""
        self.fe.record_outcome(arb_type, success)
