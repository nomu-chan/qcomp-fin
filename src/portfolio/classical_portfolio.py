from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, OrderedDict, Any, cast
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from src.utils.logging_mod import logging
from src.utils.portfolio import Portfolio, PortfolioResult

logger = logging.getLogger(__name__)

class ClassicalPortfolio(Portfolio):
    def __init__(self, name: str):
        super().__init__(name)

    def _compute_moments(self, prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        mu = expected_returns.mean_historical_return(prices)
        covariance = pd.DataFrame(risk_models.sample_cov(prices))
        return (mu, covariance)

    def run(self, prices: pd.DataFrame) -> PortfolioResult:
        """
        Loss function: negative Sharpe ratio, which we want to minimize (equivalent to maximizing Sharpe)

        Args:
            prices (pd.DataFrame): prices

        Returns:
            PortfolioResult: the output of the solver
        """
        mu, covariance = self._compute_moments(prices)

        ef = EfficientFrontier(mu, covariance)
        ef.max_sharpe()
        raw_weights = ef.clean_weights()
        cleaned_weights = OrderedDict({str(k): float(v) for k, v in raw_weights.items()})
        
        performance = ef.portfolio_performance(verbose=False)
        expected_return = float(cast(Any, performance[0]))
        volatility = float(cast(Any, performance[1]))
        sharpe_ratio = float(cast(Any, performance[2]))
        config = OrderedDict({
            "mu":mu,
            "covariance": covariance,
        })
        return PortfolioResult(
            method_name=self.name,
            weights=cleaned_weights,
            expected_return=expected_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            config=config
        )
