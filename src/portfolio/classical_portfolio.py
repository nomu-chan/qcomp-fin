from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models


@dataclass
class ClassicalResult:
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    mu: pd.Series
    covariance: pd.DataFrame


def compute_moments(prices: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    mu = expected_returns.mean_historical_return(prices)
    covariance = risk_models.sample_cov(prices)
    return mu, covariance

# Loss function: negative Sharpe ratio, which we want to minimize (equivalent to maximizing Sharpe)
def solve_max_sharpe(prices: pd.DataFrame) -> ClassicalResult:
    mu, covariance = compute_moments(prices)

    ef = EfficientFrontier(mu, covariance)
    ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    expected_return, volatility, sharpe_ratio = ef.portfolio_performance(verbose=False)

    return ClassicalResult(
        weights=cleaned_weights,
        expected_return=expected_return,
        volatility=volatility,
        sharpe_ratio=sharpe_ratio,
        mu=mu,
        covariance=covariance,
    )
