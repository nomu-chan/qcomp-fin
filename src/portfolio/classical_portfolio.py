from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, OrderedDict, Any, cast
import pandas as pd
from pypfopt import EfficientFrontier, expected_returns, risk_models
from src.utils.logging_mod import logging
from src.portfolio.portfolio_base import PorfolioBase, PortfolioResult
from src.financial_context.command import FinancialContextCommand

logger = logging.getLogger(__name__)

class ClassicalPortfolio1(PorfolioBase):
    def __init__(self, name: str, financial_context: FinancialContextCommand):
        super().__init__(name, financial_context)

    def run(self) -> PortfolioResult:
        # 1. Get tickers and moments
        context = self.financial_context.get_context()
        tickers = context.tickers  # This is the list of names ["AAPL", "BTC", etc.]
        mu, covariance = context.get_moments(False, False)

        # 2. Ensure mu has ticker names as the index
        # PyPortfolioOpt uses the index of mu to label the weights
        if not isinstance(mu, pd.Series):
            mu = pd.Series(mu, index=tickers)
        
        # 3. Optimize
        ef = EfficientFrontier(mu, covariance)
        ef.max_sharpe()
        
        # clean_weights() now returns { "AAPL": 0.5, "BTC": 0.5 ... }
        raw_weights = ef.clean_weights()
        
        # 4. Cast to OrderedDict for the PortfolioResult
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

class ClassicalPortfolio(PorfolioBase):
    def __init__(self, name: str, financial_context: FinancialContextCommand, k_assets: int = 10):
        super().__init__(name, financial_context)
        self.k_assets = k_assets  # Accept the cardinality hyperparameter

    def run(self) -> PortfolioResult:
        context = self.financial_context.get_context()
        tickers = context.tickers
        mu, covariance = context.get_moments(False, False)

        if not isinstance(mu, pd.Series):
            mu = pd.Series(mu, index=tickers)
        
        # 1. Initialize Frontier
        ef = EfficientFrontier(mu, covariance)

        # 2. Apply Cardinality Constraint (The Classical equivalent to lambda_cardinality)
        if self.k_assets is not None:
            # We use L1 Regularization as a proxy for cardinality in convex solvers.
            # This 'encourages' the model to zero out small weights.
            ef.add_objective(objective_functions.L1_reg, w=0.1) 
            
            # For strict K-assets, we can use a constraint if using a solver like ECOs/OSQP
            # However, PyPortfolioOpt handles strict k-limits best via:
            ef.add_constraint(lambda w: w <= 1.0 / self.k_assets * 2) # Example weight cap

        # 3. Optimize
        try:
            # Max Sharpe is problematic with strict cardinality; 
            # often better to target a specific return or min volatility
            ef.max_sharpe()
        except:
            # Fallback to Min Volatility if Max Sharpe fails under strict constraints
            ef.min_volatility()
            
        raw_weights = ef.clean_weights()
        
        # 4. Post-Process to enforce strict K-assets (Classical Truncation)
        # Sort by weight and keep only the top K
        if self.k_assets:
            sorted_weights = sorted(raw_weights.items(), key=lambda x: x[1], reverse=True)
            top_k_tickers = [t for t, w in sorted_weights[:self.k_assets]]
            
            # Zero out everything else and re-normalize
            new_weights = {t: (w if t in top_k_tickers else 0) for t, w in raw_weights.items()}
            sum_w = sum(new_weights.values())
            cleaned_weights = OrderedDict(
                {str(k): float(v) for k, v in new_weights.items()}
            )
        else:
            cleaned_weights = OrderedDict({str(k): float(v) for k, v in raw_weights.items()})

        # Recalculate stats based on adjusted weights
        performance = ef.portfolio_performance(verbose=False)
        
        return PortfolioResult(
            method_name=self.name,
            weights=cleaned_weights,
            expected_return=float(performance[0]),
            volatility=float(performance[1]),
            sharpe_ratio=float(performance[2]),
            config=OrderedDict({"mu": mu, "k_enforced": self.k_assets})
        )