from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, OrderedDict, Any
import pandas as pd
import numpy as np
from src.financial_context.command import FinancialContext, FinancialContextCommand
from src.symbolics.hamiltonian_modelling import QuantumProblemModelingBuilder
from src.quantum.middleware.instantiator import InstantiatorCommand
from src.quantum.middleware.minimizer import MinimizerCommand
from scipy.stats import entropy
from src.utils.logging_mod import get_logging
from enum import Enum, auto

logger = get_logging(__name__)

@dataclass
class PortfolioResult:
    method_name: str        
    weights: OrderedDict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    esg_score: float = 0.0  
    energy: float | None = None
    success_prob: float | None = None
    config: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    
    def __str__(self) -> str:
        # Assuming weights keys are already strings like 'AAPL', 'MSFT', etc.
        # If they are currently 'asset_0', 'asset_1', ensure your controller 
        # maps them before instantiating this class.
        weight_lines = []
        for ticker, weight in self.weights.items():
            if weight > 0:
                # Highlight significant allocations
                prefix = "⭐ " if weight > 0.2 else "  " 
                weight_lines.append(f"{prefix}{ticker}: {weight:.2%}")

        weight_str = "\n".join(weight_lines)
        
        config_str = "\n".join([f"  {k}: {v}" for k, v in self.config.items()]) if self.config else "  (None)"

        return "\n".join([
            f"\n┏━━━━━━━━━ Portfolio: {self.method_name} ━━━━━━━━━┓"
        ,   f"  ALLOCATION:"
        ,   weight_str if weight_str else "  (No assets selected)"
        ,   f"  ──────────────────────────────────────────"
        ,   f"  Metrics:"
        ,   f"    Return:      {self.expected_return:.2%}"
        ,   f"    Risk (Vol):  {self.volatility:.2%}"
        ,   f"    Sharpe:      {self.sharpe_ratio:.4f}"
        ,   f"    ESG Score:   {self.esg_score:.2f}"
        ,   f"  Optimization Data:"
        ,   f"    Energy:      {self.energy if self.energy is not None else 'N/A'}"
        ,   f"    Confidence:  {f'{self.success_prob:.2%}' if self.success_prob else 'N/A'}"
        ,   f"  Config:"
        ,   config_str
        ,   f"┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
        ])


class QMethod(Enum):
    GATE_BASED = auto()
    ANALOG_BASED = auto()

class PortfolioBase(ABC):
    def __init__(self, name: str, financial_context: FinancialContextCommand) -> None:
        self.name = name
        self.financial_context = financial_context

    @abstractmethod
    def run(self) -> PortfolioResult:
        raise NotImplementedError("Error, not implemented correctly")

    def _handle_empty_selection(self, mu: pd.Series, tickers: list) -> PortfolioResult:
        """
        Recovery logic for when the Quantum model fails to select any assets.
        Selects the Top-3 assets by historical return as a fallback.
        """
        logger.error(f"[{self.name}] QAOA selected 0 assets. Executing Recovery.")
        top_k = 3
        top_indices = pd.Series(mu, index=tickers).nlargest(top_k).index
        
        # Create equal weights for the top-k assets
        weights = pd.Series(0.0, index=tickers)
        weights[top_indices] = 1.0 / top_k
        
        # Return a 'Recovery' result
        return PortfolioResult(
            method_name=f"{self.name} (Recovery)",
            weights=OrderedDict({str(k): float(v) for k, v in weights.to_dict().items()}),
            expected_return=float(np.dot(weights, mu)),
            volatility=0.0, # Placeholder or calculate
            sharpe_ratio=0.0,
            config=OrderedDict({"status": "recovery_mode", "reason": "zero_selection"})
        )
    
    def _finalize_portfolio(
        self, 
        selections: list[int], 
        context: FinancialContext,
        mode: QMethod,
        hybrid_refinement = False,
        additional: Dict[str, Any] = {}
    ) -> PortfolioResult:
        """
        Transforms quantum selections into a statistically sound portfolio.
        Calculates HHI for diversification and Sharpe for performance.
        """
        print(f"DEBUG: Raw Selections Received: {selections}")
        mu_raw, cov_raw = context.get_moments(False, False)
        tickers = context.tickers
        
        # Convert NumPy arrays to Pandas Series/DataFrame using the tickers as indices
        mu = pd.Series(mu_raw, index=tickers)
        cov = pd.DataFrame(cov_raw, index=tickers, columns=tickers)
        
        weights = pd.Series(0.0, index=tickers)
        
        # 1. Identify Assets Selected by Quantum
        selected_indices = [i for i, val in enumerate(selections) if val > 0]
        selected_tickers = [tickers[i] for i in selected_indices]
        
        if not selected_tickers or sum(selections) == 0:
            # Pylance fix: mu is now already a Series
            return self._handle_empty_selection(mu, tickers)
        # 2. Hybrid Refinement (MPT)
        try:
            from pypfopt.efficient_frontier import EfficientFrontier
            # Now .loc and list-indexing will work perfectly
            if hybrid_refinement:
                try:
                    from pypfopt.efficient_frontier import EfficientFrontier
                    ef = EfficientFrontier(mu[selected_tickers], cov.loc[selected_tickers, selected_tickers])
                    ef.max_sharpe()
                    refined_weights = ef.clean_weights()
                    strategy_used = "Hybrid (QAOA + MPT)"
                except Exception as e:
                    logger.warning(f"MPT Refinement failed, falling back: {e}")
                    hybrid_refinement = False # Trigger fallback below
            else:
                # NORMALIZATION: Convert [0, 0, 0, 0, 0, 8, 0, 8, 0, 8] into weights
                total_units = sum(selections)
                # Create a dictionary of {Ticker: Weight}
                refined_weights = {
                    tickers[i]: (val / total_units) 
                    for i, val in enumerate(selections) if val > 0
                }
                strategy_used = "Pure-Quantum (Integer Proportions)"
            weights.update(pd.Series(refined_weights))
            strategy_used = "Hybrid (QAOA + MPT)"
        except Exception as e:
            logger.warning(f"MPT Refinement failed, using Quantum Proportions: {e}")
            total = sum(selections)
            for i, val in enumerate(selections):
                weights[tickers[i]] = val / total
            strategy_used = f"Quantum-Only (Integer Proportions) + {mode.__str__()}"

        # 3. Calculate Advanced Statistics
        w = weights.to_numpy()
        port_return = np.dot(w, mu)
        port_vol = np.sqrt(w.T @ cov @ w)
        sharpe = (port_return - 0.02) / (port_vol + 1e-9)
        
        # HHI Index: sum(w^2). Lower is more diversified. 
        # 1.0 = 1 stock, 0.1 = well diversified.
        hhi = np.sum(w**2)
        
        # Diversification Ratio (Return per unit of Risk compared to average)
        div_ratio = port_return / (np.dot(w, np.sqrt(np.diag(cov))) + 1e-9)

        return PortfolioResult(
            method_name=self.name,
            weights=OrderedDict({str(ticker): float(w) for ticker, w in weights.items()}),
            expected_return=float(port_return),
            volatility=float(port_vol),
            sharpe_ratio=float(sharpe),
            config=OrderedDict({
                "strategy": strategy_used,
                "assets_selected": len(selected_tickers),
                "hhi_index": round(float(hhi), 4),
                "diversification_ratio": round(float(div_ratio), 4),
                "quantum_bits": "".join(map(str, selections))
            }|additional)
        )