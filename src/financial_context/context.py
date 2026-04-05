from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from pathlib import Path
from functools import lru_cache
from pypfopt import EfficientFrontier, expected_returns, risk_models
import numpy as np
import pandas as pd
from pathlib import Path


class FinancialContext:
    def __init__(self, prices: pd.DataFrame, esg: pd.DataFrame | None = None, scalar_gain: float = 10.0):
        self.tickers = prices.columns.tolist()
        self._price_df = prices
        self._esg_df = esg
        self.mu_scalar_gain = scalar_gain
        self.standard_scaler = StandardScaler()
        self.min_max_scalar = MinMaxScaler()

    def get_moments(self, 
        standardize: bool = False, 
        normalize: bool = False,
        apply_gain: bool = False
        ):
        # 1. Raw Stats
        mu = expected_returns.mean_historical_return(self._price_df)
        cov_arr = risk_models.CovarianceShrinkage(self._price_df).ledoit_wolf()  # ledoit_wolf() returns a numpy.ndarray, not a DataFrame

        
        # mu is a Series, so it needs to_numpy()
        mu_arr = mu.to_numpy(dtype=np.float64).reshape(-1, 1)
        cov_arr = cov_arr.astype(np.float64)

        # 2. Standardize
        if standardize:
            mu_arr = self.standard_scaler.fit_transform(mu_arr)
            cov_arr = self.standard_scaler.fit_transform(cov_arr)

        # 3. Normalize
        if normalize:
            mu_arr = self.min_max_scalar.fit_transform(mu_arr)
            cov_arr = self.min_max_scalar.fit_transform(cov_arr)
            
        mu_arr = pd.Series(mu_arr.flatten()) * self.mu_scalar_gain if apply_gain else pd.Series(mu_arr.flatten()) 
        cov_arr = pd.DataFrame(cov_arr)
        
        return mu_arr.to_numpy(), cov_arr.to_numpy()

    def esg_scores(self) -> pd.Series:
        if self._esg_df is not None:
            scores = self._esg_df.set_index('ticker')['score'].reindex(self.tickers).fillna(50)
            
            # FIX: Convert to numpy explicitly before scaling to avoid Categorical/Unknown errors
            score_arr = scores.to_numpy(dtype=np.float64).reshape(-1, 1)
            scaled_esg = self.min_max_scalar.fit_transform(score_arr).flatten()
            
            return pd.Series(scaled_esg * self.mu_scalar_gain, index=self.tickers)
        
        return pd.Series(0.5 * self.mu_scalar_gain, index=self.tickers)