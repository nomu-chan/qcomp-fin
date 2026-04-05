import yfinance as yf
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from typing import Any
from src.financial_context.context import FinancialContext

class DataCollectorCommand:
    def __init__(self, tickers: list[str], csv_path: Path):
        self.tickers = tickers
        self.csv_path: Path = csv_path
        self.period = "5y"
        self.interval = "1d"

    def fetch_prices(self, period: str = "5y") -> pd.DataFrame:
        """Fetches OHLCV data and returns a cleaned Close Price DataFrame."""
        data = yf.download(
            self.tickers,
            period=self.period,
            interval=self.interval,
            auto_adjust=True,
            progress=False,
        )

        if data is None or data.empty:
            raise ValueError("No data returned from yfinance.")
        
        prices = data["Close"]
        # Ensure it's a DataFrame even for a single ticker
        if isinstance(prices, pd.Series):
            prices = prices.to_frame()
            
        return prices.dropna(axis="columns", how="any")

    def fetch_esg(self) -> pd.DataFrame:
        """Fetches ESG scores independently."""
        esg_dict = {}
        for t in self.tickers:
            try:
                ticker_obj = yf.Ticker(t)
                # Use .get() or a try-block for the specific attribute
                sustainability = ticker_obj.sustainability
                if sustainability is not None and 'totalEsg' in sustainability.index:
                    esg_dict[t] = sustainability.loc['totalEsg', 'Value']
                else:
                    esg_dict[t] = 50.0
            except Exception:
                esg_dict[t] = 50.0
        
        # FIX: Pass [0] as the index to handle scalar values
        # and then transpose so tickers are the index
        df = pd.DataFrame(esg_dict, index=['esg_score']).T
    
