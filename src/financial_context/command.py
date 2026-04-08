import pandas as pd
from pathlib import Path
from .context import FinancialContext
from .data_collection import DataCollectorCommand
from etc.config import DATAPATH

class FinancialContextCommand:
    def __init__(self, tickers: list[str], csv_path: str | Path, scalar_gain: float = 1.0):
        self.tickers = sorted(tickers) # Sort to ensure consistent CSV columns
        self.csv_path = Path(csv_path)
        self.gain = scalar_gain
        self.collector = DataCollectorCommand(self.tickers, self.csv_path)
        self.data_dir = DATAPATH
        # Define separate storage paths
        self.price_path = self.data_dir / "prices.csv"
        self.esg_path = self.data_dir / "esg.csv"

    def get_context(self, update_prices: bool = False, update_esg: bool = False) -> FinancialContext:
        """
        The Master Logic: 
        Checks for local CSV -> Updates if needed -> Returns a Cached Proxy.
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Handle Prices
        if not self.price_path.exists() or update_prices:
            prices_df = self.collector.fetch_prices()
            prices_df.to_csv(self.price_path)
        else:
            prices_df = pd.read_csv(self.price_path, index_col=0, parse_dates=True)
        prices_df = prices_df[self.tickers]

        # 2. Handle ESG
        if not self.esg_path.exists() or update_esg:
            esg_series = self.collector.fetch_esg()
            
            if esg_series is not None:
                esg_series.to_csv(self.esg_path)
            else:
                # Create a dummy if everything else fails
                pd.DataFrame(50.0, index=self.tickers, columns=['esg_score']).to_csv(self.esg_path)
            esg_series.to_csv(self.esg_path)
        else:
            # Read ESG and convert back to Series
            esg_series = pd.read_csv(self.esg_path, header=None, names=["ticker", "score"])
        
        return FinancialContext(prices_df, esg_series, scalar_gain=self.gain)