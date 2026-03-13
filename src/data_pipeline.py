from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf


def download_close_prices(
    tickers: Iterable[str],
    period: str = "5y",
    interval: str = "1d",
) -> pd.DataFrame:
    ticker_list = list(tickers)
    data = yf.download(
        ticker_list,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )

    if "Close" not in data:
        raise ValueError("No Close prices returned from yfinance.")

    close_prices = data["Close"].dropna(how="all")
    close_prices = close_prices.dropna(axis=1, how="any")

    if close_prices.empty:
        raise ValueError("Close price table is empty after cleaning.")

    return close_prices


def save_prices(prices: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(out_path)


def load_prices(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, index_col=0, parse_dates=True)
