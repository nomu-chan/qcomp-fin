from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ProjectConfig:
    tickers: List[str]
    period: str = "5y"
    interval: str = "1d"
    risk_aversion: float = 2.0
    target_cardinality: int = 5
    penalty_strength: float = 10.0


def default_tickers() -> List[str]:
    return [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "AMZN",
        "META",
        "ADBE",
        "CRM",
        "ORCL",
        "INTC",
    ]


def default_config() -> ProjectConfig:
    return ProjectConfig(tickers=default_tickers())
