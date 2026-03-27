from __future__ import annotations

from pathlib import Path

import numpy as np
from src.etl import data_pipeline
from classical_baseline import solve_max_sharpe
from config import default_config
from data_pipeline import download_close_prices, save_prices
from src.utils.qubo_builder import build_binary_selection_qubo, qubo_energy


def run() -> None:
    config = default_config()

    prices = download_close_prices(
        tickers=config.tickers,
        period=config.period,
        interval=config.interval,
    )
    save_prices(prices, Path("data/close_prices.csv"))

    classical = solve_max_sharpe(prices)

    print("Classical max-sharpe portfolio")
    print("Expected return:", round(classical.expected_return, 4))
    print("Volatility:", round(classical.volatility, 4))
    print("Sharpe:", round(classical.sharpe_ratio, 4))
    print("Weights:")
    for ticker, weight in classical.weights.items():
        if weight > 0:
            print(f"  {ticker}: {weight:.3f}")

    qubo = build_binary_selection_qubo(
        mu=classical.mu,
        covariance=classical.covariance,
        risk_aversion=config.risk_aversion,
        target_cardinality=config.target_cardinality,
        penalty_strength=config.penalty_strength,
    )

    x0 = np.zeros(len(qubo.tickers), dtype=int)
    x0[: config.target_cardinality] = 1
    baseline_energy = qubo_energy(x0, qubo)

    print("\nQUBO model ready")
    print("Assets:", len(qubo.tickers))
    print("Q matrix shape:", qubo.q_matrix.shape)
    print("Sample feasible energy:", round(baseline_energy, 6))



