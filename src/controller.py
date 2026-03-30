from __future__ import annotations

import ctypes
import os

# --- THIS MUST BE THE FIRST PIECE OF CODE EXECUTED ---
try:
    # Force symbols into global scope before any Qiskit modules are loaded
    ctypes.CDLL("libnvidia-ml.so.1", mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
except Exception:
    pass

from pathlib import Path
import pandas as pd
import numpy as np
from src.etl import data_pipeline
from src.portfolio.classical_portfolio import ClassicalPortfolio
from src.portfolio.qport_bin import QportBinary
from src.portfolio.qport_int import QportInteger
from src.portfolio.qport_div import QportDiversification
from etc.config import default_config
from src.etl.data_pipeline import download_close_prices, save_prices
from src.quantum_engine.qubo_builder import build_binary_selection_qubo, qubo_energy
from src.utils.logging_mod import get_logging
from src.simulation.benchmark import BenchmarkEngine
import random
from datetime import datetime

logger = get_logging(__name__)
SEED = 67

def run() -> None:
    config = default_config()
    prices = download_close_prices(config.tickers, config.period, config.interval)
    
    # 2. Initialize Engine
    engine = BenchmarkEngine(prices)
    
    q_algo_div_50 = QportDiversification(name="Qv3_diversification_diverse_50", p=4, maxiter=300, lam_card=50)
    stability_results = engine.run_stability_test(q_algo_div_50, num_runs=5)
    stability_results.to_csv("data/q_algo_div_50.csv")
    
    q_algo_div_250 = QportDiversification(name="Qv3_diversification_diverse_250", p=4, maxiter=300, lam_card=250)
    stability_results = engine.run_stability_test(q_algo_div_250, num_runs=5)
    stability_results.to_csv("data/q_algo_div_250.csv")
    
    q_algo_div_1000 = QportDiversification(name="Qv3_diversification_diverse_1000", p=4, maxiter=300, lam_card=1000)
    stability_results = engine.run_stability_test(q_algo_div_1000, num_runs=5)
    stability_results.to_csv("data/q_algo_div_1000.csv")
    
    # q_algo_int = QportInteger(name="Qv2_integer", p=4, maxiter=300, lam_card=50)
    # stability_results = engine.run_stability_test(q_algo_int, num_runs=15)
    # stability_results.to_csv("data/quantum_stability_results_v2.csv")
    
    # q_algo_bin = QportBinary(name="Qv1_binary", p=4, maxiter=300, lam_card=50)
    # stability_results = engine.run_stability_test(q_algo_bin, num_runs=15)
    # stability_results.to_csv("data/quantum_stability_results_v1.csv")
    
    # Classical
    classical_algo = ClassicalPortfolio(name="Markowitz_Baseline")
    engine.run_benchmark([
        q_algo_div_50, 
        q_algo_div_250, 
        q_algo_div_1000, 
        classical_algo
    ])
    df_results = engine.get_comparison_table()

    # Print with a specific width and no truncation
    logger.info(df_results.to_string(index=False, justify='center', max_colwidth=40))
    engine.export_results(f"data/latest_benchmark{datetime.now().__str__()}.txt")
    
    # config = default_config()

    # prices = download_close_prices(
    #     tickers=config.tickers,
    #     period=config.period,
    #     interval=config.interval,
    # )
    # save_prices(prices, Path("data/close_prices.csv"))

    # classical = solve_max_sharpe(prices)
    

    # qubo = build_binary_selection_qubo(
    #     mu=classical.mu,
    #     covariance=classical.covariance,
    #     risk_aversion=config.risk_aversion,
    #     target_cardinality=config.target_cardinality,
    #     penalty_strength=config.penalty_strength,
    # )

    # x0 = np.zeros(len(qubo.tickers), dtype=int)
    # x0[: config.target_cardinality] = 1
    # baseline_energy = qubo_energy(x0, qubo)

    # print("\nQUBO model ready")
    # print("Assets:", len(qubo.tickers))
    # print("Q matrix shape:", qubo.q_matrix.shape)
    # print("Sample feasible energy:", round(baseline_energy, 6))



