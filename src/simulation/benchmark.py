from dataclasses import dataclass
from typing import Dict, List
import pandas as pd
import numpy as np
from src.utils.portfolio import PortfolioResult, Portfolio
from datetime import datetime
from src.utils.logging_mod import get_logging

logger = get_logging(__name__)
class BenchmarkEngine:
    def __init__(self, prices: pd.DataFrame, esg_metadata: Dict[str, float] = {}, fear_greed: float = 0.5):
        self.prices = prices
        self.esg_metadata = esg_metadata or {}
        self.fear_greed = fear_greed
        self.results: List[PortfolioResult] = []

    def run_benchmark(self, solvers: List[Portfolio]):
        """Runs a single pass for a list of different solver algorithms."""
        for portfolio in solvers:
            try:
                # Note: Ensure your Portfolio.run signature matches your latest implementation
                result = portfolio.run(self.prices)
                self.results.append(result)
                logger.info(f"Successfully ran benchmark for: {portfolio.name}")
            except Exception as e:
                logger.error(f"Solver {portfolio.name} failed: {e}")
                
    def run_stability_test(self, portfolio: Portfolio, num_runs: int = 10) -> pd.DataFrame:
        """
        Runs the SAME portfolio multiple times with different seeds 
        to calculate variance in Sharpe Ratio and Asset Selection.
        """
        seeds = range(60, 60 + num_runs)
        stability_data = []

        logger.info(f"Starting Stability Test for {portfolio.name} ({num_runs} runs)")

        for i, seed in enumerate(seeds):
            try:
                # Dynamically update the seed attribute of the portfolio class
                if hasattr(portfolio, 'seed'):
                    portfolio.seed = seed
                
                res = portfolio.run(self.prices)
                
                # Extract only non-zero assets for selection analysis
                active_assets = tuple(sorted([k for k, v in res.weights.items() if v > 0]))
                
                stability_data.append({
                    "run": i + 1,
                    "seed": seed,
                    "method": portfolio.name,
                    "sharpe": res.sharpe_ratio,
                    "return": res.expected_return,
                    "volatility": res.volatility,
                    "assets": active_assets,
                    "num_assets": len(active_assets)
                })
            except Exception as e:
                logger.warning(f"Run {i+1} failed with seed {seed}: {e}")

        df = pd.DataFrame(stability_data)
        self._print_stability_summary(df, portfolio.name)
        return df

    def _print_stability_summary(self, df: pd.DataFrame, name: str):
        """Prints a quick statistical breakdown for your logs."""
        if df.empty: return
        
        most_common_set = df['assets'].mode()[0]
        selection_consistency = (df['assets'] == most_common_set).mean()

        logger.info(f"--- Stability Summary: {name} ---")
        logger.info(f"Mean Sharpe: {df['sharpe'].mean():.4f} (±{df['sharpe'].std():.4f})")
        logger.info(f"Selection Consistency: {selection_consistency:.2%}")
        logger.info(f"Most Frequent Portfolio: {most_common_set}")
        logger.info("---------------------------------")

    def get_comparison_table(self) -> pd.DataFrame:
        """Returns the results of the standard benchmark runs."""
        return pd.DataFrame([r.__dict__ for r in self.results])
    
    def export_results(self, filename: str = "benchmark_report.txt"):
        """Exports the current results and stability metrics to a formatted text file."""
        df = self.get_comparison_table()
        
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"PORTFOLIO OPTIMIZATION BENCHMARK REPORT\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*100 + "\n\n")

            # 1. SUMMARY STATS TABLE (Cleaned up)
            f.write("### 1. SUMMARY PERFORMANCE METRICS ###\n")
            # We drop the heavy columns for the summary table view
            summary_cols = ['method_name', 'expected_return', 'volatility', 'sharpe_ratio']
            f.write(df[summary_cols].to_string(index=False, justify='left'))
            f.write("\n\n" + "="*100 + "\n\n")

            # 2. FULL RAW DATA (No Truncation)
            f.write("### 2. DETAILED BREAKDOWN & CONFIGURATION ###\n\n")
            for _, row in df.iterrows():
                f.write(f"METHOD: {row['method_name']}\n")
                f.write(f"{'-'*len(row['method_name'])}\n")
                
                # Write Weights (Active only for readability)
                f.write("Final Weights:\n")
                active_weights = {k: f"{v:.4%}" for k, v in row['weights'].items() if v > 0}
                for ticker, weight in active_weights.items():
                    f.write(f"  - {ticker}: {weight}\n")
                
                # Write Full Config (Manually iterate to avoid '...')
                f.write("\nAlgorithm Configuration:\n")
                if not row['config']:
                    f.write("  (None)\n")
                else:
                    for cfg_key, cfg_val in row['config'].items():
                        f.write(f"  - {cfg_key}: {cfg_val}\n")
                
                f.write(f"\nFinal Sharpe: {row['sharpe_ratio']:.6f}\n")
                f.write("\n" + "*"*50 + "\n\n")

        logger.info(f"Full report (no truncation) exported to {filename}")