from src.financial_context.command import FinancialContextCommand
from src.portfolio.quantum_portfolio import QPortRiskRewardCardinality, QportHyperparameterProduct, QMethod, QPortRiskReward, QPortRiskRewardCardinalityTurnover
from src.portfolio.portfolio_base import PortfolioResult
from src.utils.logging_mod import get_logging
from etc.config import DEFAULT_PROJECT_CONFIG, DATAPATH
from src.portfolio.classical_portfolio import ClassicalPortfolio1
import os
import pandas as pd
import time
# Initialize specialized logging for the Quantum Pipeline
logger = get_logging(__name__)

def log_portfolio_results(result):
    # 1. Filter for active assets (using a 0.1% threshold to ignore float noise)
    active_weights = {ticker: w for ticker, w in result.weights.items() if w > 0.001}
    sorted_assets = sorted(active_weights.items(), key=lambda x: x[1], reverse=True)
    
    # 2. Build the "Selected Assets" string with Names
    if not sorted_assets:
        assets_str = "⚠️  NO ASSETS SELECTED (Optimization converged to null)"
    else:
        # Format the top 5 tickers with their allocation percentages
        top_assets = [f"{ticker} ({w:.1%})" for ticker, w in sorted_assets[:5]]
        assets_str = ", ".join(top_assets)
        
        # Append count if there are more than 5 assets in the basket
        if len(sorted_assets) > 5:
            assets_str += f" ... (+{len(sorted_assets)-5} more)"

    # 3. Construct the "Visual Report" Buffer
    # Added padding and box-drawing characters for better terminal visibility
    report = (
        f"\n{'#'*60}\n"
        f"📊 PORTFOLIO EXECUTION COMPLETE: {result.method_name}\n"
        f"{'━'*60}\n"
        f"  PERFORMANCE METRICS:\n"
        f"    • Expected Return:  {result.expected_return:>10.2%}\n"
        f"    • Annual Vol:       {result.volatility:>10.2%}\n"
        f"    • Sharpe Ratio:     {result.sharpe_ratio:>12.4f}\n"
        f"    • HHI Diversity:    {result.config.get('hhi_index', 0):>12.4f}\n"
        f"{'─'*60}\n"
        f"  EXECUTION STRATEGY:\n"
        f"    • Model Type:       {result.config.get('strategy', 'Quantum-Hybrid')}\n"
        f"    • Active Assets:    {len(active_weights)} / {len(result.weights)}\n"
        f"    • Tickers:          {assets_str}\n"
        f"{'#'*60}"
    )

    # 4. Single Atomic Log Entry
    logger.info(report)
    
def automated_grid_search():
    csv_filename = DATAPATH / 'quantum_grid_search_results.csv'
    # Define ranges
    mu_scalars = [10, 100, 500] 
    lambda_risks = [0.1, 0.5, 1.0, 5.0, 10.0]
    lambda_turnover = [0.1, 1.0, 10.0] # High, Med, Low friction
    lambda_cards = [1, 10, 50, 100, 500]
    lambda_reward = [5.0, 25.0, 50.0, 100.0, 200.0]
    k_targets = [3, 5]
    
    
    print(f"🚀 Starting Grid Search: {len(lambda_risks) * len(lambda_cards) * len(k_targets)} iterations.")
    

    if not os.path.isfile(csv_filename):
        pd.DataFrame(columns=[
            'lambda_risk', 'lambda_reward', 'lambda_cardinality', 'lambda_turnover','K_target', 
            'expected_return', 'annual_vol', 'sharpe_ratio', 
            'active_assets', 'hhi_diversity', "base_energy", "neighbor_energies", "ruggedness", "mu_scalar", 'time_to_solve', 'portfolio_name', 'choice'
        ]).to_csv(csv_filename, index=False)
    for mu_scalar in mu_scalars:
        for lturno in lambda_turnover:
            for lrisk in lambda_risks:
                for lcardinality in lambda_cards:
                    for kt in k_targets:
                        for lreward in lambda_reward:
                            proxy = FinancialContextCommand(DEFAULT_PROJECT_CONFIG.tickers, csv_path="./data/close_prices.csv",scalar_gain=mu_scalar)
                            print(f"Testing: Risk={lrisk}, Card_Weight={lcardinality}, K={kt}...", end=" ", flush=True)
                            
                            hparams = QportHyperparameterProduct(
                                p_qaoa_layers=3, 
                                max_iterations=1024, 
                                samples=1024,
                                shots=2048, 
                                seed=42, 
                                lambda_cardinality=lcardinality,
                                lambda_risk=lrisk, 
                                lambda_reward=lreward, 
                                k_cardinality=kt, 
                                n_bits=8,
                                mu_scalar=mu_scalar,
                                lambda_turnover=lturno,
                                prev_weights=None
                            )

                            portfolio_model = QPortRiskRewardCardinalityTurnover(
                                name="Quantum-Beta-Risk-Reward-Cardinality-Turnover-Portfolio",
                                proxy=proxy, hparaproduct=hparams
                            )
                            
                            try:
                                start_time = time.perf_counter()
                                # metrics is an instance of PortfolioResult
                                metrics = portfolio_model.run(QMethod.ANALOG_BASED)
                                tts = time.perf_counter() - start_time
                                
                                # Calculate derived metrics from the weights dictionary
                                weights_list = list(metrics.weights.values())
                                active_assets = sum(1 for w in weights_list if w > 1e-4) # Count non-zero weights
                                
                                # Herfindahl-Hirschman Index (Diversity)
                                # HHI = sum(w^2). Lower is more diverse.
                                hhi = sum(w**2 for w in weights_list) if weights_list else 1.0
                                row = pd.DataFrame([{
                                    'lambda_risk': lrisk,
                                    'lambda_reward': lreward,
                                    'lambda_cardinality': lcardinality,
                                    'lambda_turnover': lturno,
                                    'K_target': kt,
                                    'expected_return': metrics.expected_return,
                                    'annual_vol': metrics.volatility,
                                    'sharpe_ratio': metrics.sharpe_ratio,
                                    'active_assets': active_assets,
                                    'hhi_diversity': hhi,
                                    'base_energy': metrics.config.get("base_energy", 0.0),
                                    'neighbor_energies': metrics.config.get("neighbor_energies", 0.0),
                                    'ruggedness': metrics.config.get("ruggedness", 0.0),
                                    "mu_scalar": mu_scalar,
                                    "time_to_solve": tts,
                                    'portfolio_name': portfolio_model.name,
                                    'choice': metrics.weights
                                }])
                                
                                row.to_csv(csv_filename, mode='a', header=False, index=False)
                                print("✅ Saved.")
                                
                            except Exception as e:
                                print(f"❌ Error: {e}")

    print(f"✅ Grid Search Complete. Final results in '{csv_filename}'")
    
def run_portfolio_stress_test():
    """
    Orchestrates a full 20-asset Portfolio Optimization.
    1. Loads Financial Data
    2. Connects to GPU Server
    3. Runs QAOA + MPT Hybrid Optimization
    """
    logger.info("🎬 Initializing Quantum Portfolio Stress Test...")

    # 1. Setup Financial Context (Points to your close_prices.csv)
    # The proxy handles the mu (returns) and sigma (covariance) calculation
    config = DEFAULT_PROJECT_CONFIG
    
    proxy = FinancialContextCommand(config.tickers, csv_path="./data/close_prices.csv")
    
    # 2. Configure Hyperparameters
    # We'll use p=4 for high-precision landscape mapping
    hparams = QportHyperparameterProduct(
        p_qaoa_layers=3,
        max_iterations=1024,
        samples=1024,
        shots=2048,
        seed=42,
        lambda_cardinality=10,
        lambda_risk=1.5,
        lambda_reward=5,
        k_cardinality=3,
        n_bits = 6,
        lambda_turnover=1
    )

    portfolio_model = QPortRiskRewardCardinalityTurnover(
        name="Quantum-Beta-Diversifier",
        proxy=proxy,
        hparaproduct=hparams
    )
    results: list[PortfolioResult] = []
    k1 = portfolio_model.run(QMethod.ANALOG_BASED)
    log_portfolio_results(k1)
    try:
        portfolio_model.run_quantum_annealing_analysis()
        # results.append(portfolio_model.run())
        results.append(k1)
    except Exception as e:
        logger.error(f"❌ Portfolio Run Crashed: {str(e)}", exc_info=True)
    
    classical = ClassicalPortfolio1("Classical-Control", proxy)
    try:
        result = classical.run()
        results.append(result)
    except Exception as e:
        logger.error(f"❌ Portfolio Run Crashed: {str(e)}", exc_info=True)
    for i in results:
        log_portfolio_results(i)
    portfolio_model.run_quantum_annealing_analysis()
if __name__ == "__main__":
    # run_portfolio_stress_test()
    automated_grid_search()