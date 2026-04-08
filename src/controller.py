from src.financial_context.command import FinancialContextCommand
from src.portfolio.quantum_portfolio import QPortRiskRewardCardinality, QportHyperparameterProduct, QMethod, QPortRiskReward, QPortRiskRewardCardinalityTurnover
from src.portfolio.portfolio_base import PortfolioResult
from src.utils.logging_mod import get_logging
from etc.config import DEFAULT_PROJECT_CONFIG, DATAPATH
from src.portfolio.classical_portfolio import IdealClassicalPortfolio, DiscretePortfolio
import os
import pandas as pd
import time
import itertools
import json
import numpy as np

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

def automated_classical_grid_search():
    csv_filename = DATAPATH / 'classical_grid_search_results.csv'
    checkpoint_file = DATAPATH / 'classical_checkpoint.json'
    
    # We use the EXACT same space as the Quantum run to ensure 1:1 row comparison
    space = {
        'l_risk': [0.1, 0.5, 1.0, 5.0],
        'l_reward': [1, 5, 10],
        'l_card': [0.5, 1, 10],
        'k_target': [1, 2, 4],
        'mu_scalar': [1],
        'n_assets': [4, 8],
        'n_bits' : [1, 2, 3], # Included to maintain row alignment and discrete scaling
        'algo': [('CLASSICAL_HEURISTIC', 0), ('CLASSICAL_UNCONSTRAINED', 0)] 
    }
    
    keys = space.keys()
    iterations = list(itertools.product(*space.values()))
    total_runs = len(iterations)

    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            start_index = json.load(f).get("last_completed_index", -1) + 1
        logger.info(f"🔄 Resuming Classical Baseline from {start_index}/{total_runs}...")

    COLUMN_ORDER = [
        'method', 'p_layers', 'n_assets', 'n_bits',
        'lambda_risk', 'lambda_reward', 'lambda_cardinality',
        'K_target', 'mu_scalar',
        'expected_return', 'annual_vol', 'sharpe_ratio', 
        'active_assets', 'hhi_diversity', 
        'prob_success', 'base_energy', 'mean_neighborhood_energy', 
        'ruggedness', 'time_to_solve', 'portfolio_name',
        'choice', 'neighbor_energies'
    ]

    if not os.path.isfile(csv_filename):
        pd.DataFrame(columns=COLUMN_ORDER).to_csv(csv_filename, index=False)

    for i in range(start_index, total_runs):
        conf = dict(zip(keys, iterations[i]))
        mode_name, _ = conf['algo']
        
        current_tickers = DEFAULT_PROJECT_CONFIG.tickers[:conf['n_assets']]
        proxy = FinancialContextCommand(current_tickers, csv_path="./data/close_prices.csv")
        context = proxy.get_context()
        mu, sigma = context.get_moments(False, False)

        logger.info(f"[{i+1}/{total_runs}] Classical {mode_name} | N:{conf['n_assets']} Bits:{conf['n_bits']}")

        start_t = time.perf_counter()
        
        try:
            hparams = QportHyperparameterProduct(
                p_qaoa_layers=0, 
                max_iterations=150, 
                samples=256,
                shots=1024, 
                seed=42, 
                lambda_cardinality=conf['l_card'],
                lambda_risk=conf['l_risk'], 
                lambda_reward=conf['l_reward'], 
                k_cardinality=conf['k_target'], 
                n_bits=conf['n_bits'],
                mu_scalar=conf['mu_scalar'],
                prev_weights=0.0
            )
            # 1. Select the appropriate classical model
            if mode_name == 'CLASSICAL_UNCONSTRAINED':
                model = IdealClassicalPortfolio(name=mode_name, financial_context=proxy, hparams=hparams)
            else:
                model = DiscretePortfolio(name=mode_name, financial_context=proxy, k_assets=conf['k_target'], hparams=hparams)
            
            # 2. Execute Classical Optimization
            res = model.run()
            tts = time.perf_counter() - start_t
            
            # 3. CRITICAL: Calculate QUBO Energy for the classical result
            # This allows us to see if Quantum found a lower energy than the CPU's heuristic
            w_vec = np.array(list(res.weights.values()))

            active_count = sum(1 for w in w_vec if w > 1e-4)
            

            if mode_name == 'CLASSICAL_HEURISTIC': # This is your DiscretePortfolio
                assert isinstance(model, DiscretePortfolio)
                # 1. Map to bits (Discrete only)
                bitstring = model.weights_to_bitstring(res.weights)
                
                # 2. Get the same QUBO dict used by the Quantum models
                qubo_dict = model.build_qubo() 
                
                # 3. Calculate Ruggedness
                landscape_report = model.map_landscape_ruggedness(bitstring, qubo_dict)
                
                base_energy = landscape_report['base_energy']
                ruggedness = landscape_report['ruggedness']
                mean_neighbor = landscape_report['mean_neighborhood_energy']
                neighbor_energies_str = landscape_report['neighbor_energies']
            else: 
                # IdealClassicalPortfolio: Ruggedness is not applicable to continuous space
                base_energy = 0.0 
                ruggedness = 0.0
                mean_neighbor = 0.0
                neighbor_energies_str = "[]"
            
            # 4. Data Formatting
            row = pd.DataFrame([{
                'method': mode_name,
                'p_layers': 0,
                'n_assets': conf['n_assets'],
                'n_bits': conf['n_bits'],
                'lambda_risk': conf['l_risk'],
                'lambda_reward': conf['l_reward'],
                'lambda_cardinality': conf['l_card'],
                'K_target': conf['k_target'],
                'mu_scalar': conf['mu_scalar'],
                'expected_return': round(res.expected_return, 6),
                'annual_vol': round(res.volatility, 6),
                'sharpe_ratio': round(res.sharpe_ratio, 6),
                'active_assets': active_count,
                'hhi_diversity': sum(w**2 for w in w_vec),
                'prob_success': 1.0, 
                'base_energy': round(base_energy, 6),
                'mean_neighborhood_energy': mean_neighbor,
                'ruggedness': ruggedness, 
                'time_to_solve': tts,
                'portfolio_name': json.dumps(model.name),
                'choice': json.dumps(res.weights),
                'neighbor_energies': neighbor_energies_str,
            }])

            row.reindex(columns=COLUMN_ORDER).to_csv(csv_filename, mode='a', header=False, index=False)
            
            with open(checkpoint_file, 'w') as f:
                json.dump({"last_completed_index": i}, f)
                
        except Exception as e:
            logger.warning(f"❌ Classical Error at {i}: {e}")
            continue

    logger.info("🏁 Classical Control Search Complete.")

def automated_quantum_grid_search():
    csv_filename = DATAPATH / 'quantum_grid_search_results.csv'
    checkpoint_file = DATAPATH / 'grid_search_checkpoint.json'
    # Define ranges
    
    # Hierarchical Ranges
    space = {
        'l_risk': [0.1, 0.5, 1.0, 5.0],
        'l_reward': [1, 5, 10],
        'l_card': [0.5, 1, 10],
        'k_target': [1, 2, 4],
        'mu_scalar': [1],
        'n_assets': [4, 8],
        'n_bits' : [1, 2, 3],
        # Algorithm Tuple: (Method, p_layers)
        'algo': [(QMethod.ANALOG_BASED, 0), 
                 (QMethod.GATE_BASED, 1), 
                 (QMethod.GATE_BASED, 2),
                 (QMethod.GATE_BASED, 3)]
    }
    keys = space.keys()
    iterations = list(itertools.product(*space.values()))
    total_runs = len(iterations)

    # 2. Load Checkpoint (Resume Logic)
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            start_index = checkpoint_data.get("last_completed_index", -1) + 1
        logger.info(f"🔄 Resuming from iteration {start_index}/{total_runs}...")
    else:
        logger.info(f"🚀 Starting fresh Grid Search: {total_runs} total runs.")
    COLUMN_ORDER=[
            # --- Metadata & Logic ---
            'method', 
            'p_layers', 
            'n_assets', 
            'n_bits',
            
            # --- Hyperparameters ---
            'lambda_risk', 
            'lambda_reward', 
            'lambda_cardinality',
            'K_target', 
            'mu_scalar',
            
            # --- Portfolio Performance ---
            'expected_return', 
            'annual_vol', 
            'sharpe_ratio', 
            'active_assets', 
            'hhi_diversity', 
            
            # --- Landscape & Optimization Metrics ---
            'prob_success', 
            'base_energy', 
            'mean_neighborhood_energy', 
            'ruggedness', 
            'time_to_solve', 
            'portfolio_name',
            # --- Execution Data ---            
            'choice', # The weights dictionary
            'neighbor_energies', # This will store the stringified list

        ]
    if not os.path.isfile(csv_filename):
        pd.DataFrame(columns=COLUMN_ORDER).to_csv(csv_filename, index=False)
        
        
    for i in range(start_index, total_runs):
        values = iterations[i]
        conf = dict(zip(keys, values))
        mode, p_val = conf['algo']
        
        # Setup Context for specific asset size
        current_tickers = DEFAULT_PROJECT_CONFIG.tickers[:conf['n_assets']]
        proxy = FinancialContextCommand(current_tickers, csv_path="./data/close_prices.csv", scalar_gain=1.0)
        
        logger.info(f"Testing {mode.name} | N:{conf['n_assets']} tickers:[{current_tickers}] P:{p_val} | R:{conf['l_risk']} C:{conf['l_card']}...")

        hparams = QportHyperparameterProduct(
            p_qaoa_layers=p_val, 
            max_iterations=150, 
            samples=256,
            shots=1024, 
            seed=42, 
            lambda_cardinality=conf['l_card'],
            lambda_risk=conf['l_risk'], 
            lambda_reward=conf['l_reward'], 
            k_cardinality=conf['k_target'], 
            n_bits=conf['n_bits'],
            mu_scalar=conf['mu_scalar'],
            prev_weights=0.0
        )

        model = QPortRiskRewardCardinalityTurnover(
            name=f"{mode.name}-N{conf['n_assets']}-P{p_val}",
            proxy=proxy, hparaproduct=hparams
        )
        
        try:
            progress = ((i + 1) / total_runs) * 100
            logger.info(f"[{progress:.1f}%] Testing {mode.name} | N:{conf['n_assets']} Bits:{conf['n_bits']} | Iter {i+1}/{total_runs}")
            start_t = time.perf_counter()
            metrics = model.run(mode)
            tts = time.perf_counter() - start_t
            
            # Data Formatting
            weights = list(metrics.weights.values())
            row = pd.DataFrame([{
                # --- Experiment Metadata ---
                'method': mode.name,
                'p_layers': p_val,
                'n_assets': conf['n_assets'],
                'n_bits' : conf['n_bits'],
                # --- Hyperparameters ---
                'lambda_risk': conf['l_risk'],
                'lambda_reward': conf['l_reward'],
                'lambda_cardinality': conf['l_card'],
                'K_target': conf['k_target'],
                'mu_scalar': conf['mu_scalar'],
                
                # --- Financial Performance Metrics ---
                'expected_return': round(metrics.expected_return, 6),
                'annual_vol': round(metrics.volatility, 6),
                'sharpe_ratio': round(metrics.sharpe_ratio, 6),
                'active_assets': sum(1 for w in weights if w > 1e-4),
                'hhi_diversity': sum(w**2 for w in weights) if weights else 1.0,
                
                # --- Optimization & Landscape Metrics ---
                'prob_success': round(metrics.config.get("prob_success", 0.0), 6),
                'base_energy': round(metrics.config.get("base_energy", 0.0), 6),
                'mean_neighborhood_energy': round(metrics.config.get("mean_neighborhood_energy", 0.0), 6),
                'ruggedness': round(metrics.config.get("ruggedness", 0.0), 6),
                # --- Execution Metadata ---
                'time_to_solve': tts,
                'portfolio_name': json.dumps(model.name),
                'choice': json.dumps(metrics.weights),
                # -- Neighbood ---
                'neighbor_energies': json.dumps(metrics.config.get("neighbor_energies", [])), # <--- Stringified List
                
            }])
            row = row.reindex(columns=COLUMN_ORDER)
            row.to_csv(csv_filename, mode='a', header=False, index=False)

            # SAVE CHECKPOINT AFTER SUCCESSFUL SAVE
            with open(checkpoint_file, 'w') as f:
                json.dump({"last_completed_index": i}, f)
            
            logger.info(f"✅ Iteration {i+1}/{total_runs} Saved (N:{conf['n_assets']} P:{p_val})")
            
        except Exception as e:
            logger.warning(f"❌ Error at iteration {i}: {e}")
            # We don't update checkpoint here so we can retry this specific one later
            continue

    logger.info(f"🏁 Grid Search Complete.")


def automated_quantumannealing_grid_search():
    csv_filename = DATAPATH / 'classicalannealing_grid_search_results.csv'
    checkpoint_file = DATAPATH / 'grid_classicalannealing_search_checkpoint.json'
    # Define ranges
    
    # Hierarchical Ranges
    space = {
        'l_risk': [0.1, 0.5, 1.0, 5.0],
        'l_reward': [1, 5, 10],
        'l_card': [0.5, 1, 10],
        'k_target': [1, 2, 4],
        'mu_scalar': [1],
        'n_assets': [4, 8],
        'n_bits' : [1, 2, 3],
        # Algorithm Tuple: (Method, p_layers)
        'algo': [(QMethod.ANALOG_BASED, 0)]
    }
    keys = space.keys()
    iterations = list(itertools.product(*space.values()))
    total_runs = len(iterations)

    # 2. Load Checkpoint (Resume Logic)
    start_index = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint_data = json.load(f)
            start_index = checkpoint_data.get("last_completed_index", -1) + 1
        logger.info(f"🔄 Resuming from iteration {start_index}/{total_runs}...")
    else:
        logger.info(f"🚀 Starting fresh Grid Search: {total_runs} total runs.")
    COLUMN_ORDER=[
            # --- Metadata & Logic ---
            'method', 
            'p_layers', 
            'n_assets', 
            'n_bits',
            
            # --- Hyperparameters ---
            'lambda_risk', 
            'lambda_reward', 
            'lambda_cardinality',
            'K_target', 
            'mu_scalar',
            
            # --- Portfolio Performance ---
            'expected_return', 
            'annual_vol', 
            'sharpe_ratio', 
            'active_assets', 
            'hhi_diversity', 
            
            # --- Landscape & Optimization Metrics ---
            'prob_success', 
            'base_energy', 
            'mean_neighborhood_energy', 
            'ruggedness', 
            'time_to_solve', 
            'portfolio_name',
            # --- Execution Data ---            
            'choice', # The weights dictionary
            'neighbor_energies', # This will store the stringified list

        ]
    if not os.path.isfile(csv_filename):
        pd.DataFrame(columns=COLUMN_ORDER).to_csv(csv_filename, index=False)
        
        
    for i in range(start_index, total_runs):
        values = iterations[i]
        conf = dict(zip(keys, values))
        mode, p_val = conf['algo']
        
        # Setup Context for specific asset size
        current_tickers = DEFAULT_PROJECT_CONFIG.tickers[:conf['n_assets']]
        proxy = FinancialContextCommand(current_tickers, csv_path="./data/close_prices.csv", scalar_gain=1.0)
        
        logger.info(f"Testing {mode.name} | N:{conf['n_assets']} tickers:[{current_tickers}] P:{p_val} | R:{conf['l_risk']} C:{conf['l_card']}...")

        hparams = QportHyperparameterProduct(
            p_qaoa_layers=p_val, 
            max_iterations=150, 
            samples=256,
            shots=1024, 
            seed=42, 
            lambda_cardinality=conf['l_card'],
            lambda_risk=conf['l_risk'], 
            lambda_reward=conf['l_reward'], 
            k_cardinality=conf['k_target'], 
            n_bits=conf['n_bits'],
            mu_scalar=conf['mu_scalar'],
            prev_weights=0.0
        )

        model = QPortRiskRewardCardinalityTurnover(
            name=f"{mode.name}-N{conf['n_assets']}-P{p_val}_CAonly",
            proxy=proxy, hparaproduct=hparams
        )
        
        try:
            progress = ((i + 1) / total_runs) * 100
            logger.info(f"[{progress:.1f}%] Testing {mode.name} | N:{conf['n_assets']} Bits:{conf['n_bits']} | Iter {i+1}/{total_runs}")
            start_t = time.perf_counter()
            metrics = model.run(mode)
            tts = time.perf_counter() - start_t
            
            # Data Formatting
            weights = list(metrics.weights.values())
            row = pd.DataFrame([{
                # --- Experiment Metadata ---
                'method': mode.name,
                'p_layers': p_val,
                'n_assets': conf['n_assets'],
                'n_bits' : conf['n_bits'],
                # --- Hyperparameters ---
                'lambda_risk': conf['l_risk'],
                'lambda_reward': conf['l_reward'],
                'lambda_cardinality': conf['l_card'],
                'K_target': conf['k_target'],
                'mu_scalar': conf['mu_scalar'],
                
                # --- Financial Performance Metrics ---
                'expected_return': round(metrics.expected_return, 6),
                'annual_vol': round(metrics.volatility, 6),
                'sharpe_ratio': round(metrics.sharpe_ratio, 6),
                'active_assets': sum(1 for w in weights if w > 1e-4),
                'hhi_diversity': sum(w**2 for w in weights) if weights else 1.0,
                
                # --- Optimization & Landscape Metrics ---
                'prob_success': round(metrics.config.get("prob_success", 0.0), 6),
                'base_energy': round(metrics.config.get("base_energy", 0.0), 6),
                'mean_neighborhood_energy': round(metrics.config.get("mean_neighborhood_energy", 0.0), 6),
                'ruggedness': round(metrics.config.get("ruggedness", 0.0), 6),
                # --- Execution Metadata ---
                'time_to_solve': tts,
                'portfolio_name': json.dumps(model.name),
                'choice': json.dumps(metrics.weights),
                # -- Neighbood ---
                'neighbor_energies': json.dumps(metrics.config.get("neighbor_energies", [])), # <--- Stringified List
                
            }])
            row = row.reindex(columns=COLUMN_ORDER)
            row.to_csv(csv_filename, mode='a', header=False, index=False)

            # SAVE CHECKPOINT AFTER SUCCESSFUL SAVE
            with open(checkpoint_file, 'w') as f:
                json.dump({"last_completed_index": i}, f)
            
            logger.info(f"✅ Iteration {i+1}/{total_runs} Saved (N:{conf['n_assets']} P:{p_val})")
            
        except Exception as e:
            logger.warning(f"❌ Error at iteration {i}: {e}")
            # We don't update checkpoint here so we can retry this specific one later
            continue

    logger.info(f"🏁 Grid Search Complete.")
    

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
        p_qaoa_layers=4,
        max_iterations=1024,
        samples=1024,
        shots=2048,
        seed=42,
        lambda_cardinality=10,
        lambda_risk=1.5,
        lambda_reward=5,
        k_cardinality=3,
        n_bits = 4,
        lambda_turnover=1
    )

    portfolio_model = QPortRiskRewardCardinality(
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
    
    # classical = IdealClassicalPortfolio("Classical-Control", proxy)
    # try:
    #     # result = classical.run()
    #     # results.append(result)
    # except Exception as e:
    #     logger.error(f"❌ Portfolio Run Crashed: {str(e)}", exc_info=True)
    # for i in results:
    #     log_portfolio_results(i)
    # portfolio_model.run_quantum_annealing_analysis()
    
if __name__ == "__main__":
    # run_portfolio_stress_test()
    automated_classical_grid_search()