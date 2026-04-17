# qcomp-fin

Quantum Portfolio Optimization (Mean-Variance) starter project.

Goal: build an Efficient Frontier workflow with a classical Markowitz baseline, then encode a binary portfolio-selection version as QUBO for quantum optimization (QAOA or VQE).

## 1) Quick Start

This project uses a Dual-Environment Architecture to separate high-level financial modeling from heavy GPU-accelerated simulation. This prevents C++ build conflicts and keeps the workspace clean.

Prerequisites:

- uv installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- NVIDIA Drivers & CUDA 12.x (for GPU acceleration)

### Initialize the Environments

We use uv sync to create two distinct environments. Run these commands from the project root:

#### A. The Architect (Modeling & Dashboards)

This environment contains qamomile, marimo, and financial libraries.

```bash
uv venv .venv-qam --python 3.12
uv sync --extra modeling --python ./.venv-qam/bin/python
```

#### B. The Engine (GPU Executor)

This environment contains qiskit-aer-gpu and cuquantum.

```bash
uv venv .venv-gpu --python 3.12
uv sync --extra gpu --python ./.venv-gpu/bin/python
```

#### Running

Simply run the command:
```bash
# (in ./.venv-qam/bin/python)
uv run --active main.py
```

### Code Architecture Outline

The project follows a modular, layered architecture with clear separation of concerns. It's structured around the src directory, with supporting folders for data, notebooks, tests, and configuration. The core flow is: data ingestion → problem modeling → optimization execution → result analysis.

#### 1. **Entry Point and Orchestration**

- **main.py**: Simple entry point that calls `controller.automated_classicalMILP_grid_search()` or similar grid search functions. Acts as a launcher for experiments.
- **controller.py**: Central orchestrator for running comparative grid searches. Handles:
  - Hyperparameter sweeps (e.g., risk/reward lambdas, asset counts, QAOA layers).
  - Checkpointing (resumes interrupted runs via JSON files).
  - Logging results to CSV (e.g., quantum_grid_search_results.csv).
  - Integration with both classical and quantum portfolios.
  - Functions like `automated_classical_grid_search()`, `automated_quantum_grid_search()`, and `automated_classicalMILP_grid_search()` for different experiment types.

#### 2. **Portfolio Optimization Layer (portfolio)**

Abstracts portfolio strategies into a common interface. All portfolios inherit from `PortfolioBase` and return a `PortfolioResult` (dataclass with weights, returns, volatility, Sharpe ratio, etc.).

- **portfolio_base.py**:
  - Defines `PortfolioResult` dataclass for standardized outputs.
  - `PortfolioBase` ABC: Abstract `run()` method; common helpers like `_finalize_portfolio()` (converts quantum bitstrings to weights, applies hybrid MPT refinement, calculates HHI diversification).
  - Handles edge cases (e.g., zero-asset selection fallback to top returns).

- **`classical_portfolio.py`**:
  - `IdealClassicalPortfolio`: Continuous optimization using PyPortfolioOpt's EfficientFrontier (max Sharpe).
  - `DiscretePortfolio`: Heuristic discrete selection (e.g., greedy or random) with cardinality constraints.
  - `DiscreteMILPPortfolio`: MILP-based discrete optimization using Gurobi/PuLP for exact solutions.

- **quantum_portfolio.py**:
  - `QuantumPortfolioComposite` ABC: Base for quantum portfolios; integrates with symbolics, instantiator, and minimizer.
  - Concrete classes like `QPortRiskRewardCardinality` (binary selection with risk/reward/cardinality penalties), `QPortRiskReward` (simpler binary), `QPortRiskRewardCardinalityTurnover` (adds transaction costs).
  - Handles QAOA parameter binding, result decoding (bitstrings to asset weights), and landscape analysis (energy ruggedness for benchmarking).
  - `QportHyperparameterProduct` dataclass: Bundles hyperparameters (QAOA layers, shots, lambdas for penalties).

#### 3. **Financial Data Layer (financial_context)**

Manages market data, moments calculation, and ESG integration.

- **`command.py`**: `FinancialContextCommand` class; downloads prices via yfinance, caches to CSV, builds `FinancialContext`.
- **context.py**: `FinancialContext` class; computes/scales expected returns (mu) and covariance (sigma) using PyPortfolioOpt; handles ESG scores (optional, defaults to neutral).
- **`data_collection.py`**: Utilities for data fetching and preprocessing.

#### 4. **Quantum Computation Layer (quantum)**

Handles quantum problem execution, separated into engines and middleware.

- **`engine/`**:
  - digital_engine.py: `GateBasedQuantumEngine` for QAOA; uses Qiskit Aer (GPU-preferred, CPU fallback); executes circuits and returns bitstring counts.
  - `analog_engine.py`: For simulated annealing (e.g., OpenJij); handles Ising model solving.
  - Other files: `entrypoint.py`, `logging_mod.py`, `server.py`.

- **`middleware/`**:
  - `instantiator.py`: `InstantiatorCommand`; builds QAOA circuits from QUBO (sets up ansatz, parameters).
  - `minimizer.py`: `MinimizerCommand`; runs optimization loops (e.g., COBYLA on QAOA parameters).
  - `bridge.py`: `ModelBridgeCommand`; bridges JijModeling to Qiskit execution.

#### 5. **Symbolic Modeling Layer (symbolics)**

Encodes financial problems as quantum Hamiltonians using JijModeling.

- **hamiltonian_modelling.py**: `QuantumProblemModelingBuilder` ABC; composes decision variables and Hamiltonian terms; builds Jij `Problem` objects.
- **`hamiltonians.py`**: Strategy classes for Hamiltonian components:
  - `RewardMuIntegerStrategy`: Penalizes low returns.
  - `RiskCovarianceIntegerStrategy`: Penalizes high variance.
  - `CardinalityBinaryStrategy`: Enforces asset count limits.
  - `TransactionCostStrategy`: Adds turnover penalties.
- **`decisions.py`**: Decision variable strategies (e.g., `BinarySelectionStrategy` for 0/1 choices, `IntegerWeightingStrategy` for multi-bit weights).

#### 6. **Utilities and Configuration (utils, etc)**

- **`utils/`**: `logging_mod.py` (centralized logging), `cache.py`, `job_manifest.py`.
- **config.py**: `ProjectConfig` dataclass with tickers (default: 16 S&P 500 stocks like AAPL, MSFT), periods, hyperparameters.
- **__init__.py**: Likely empty or imports.

#### 7. **Supporting Modules**

- **`dashboard/`**: Presumably for result visualization (e.g., efficient frontiers, comparisons).
- **tests**: Unit tests for portfolios, quantum engines, etc. (e.g., `test_portfolio_formulation.py`).
- **notebooks**: Marimo notebooks for analysis (`analysis.py`, `marimo_gridsearch.py`); summary reports in `layer1_summaries/` (e.g., classical annealing, quantum QAOA).
- **data**: CSVs for prices, results, checkpoints; subfolders for analog/digital saves.
- **scripts**: Shell scripts for environment setup (`uv_create_all.sh`).

### Component Roles and Interactions

- **Data Flow**: `FinancialContextCommand` → downloads data → `FinancialContext` computes moments → passed to portfolios.
- **Classical Path**: Portfolio classes use PyPortfolioOpt/Gurobi directly for optimization.
- **Quantum Path**: `QuantumPortfolioComposite` → `QuantumProblemModelingBuilder` (builds QUBO) → `InstantiatorCommand` (QAOA circuit) → `GateBasedQuantumEngine` (execute) → decode to weights.
- **Middleware**: Acts as adapters between symbolic models (Jij) and execution (Qiskit/OpenJij).
- **Controller**: Drives experiments, aggregates results, ensures reproducibility via seeds/checkpoints.
- **Symbolics**: Decouples problem definition from solvers; easy to add new constraints (e.g., ESG via `lambda_esg`).

This architecture enables easy extension (e.g., new quantum backends) and fair comparisons, with a focus on benchmarking quantum vs. classical for NP-hard portfolio problems. 