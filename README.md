# qcomp-fin

Quantum Portfolio Optimization (Mean-Variance) starter project.

Goal: build an Efficient Frontier workflow with a classical Markowitz baseline, then encode a binary portfolio-selection version as QUBO for quantum optimization (QAOA or VQE).

## 1) Quick Start

Open a terminal in the project root and run:

python -m venv .venv

Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt

Run the baseline pipeline:

python src/main.py

What this does:
- Downloads 5 years of daily close prices for 10 tech stocks with yfinance.
- Computes expected returns and covariance matrix.
- Solves a classical max-Sharpe portfolio with PyPortfolioOpt.
- Builds a QUBO matrix for binary asset selection.

## 2) Project Structure

- src/config.py: tickers and model hyperparameters.
- src/data_pipeline.py: market data download and CSV IO.
- src/classical_baseline.py: Markowitz baseline using PyPortfolioOpt.
- src/qubo_builder.py: QUBO conversion for binary selection portfolio.
- src/main.py: end-to-end run for Week 1 setup.
- data/: downloaded market data.
- results/: output artifacts for later experiments.

## 3) Mapping to Your 7-Week Plan

Weeks 1-2:
- Tune ticker universe in src/config.py.
- Save baseline performance metrics into results/.
- Add Efficient Frontier plot for write-up.

Weeks 3-4:
- Start with 4-5 assets only.
- Use QUBO from src/qubo_builder.py and convert to Ising/Qiskit Optimization model.

Weeks 5-6:
- Run QAOA on simulator.
- Compare Sharpe ratio and risk-return points versus classical baseline.

Week 7:
- Final LaTeX write-up with probability distributions and hardware noise discussion.

## 4) First Edits You Should Make

In src/config.py:
- tickers: your chosen 10-20 names.
- target_cardinality: number of assets allowed in selected basket.
- risk_aversion and penalty_strength: trade-off tuning parameters.

## 5) Next Implementation Step

After this setup runs, add a new file for Qiskit Optimization integration:
- Convert the QUBO matrix into a QuadraticProgram.
- Solve with a local QAOA sampler.
- Save bitstring frequency distribution and best portfolio candidate to results/.
