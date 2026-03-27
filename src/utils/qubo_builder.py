# QUBO Builder: Converts expected returns and covariance into a QUBO model for portfolio optimization.

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class QuboModel:
    q_matrix: np.ndarray
    linear: np.ndarray
    constant: float
    tickers: List[str]


def build_binary_selection_qubo(
    mu: pd.Series,
    covariance: pd.DataFrame,
    risk_aversion: float,
    target_cardinality: int,
    penalty_strength: float,
) -> QuboModel:
    if list(mu.index) != list(covariance.index):
        covariance = covariance.loc[mu.index, mu.index]

    n = len(mu)
    ones = np.ones(n)

    # Objective: risk_aversion * x^T Sigma x - mu^T x + penalty * (sum(x) - K)^2
    sigma = covariance.to_numpy(dtype=float)
    mu_vec = mu.to_numpy(dtype=float)

    q_risk = risk_aversion * sigma

    penalty_quadratic = penalty_strength * np.outer(ones, ones)
    penalty_linear = penalty_strength * (1 - 2 * target_cardinality) * ones
    penalty_constant = penalty_strength * (target_cardinality ** 2)

    q_matrix = q_risk + penalty_quadratic
    linear = -mu_vec + penalty_linear

    return QuboModel(
        q_matrix=q_matrix,
        linear=linear,
        constant=float(penalty_constant),
        tickers=list(mu.index),
    )


def qubo_energy(x: np.ndarray, model: QuboModel) -> float:
    x = x.astype(float)
    return float(x.T @ model.q_matrix @ x + model.linear.T @ x + model.constant)
