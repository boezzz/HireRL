

"""
Utilities for generating match-specific profits and updating firm beliefs
based on realized profits, consistent with the paper's specification.

Generated profit:

    p_{ij,t}
    = f [
        exp_{j,t-1}
        +
        (g_0 + g_1 * sigma_j)
        * 1{ j employed at t-1 }
        * exp(-theta * exp_{j,t-1})
      ] + eps_{ij,t},

    eps_{ij,t} ~ N(0, delta_eps^2).

Belief updating (post-hire learning):

    \tilde{\sigma}_{ij,t+1}
    = (1 - v_x) * \tilde{\sigma}_{ij, t = interview}
      + v_x * \sigma_j,

    v_j,t = [exp_{j,t} * K_1] / [1 + (exp_{j,t} - 1) * K_1],
    K_1 = delta_interview^2 / (delta_interview^2 + delta_eps^2).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def normalize_profit_signal(p: float, method: str = "tanh", scale: float = 1.0) -> float:
    """
    Normalize profit signals before feeding them into wage rules.
    """
    p = float(p)
    if method == "tanh":
        return float(np.tanh(p / max(scale, 1e-6)))
    if method == "clip":
        return float(np.clip(p, -scale, scale))
    return p


def generate_profit_array(
    exp_tm1: np.ndarray,
    sigma_j: np.ndarray,
    employed_by: np.ndarray,
    g0: float = 0.1,
    g1: float = 0.5,
    theta: float = 0.05,
    delta_eps_sq: float = 0.1,
    f_type: str = "diminishing",
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """
    Step 3 profit draw for every worker (array output matches sigma_j size).
    Unemployed workers receive zero profit by construction.
    """
    if rng is None:
        rng = np.random

    exp_tm1 = np.asarray(exp_tm1, dtype=float).reshape(-1)
    sigma_j = np.asarray(sigma_j, dtype=float).reshape(-1)
    employed_by = np.asarray(employed_by, dtype=int).reshape(-1)
    profits = np.zeros_like(sigma_j, dtype=float)

    employed_mask = employed_by >= 0
    if not employed_mask.any():
        return profits.astype(np.float32)

    core = exp_tm1 + (g0 + g1 * sigma_j) * employed_mask * np.exp(-theta * exp_tm1)
    if f_type == "diminishing":
        val = core / (1.0 + 0.1 * core)
    else:
        raise ValueError(f"Unknown f_type '{f_type}'. Expected 'diminishing'.")

    eps_std = float(np.sqrt(delta_eps_sq))
    eps = rng.normal(0.0, eps_std, size=val.shape)
    profits[employed_mask] = val[employed_mask] + eps[employed_mask]
    return profits.astype(np.float32)


def update_sigma_tilde_from_profit(
    sigma_tilde_interview: float,
    sigma_true: float,
    sigma_hat: float,
    exp_t: float,
    delta_interview_sq: float,
    delta_eps_sq: float,
) -> Tuple[float, float]:
    """
    Private-belief update using interview signal and realized performance.
    """
    sigma_tilde_interview = float(sigma_tilde_interview)
    sigma_true = float(sigma_true)
    sigma_hat = float(sigma_hat)
    exp_t = max(0.0, float(exp_t))
    delta_interview_sq = float(delta_interview_sq)
    delta_eps_sq = float(delta_eps_sq)

    denom = delta_interview_sq + delta_eps_sq
    K1 = delta_interview_sq / denom if denom > 0.0 else 0.0

    if K1 > 0.0:
        vx_denom = 1.0 + (exp_t - 1.0) * K1
        vx = (exp_t * K1) / vx_denom if abs(vx_denom) >= 1e-12 else 0.3
    else:
        vx = 0.3

    new_belief = (1.0 - vx) * sigma_tilde_interview + vx * sigma_true

    # Enforce |sigma_tilde - sigma_true| < |sigma_hat - sigma_true|
    d_hat = abs(sigma_hat - sigma_true)
    d_tilde = abs(new_belief - sigma_true)
    eps = 1e-6
    if d_hat <= eps:
        new_belief = sigma_true
    elif d_tilde >= d_hat:
        direction = 1.0 if (sigma_hat - sigma_true) >= 0 else -1.0
        new_belief = sigma_true + direction * max(d_hat - eps, 0.0)

    return float(new_belief), float(vx)


def update_sigma_hat(sigma_hat: float, sigma_tilde: float, weight: float = 0.5) -> float:
    """
    Public-signal adjustment when a worker is hired elsewhere.
    """
    weight = float(np.clip(weight, 0.0, 1.0))
    return float((1.0 - weight) * sigma_hat + weight * sigma_tilde)


def update_beliefs_and_experience(
    sigma_tilde: np.ndarray,
    sigma_hat: np.ndarray,
    sigma_true: np.ndarray,
    employed_by: np.ndarray,
    experience: np.ndarray,
    profits: np.ndarray,
    interview_vars: np.ndarray,
    delta_eps_sq: float,
    g0: float = 0.1,
    g1: float = 0.5,
    theta: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Step 3 belief + experience update (array outputs match sigma_j size).

    - If worker j is employed by firm i, update firm i's sigma_tilde_{ij}.
    - Public signal sigma_hat_j moves toward the hiring firm's belief.
    - Experience accumulates only for employed workers.
    - Returns updated (sigma_tilde, sigma_hat, experience, vx_per_worker).
    """
    sigma_tilde = np.asarray(sigma_tilde, dtype=float)
    sigma_hat = np.asarray(sigma_hat, dtype=float)
    sigma_true = np.asarray(sigma_true, dtype=float).reshape(-1)
    employed_by = np.asarray(employed_by, dtype=int).reshape(-1)
    experience = np.asarray(experience, dtype=float).reshape(-1)
    profits = np.asarray(profits, dtype=float).reshape(-1)
    interview_vars = np.asarray(interview_vars, dtype=float)

    num_workers = sigma_true.shape[0]
    vx_per_worker = np.zeros(num_workers, dtype=float)

    for j in range(num_workers):
        firm_idx = int(employed_by[j])
        if firm_idx < 0:
            continue
        delta_interview_sq = interview_vars[firm_idx, j]
        new_belief, vx = update_sigma_tilde_from_profit(
            sigma_tilde_interview=float(sigma_tilde[firm_idx, j]),
            sigma_true=float(sigma_true[j]),
            sigma_hat=float(sigma_hat[j]),
            exp_t=float(experience[j]),
            delta_interview_sq=float(delta_interview_sq),
            delta_eps_sq=float(delta_eps_sq),
        )
        sigma_tilde[firm_idx, j] = new_belief
        sigma_hat[j] = update_sigma_hat(sigma_hat[j], new_belief)
        exp_increment = (g0 + g1 * sigma_true[j]) * np.exp(-theta * experience[j])
        experience[j] = float(experience[j] + exp_increment)
        vx_per_worker[j] = vx

    return sigma_tilde.astype(np.float32), sigma_hat.astype(np.float32), experience.astype(np.float32), vx_per_worker.astype(np.float32)
