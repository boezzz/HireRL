

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
      + v_x * p_{ij,t},

    v_x = [exp_{j,t} * K_1] / [1 + (exp_{j,t} - 1) * K_1],
    K_1 = delta_interview^2 / (delta_interview^2 + delta_eps^2).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def generate_profit(
    exp_tm1: float,
    sigma_j: float,
    employed_tm1: bool,
    g0: float = 0.1,
    g1: float = 0.5,
    theta: float = 0.05,
    delta_eps_sq: float = 0.1,
    f_type: str = "linear",
    rng: Optional[np.random.RandomState] = None,
) -> float:
    """
    Generate a realized profit p_{ij,t} given true ability and experience.

    This implements

        p_{ij,t}
        = f [
            exp_{j,t-1}
            + (g_0 + g_1 * sigma_j)
              * 1{ j employed at t-1 }
              * exp(-theta * exp_{j,t-1})
          ] + eps_{ij,t},

        eps_{ij,t} ~ N(0, delta_eps^2).

    Args:
        exp_tm1: exp_{j,t-1}, on-the-job experience at t-1.
        sigma_j: true ability sigma_j.
        employed_tm1: indicator 1{ j employed at t-1 }.
        g0, g1: experience-growth parameters.
        theta: decay parameter in exp(-theta * exp_tm1).
        delta_eps_sq: variance of profit shock eps_{ij,t}.
        f_type: functional form for f(·):
            - "linear": f(x) = x
            - "log": f(x) = log(1 + x)
            - "diminishing": f(x) = x / (1 + 0.1 x)
        rng: optional numpy RandomState; if None, use np.random.

    Returns:
        A scalar realized profit p_{ij,t}.
    """
    if rng is None:
        rng = np.random

    exp_tm1 = float(exp_tm1)
    sigma_j = float(sigma_j)

    # Deterministic component inside f[·]
    employed_indicator = 1.0 if employed_tm1 else 0.0
    core = exp_tm1 + (g0 + g1 * sigma_j) * employed_indicator * np.exp(-theta * exp_tm1)

    # Apply chosen f(·)
    if f_type == "linear":
        val = core
    elif f_type == "log":
        # log(1 + x) to avoid issues at x close to 0
        val = np.log1p(max(core, -0.999999))
    elif f_type == "diminishing":
        val = core / (1.0 + 0.1 * core)
    else:
        raise ValueError(f"Unknown f_type '{f_type}'. Expected 'linear', 'log', or 'diminishing'.")

    # Add normally distributed noise eps_{ij,t}
    eps_std = float(np.sqrt(delta_eps_sq))
    eps = rng.normal(0.0, eps_std)

    return float(val + eps)


def update_belief_from_profit(
    tilde_sigma_interview: float,
    p_ijt: float,
    exp_t: float,
    delta_interview_sq: float,
    delta_eps_sq: float,
) -> Tuple[float, float]:
    """
    Update a firm's private belief about a worker using realized profit.

    This implements

        \tilde{\sigma}_{ij,t+1}
        = (1 - v_x) * \tilde{\sigma}_{ij, t = interview}
          + v_x * p_{ij,t},

    where

        v_x = [exp_{j,t} * K_1] / [1 + (exp_{j,t} - 1) * K_1],
        K_1 = delta_interview^2 / (delta_interview^2 + delta_eps^2).

    Args:
        tilde_sigma_interview: firm's private interview signal
            \tilde{\sigma}_{ij, t = interview}.
        p_ijt: realized profit p_{ij,t} observed by the firm.
        exp_t: experience exp_{j,t} used in the v_x formula.
        delta_interview_sq: interview noise variance delta_interview^2.
        delta_eps_sq: profit noise variance delta_eps^2.

    Returns:
        A tuple (new_belief, v_x), where
            new_belief = \tilde{\sigma}_{ij,t+1},
            v_x is the weight placed on profit in the convex combination.
    """
    tilde_sigma_interview = float(tilde_sigma_interview)
    p_ijt = float(p_ijt)
    exp_t = max(0.0, float(exp_t))
    delta_interview_sq = float(delta_interview_sq)
    delta_eps_sq = float(delta_eps_sq)

    # Compute K_1 and v_x as in the paper
    denom = delta_interview_sq + delta_eps_sq
    if denom > 0.0:
        K1 = delta_interview_sq / denom
    else:
        K1 = 0.0

    if K1 > 0.0:
        vx = (exp_t * K1) / (1.0 + (exp_t - 1.0) * K1)
    else:
        vx = 0.0

    # Convex combination of interview belief and realized profit
    new_belief = (1.0 - vx) * tilde_sigma_interview + vx * p_ijt

    return float(new_belief), float(vx)