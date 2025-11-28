

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


def _profit_core(
    exp_tm1: float,
    sigma_j: float,
    employed_tm1: bool,
    g0: float,
    g1: float,
    theta: float,
    f_type: str,
) -> float:
    """
    Deterministic component of profit without noise eps.
    """
    exp_tm1 = float(exp_tm1)
    sigma_j = float(sigma_j)
    employed_indicator = 1.0 if employed_tm1 else 0.0
    core = exp_tm1 + (g0 + g1 * sigma_j) * employed_indicator * np.exp(-theta * exp_tm1)

    if f_type == "linear":
        val = core
    elif f_type == "log":
        val = np.log1p(max(core, -0.999999))
    elif f_type == "diminishing":
        val = core / (1.0 + 0.1 * core)
    else:
        raise ValueError(f"Unknown f_type '{f_type}'. Expected 'linear', 'log', or 'diminishing'.")
    return float(val)


def normalize_profit_signal(p: float, method: str = "tanh", scale: float = 1.0) -> float:
    """
    Map raw profit to a bounded/normalized signal on ability's scale.

    Args:
        p: raw profit
        method: "tanh" (default) or "log"
        scale: scaling factor for tanh, ignored for log
    """
    if method == "tanh":
        s = np.tanh(p / max(scale, 1e-8))
    elif method == "log":
        s = np.log1p(max(p, -0.999999))
    else:
        raise ValueError(f"Unknown profit normalization '{method}'")
    return float(s)


def _invert_profit_to_sigma_estimate(
    p_ijt: float,
    exp_t: float,
    g0: float,
    g1: float,
    theta: float,
    f_type: str,
) -> Optional[float]:
    """
    Attempt to back out an estimate of sigma_true from observed profit, given
    the profit function form f_type. This is a heuristic inversion; if it fails
    (e.g., division by zero, invalid domain), return None.
    """
    try:
        # Invert f(core) to recover core estimate
        if f_type == "linear":
            core_est = p_ijt
        elif f_type == "log":
            core_est = np.expm1(p_ijt)
        elif f_type == "diminishing":
            denom = 1.0 - 0.1 * p_ijt
            if abs(denom) < 1e-8:
                return None
            core_est = p_ijt / denom
        else:
            return None

        if abs(g1) < 1e-8:
            return None

        log_term = np.log(core_est - exp_t) + theta * exp_t
        term = np.exp(log_term)

        sigma_est = (term - g0) / g1
        return float(sigma_est)
    except Exception:
        return None


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

    val = _profit_core(
        exp_tm1=exp_tm1,
        sigma_j=sigma_j,
        employed_tm1=employed_tm1,
        g0=g0,
        g1=g1,
        theta=theta,
        f_type=f_type,
    )

    # Add normally distributed noise eps_{ij,t}
    eps_std = float(np.sqrt(delta_eps_sq))
    eps = rng.normal(0.0, eps_std)

    return float(val + eps)


def update_belief_from_profit(
    sigma_tilde_interview: float,
    sigma_true: float,
    exp_t: float,
    delta_interview_sq: float,
    delta_eps_sq: float,
) -> Tuple[float, float]:
    """
    Update a firm's private belief using interview signal and true ability.

    Implements
        \tilde{\sigma}_{ij,t+1}
        = (1 - v_x) * \tilde{\sigma}_{ij, \text{interview}} + v_x * \sigma_j,
    where
        v_x = [exp_{j,t} * K_1] / [1 + (exp_{j,t} - 1) * K_1],
        K_1 = delta_interview^2 / (delta_interview^2 + delta_eps^2).

    Returns:
        (new_belief, v_x)
    """
    sigma_tilde_interview = float(sigma_tilde_interview)
    sigma_true = float(sigma_true)
    exp_t = max(0.0, float(exp_t))
    delta_interview_sq = float(delta_interview_sq)
    delta_eps_sq = float(delta_eps_sq)

    denom = delta_interview_sq + delta_eps_sq
    if denom > 0.0:
        K1 = delta_interview_sq / denom
    else:
        K1 = 0.0

    if K1 > 0.0:
        vx_denom = 1.0 + (exp_t - 1.0) * K1
        if abs(vx_denom) < 1e-12:
            vx = 0.3
        else:
            vx = (exp_t * K1) / vx_denom
    else:
        vx = 0.3

    new_belief = (1.0 - vx) * sigma_tilde_interview + vx * sigma_true
    return float(new_belief), float(vx)
