

"""
Post-hiring wage adjustment and firing decision rules.
Step 5 (Wage adjustment):
Step 6 (Firing rule):
    \text{fire}_{ij,t} = 1\{ p_{ij,t} - w_{ij,t} < - c_{i,t}^{\text{fire}} \},
    \quad C_{\text{fire}} \gg C_{\text{interview}}.

This module provides helper functions to compute post-hire wages and
firing decisions, consistent with the profit generation and belief
updating logic implemented elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from generated_profit2 import generate_profit_array, update_sigma_tilde_from_profit


@dataclass
class WageAdjustmentResult:
    """
    Result of post-hiring wage adjustment for a single firm–worker pair.

    Attributes
    ----------
    wage_t : float
        The adjusted wage w_{j,t}.
    vx : float
        The weight v_x placed on past realized profit.
    signal_component : float
        (1 - v_x) g(\tilde{\sigma}_{ij, \text{interview}}).
    profit_component : float
        v_x \psi p_{ij,t-1}.
    """

    wage: np.ndarray
    signal_component: np.ndarray
    profit_component: np.ndarray


def default_g_bounded(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Default g(·): bounded, increasing, with diminishing marginal returns.

    We use
        g(x) = 0.5 * (1 + tanh(alpha * x)) \in (0, 1),

    so higher interview signals raise wages but the effect approaches
    an upper bound, capturing the idea of a wage ceiling and decreasing
    marginal returns to ability.
    """
    return 0.5 * (1.0 + np.tanh(alpha * x))


def adjust_wage_post_hire(
    sigma_tilde_interview: np.ndarray,
    p_ij_tm1: Optional[np.ndarray] = None,
    psi: float = 0.5,
    exp_t: Optional[np.ndarray] = None,
    delta_interview_sq: Optional[np.ndarray] = None,
    delta_eps_sq: Optional[np.ndarray] = None,
    company_type: float = 1.0,
    employed_by: Optional[np.ndarray] = None,
    exp_tm1: Optional[np.ndarray] = None,
    sigma_true: Optional[np.ndarray] = None,
    g0: float = 0.1,
    g1: float = 0.5,
    theta: float = 0.05,
    rng: Optional[np.random.RandomState] = None,
    g_fn: Callable[[np.ndarray], np.ndarray] = default_g_bounded,
    **unused,
) -> WageAdjustmentResult:
    """
    Compute the post-hiring wage using updated beliefs:

        w_{j,t} = company_time * [(1-v_{j,t}) g(tilde_σ_{ij,t}) + v_{j,t} ψ p_{ij,t-1}]

    """
    if sigma_true is None:
        raise ValueError("sigma_true must be provided for belief updating")

    mask = None
    if employed_by is not None:
        mask = np.asarray(employed_by, dtype=float)

    profit_prev = p_ij_tm1
    if profit_prev is None:
        if exp_tm1 is None or sigma_true is None or employed_by is None:
            raise ValueError(
                "p_ij_tm1 missing: provide exp_tm1, sigma_true, employed_by to generate profit."
            )
        profit_prev = generate_profit_array(
            exp_tm1=np.asarray(exp_tm1, dtype=float),
            sigma_true=np.asarray(sigma_true, dtype=float),
            employed_by=np.asarray(employed_by, dtype=float),
            g0=float(g0),
            g1=float(g1),
            theta=float(theta),
            delta_eps_sq=float(np.asarray(delta_eps_sq, dtype=float).mean()),
            rng=rng,
        )

    profit_prev = np.asarray(profit_prev, dtype=float)

    _, _, vx = update_sigma_tilde_from_profit(
        sigma_tilde_interview=sigma_tilde_interview,
        sigma_true=sigma_true,
        exp_t=exp_t,
        delta_interview_sq=delta_interview_sq,
        delta_eps_sq=delta_eps_sq,
    )

    g_val = g_fn(sigma_tilde_interview)
    if mask is not None:
        g_val = g_val * mask
        profit_prev = profit_prev * mask

    signal_component = (1.0 - vx) * g_val
    profit_component = vx * psi * profit_prev
    wage = (signal_component + profit_component) * company_type

    return wage


@dataclass
class FiringDecisionResult:
    """
    Result of the firing decision for a single firm–worker pair.

    Attributes
    ----------
    fire : bool
        True if the firm fires the worker in period t.
    margin : float
        Net surplus p_{ij,t} - w_{ij,t}.
    threshold : float
        Firing threshold -c_{i,t}^{fire} used in the rule.
    """

    fire: bool
    margin: float
    threshold: float


def firing_decision(
    profit: np.ndarray,
    wage: np.ndarray,
    c_fire_t: np.ndarray
) -> FiringDecisionResult:
    """
    Implement the deterministic firing rule

        fire_{ij,t} = 1{ p_{ij,t} - w_{ij,t} < - c_{i,t}^{fire} }.

    That is, the firm fires the worker if the net surplus p_{ij,t} - w_{ij,t}
    is sufficiently negative, by more than the firing cost c_{i,t}^{fire}.

    Args
    ----
    p_ijt : float
        Realized profit p_{ij,t} in the current period.
    w_ijt : float
        Wage paid w_{ij,t} in the current period.
    c_fire_t : float
        Firing cost c_{i,t}^{fire}. Larger values make firing less likely.

    Returns
    -------
    FiringDecisionResult
        Contains the firing indicator, the net margin, and the threshold.
    """

    margin = profit - wage
    threshold = -c_fire_t
    fire = margin < threshold

    return FiringDecisionResult(
        fire=fire,
        margin=float(margin),
        threshold=float(threshold),
    )
