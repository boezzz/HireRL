

"""
Post-hiring wage adjustment and firing decision rules.

Step 5 (Wage adjustment):
    w_{j,t}
    = (1 - v_x) g(\tilde{\sigma}_{ij, t = \text{interview}})
      + v_x \psi \, p_{ij,t-1},

    v_x = \frac{\exp_{j,t} K_1}{1 + (\exp_{j,t} - 1) K_1},
    K_1 = \frac{\delta_{\text{interview}}^2}{\delta_{\text{interview}}^2 + \delta_{\varepsilon}^2},
    \psi \in (0, 1).

Step 6 (Firing rule):
    \text{fire}_{ij,t} = 1\{ p_{ij,t} - w_{ij,t} < - c_{i,t}^{\text{fire}} \},
    \quad C_{\text{fire}} \gg C_{\text{interview}}.

This module provides helper functions to compute post-hire wages and
firing decisions, consistent with the profit generation and belief
updating logic implemented elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


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

    wage_t: float
    vx: float
    signal_component: float
    profit_component: float


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
    tilde_sigma_interview: float,
    p_ij_tm1: float,
    exp_t: float,
    delta_interview_sq: float,
    delta_eps_sq: float,
    psi: float = 0.5,
    g: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> WageAdjustmentResult:
    """
    Compute the post-hiring wage w_{j,t} based on the wage rule

        w_{j,t}
        = (1 - v_x) g(\tilde{\sigma}_{ij, t = \text{interview}})
          + v_x \psi \, p_{ij,t-1},

    where

        v_x = [exp_{j,t} K_1] / [1 + (exp_{j,t} - 1) K_1],
        K_1 = delta_interview^2 / (delta_interview^2 + delta_eps^2).

    Args
    ----
    tilde_sigma_interview : float
        Firm i's private interview signal \tilde{\sigma}_{ij, t = interview}.
    p_ij_tm1 : float
        Realized profit p_{ij,t-1} from the previous period.
    exp_t : float
        Experience level exp_{j,t} used in the v_x formula.
    delta_interview_sq : float
        Interview noise variance \delta_{\text{interview}}^2.
    delta_eps_sq : float
        Profit noise variance \delta_{\varepsilon}^2.
    psi : float, optional
        Share of profit passed through to wages, \psi \in (0, 1).
    g : callable, optional
        Function g(·) applied to the interview signal. If None, use
        the bounded, diminishing-returns default g defined above.

    Returns
    -------
    WageAdjustmentResult
        Contains the new wage, the v_x used, and the signal/profit
        components of the wage rule.
    """
    tilde_sigma_interview = float(tilde_sigma_interview)
    p_ij_tm1 = float(p_ij_tm1)
    exp_t = max(0.0, float(exp_t))
    delta_interview_sq = float(delta_interview_sq)
    delta_eps_sq = float(delta_eps_sq)
    psi = float(psi)

    if not (0.0 < psi < 1.0):
        raise ValueError(f"psi must be in (0, 1), got {psi}.")

    # Compute K_1 and v_x as in the belief-updating formula
    denom = delta_interview_sq + delta_eps_sq
    if denom > 0.0:
        K1 = delta_interview_sq / denom
    else:
        K1 = 0.0

    if K1 > 0.0:
        vx = (exp_t * K1) / (1.0 + (exp_t - 1.0) * K1)
    else:
        vx = 0.0

    # Choose g(·)
    if g is None:
        g = default_g_bounded

    # Apply g to the (scalar) interview signal
    g_input = np.array([tilde_sigma_interview], dtype=float)
    g_val = float(g(g_input)[0])

    signal_component = (1.0 - vx) * g_val
    profit_component = vx * psi * p_ij_tm1
    wage_t = signal_component + profit_component

    return WageAdjustmentResult(
        wage_t=float(wage_t),
        vx=float(vx),
        signal_component=float(signal_component),
        profit_component=float(profit_component),
    )


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
    p_ijt: float,
    w_ijt: float,
    c_fire_t: float,
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
    p_ijt = float(p_ijt)
    w_ijt = float(w_ijt)
    c_fire_t = float(c_fire_t)

    margin = p_ijt - w_ijt
    threshold = -c_fire_t
    fire = margin < threshold

    return FiringDecisionResult(
        fire=fire,
        margin=float(margin),
        threshold=float(threshold),
    )