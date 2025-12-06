

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




def generate_profit_array(
    exp_tm1: np.ndarray, # expier. of each worker
    sigma_true: np.ndarray, # of each worker
    employed_by: np.ndarray, # but should be a 0,1 mask indicating whether this person is being hired by us
    g0: float = 0.1,
    g1: float = 0.5,
    theta: float = 0.05,
    delta_eps_sq: float = 0.1,
    rng: Optional[np.random.RandomState] = None,
):
    """
    Step 3 profit draw for every worker (array output matches sigma_j size).
    Unemployed workers receive zero profit by construction.
    """
    if rng is None:
        rng = np.random

    employed_mask = employed_by == 1
    profits = np.zeros(sigma_true.shape)
    if not employed_mask.any():
        return profits.astype(np.float32)

    core = exp_tm1 + (g0 + g1 * sigma_true) * employed_mask * np.exp(-theta * exp_tm1) # experience or some shit
    # f(x) = x / (1+0.1x) increasing and diminishing returns mapping (proof?)
    val = core / (1.0 + 0.1 * core)

    eps_std = float(np.sqrt(delta_eps_sq))
    profit_noise = rng.normal(0.0, eps_std, size=val.shape)
    profits[employed_mask] = val[employed_mask] + profit_noise[employed_mask]
    return profits.astype(np.float32)


def update_sigma_tilde_from_profit(
    sigma_tilde_interview: np.ndarray,
    sigma_true: np.ndarray,
    exp_t: np.ndarray,
    delta_interview_sq: np.ndarray,
    delta_eps_sq: np.ndarray,
) :
    """
    Private-belief update using interview signal and realized performance.
    """
    if exp_t <= 0:
        raise ValueError(f"experice should be larger than 0")

    denom = delta_interview_sq + delta_eps_sq
    if denom > 0.0:
        K1 = delta_interview_sq / denom
    else:
        raise ValueError(f"interview signal is 0")

    if K1 > 0.0:
        vx_denom = 1.0 + (exp_t - 1.0) * K1
        if abs(vx_denom) >= 1e-12:
            vx = (exp_t * K1) / vx_denom
        else:
            raise ValueError(f"experience is somehow 0")
    else:
        raise ValueError(f"interview signal is 0")

    sigma_tilde_t1 = (1.0 - vx) * sigma_tilde_interview + vx * sigma_true
    sigma_update = sigma_tilde_t1 - sigma_tilde_interview

    return sigma_tilde_t1, sigma_update, vx


def update_sigma_hat_accepted(sigma_true: np.ndarray, #?
                    sigma_update: np.ndarray):
    """
    this function only calls for the company who hires this guy
    the company shares sigma_hat with all other companys
    1) if a person is being interviewed at company A and accept the offer from company A,
     then for company B:
     sigma_hat should alo converge to the sigma_true, but at a larger noise than sigma_tilde and
    smaller noise than before accepting the offer
     # only call this function when company hires this guy

    """

    sigma_hat_t1 = sigma_true + 2 * sigma_update


    return sigma_hat_t1

def update_sigma_no_offer(sigma_hat):
    """
    2) if a person is being interviewed and not given any offer, (remember always accepts best offer)
    sigma_hat should act the same as before, with the same previous noise generation behavior over
    timesteps
    sigma_hat_(t = interview+1) = sigma_tilde_(t = interview+1) = sigma_hat(before interview)
     and so on so forth.
    Returns:

    """
    sigma_tilde_t1 = sigma_hat
    sigma_hat_t1 = sigma_hat

    return sigma_hat_t1, sigma_tilde_t1

