

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
    c_fire_t = 6.0 * wage  # deterministic: 6x current wage
    margin = profit - wage
    threshold = -c_fire_t
    fire = margin < threshold

    return FiringDecisionResult(
        fire=fire,
        margin=float(margin),
        threshold=float(threshold),
    )


def _demo_wage_trajectories_two_firms():
    import matplotlib.pyplot as plt
    from pathlib import Path
    from generated_profit2 import (
        generate_profit_array,
        update_sigma_tilde_from_profit,
        update_sigma_hat_accepted,
        update_sigma_no_offer,
    )

    rng = np.random.RandomState(0)
    num_workers = 10
    num_firms = 2
    periods = 10

    sigma_true = rng.normal(0.0, 1.0, size=num_workers).astype(np.float32)
    sigma_hat = sigma_true + rng.normal(0.0, 0.3, size=num_workers).astype(np.float32)
    sigma_tilde = np.stack(
        [
            sigma_hat + rng.normal(0.0, 0.2, size=num_workers),
            sigma_hat + rng.normal(0.0, 0.2, size=num_workers),
        ]
    ).astype(np.float32)

    employed_by = np.array([0] * 5 + [1] * 5, dtype=int)  # firm0: workers 0-4; firm1: 5-9
    experience = np.zeros(num_workers, dtype=np.float32)

    wages_hist = [np.zeros((num_firms, num_workers), dtype=np.float32)]  # t0 placeholder
    sigma_hat_hist = [sigma_hat.copy()]
    sigma_tilde_hist = [sigma_tilde.copy()]
    profits_hist = []

    delta_interview_sq = 0.4
    delta_eps_sq = 0.1
    psi = 0.5
    firm_multiplier = [1.0, 1.2]

    for _ in range(periods):
        profits = np.zeros((num_firms, num_workers), dtype=np.float32)
        for firm_id in range(num_firms):
            mask = (employed_by == firm_id).astype(np.int8)
            profits[firm_id] = generate_profit_array(
                exp_tm1=experience,
                sigma_true=sigma_true,
                employed_by=mask,
                g0=0.1,
                g1=0.5,
                theta=0.05,
                delta_eps_sq=delta_eps_sq,
                rng=rng,
            )
        profits_hist.append(profits.copy())

        wages_t = np.zeros((num_firms, num_workers), dtype=np.float32)
        sigma_hat_next = sigma_hat.copy()
        sigma_tilde_next = sigma_tilde.copy()

        for worker_id in range(num_workers):
            firm_id = employed_by[worker_id]
            if firm_id < 0:
                sigma_hat_no_offer, sigma_tilde_no_offer = update_sigma_no_offer(sigma_hat[worker_id])
                sigma_hat_next[worker_id] = float(sigma_hat_no_offer)
                sigma_tilde_next[:, worker_id] = float(sigma_tilde_no_offer)
                continue

            exp_t = max(float(experience[worker_id]), 1.0)
            profit_prev = float(profits[firm_id, worker_id])

            wage = adjust_wage_post_hire(
                sigma_tilde_interview=float(sigma_tilde[firm_id, worker_id]),
                p_ij_tm1=profit_prev,
                psi=psi,
                exp_t=exp_t,
                delta_interview_sq=delta_interview_sq,
                delta_eps_sq=delta_eps_sq,
                company_type=firm_multiplier[firm_id],
                employed_by=1,
                g_fn=default_g_bounded,
                sigma_true=float(sigma_true[worker_id]),
            )
            wages_t[firm_id, worker_id] = float(wage)

            sigma_tilde_new, sigma_update, _ = update_sigma_tilde_from_profit(
                sigma_tilde_interview=float(sigma_tilde[firm_id, worker_id]),
                sigma_true=float(sigma_true[worker_id]),
                exp_t=exp_t,
                delta_interview_sq=delta_interview_sq,
                delta_eps_sq=delta_eps_sq,
            )
            sigma_tilde_next[firm_id, worker_id] = float(sigma_tilde_new)
            sigma_hat_next[worker_id] = float(
                update_sigma_hat_accepted(
                    sigma_tilde=float(sigma_tilde[firm_id, worker_id]),
                    sigma_update=float(sigma_update),
                )
            )

        wages_hist.append(wages_t.copy())
        sigma_hat_hist.append(sigma_hat_next.copy())
        sigma_tilde_hist.append(sigma_tilde_next.copy())

        experience = experience + (employed_by >= 0).astype(np.float32)
        sigma_hat = sigma_hat_next
        sigma_tilde = sigma_tilde_next

    # Stack histories
    profits_hist = np.stack(profits_hist, axis=0)  # (periods, firms, workers)
    wages_hist = np.stack(wages_hist, axis=0)  # (periods+1, firms, workers)
    sigma_hat_hist = np.stack(sigma_hat_hist, axis=0)  # (periods+1, workers)
    sigma_tilde_hist = np.stack(sigma_tilde_hist, axis=0)  # (periods+1, firms, workers)

    # Firm-level wage trajectories
    fig, axes = plt.subplots(num_firms, 1, figsize=(10, 6), sharex=True)
    for firm_id in range(num_firms):
        ax = axes[firm_id] if num_firms > 1 else axes
        for worker_id in range(num_workers):
            ax.plot(wages_hist[:, firm_id, worker_id], label=f"w{worker_id}")
        ax.set_title(f"Firm {firm_id} wages over time")
        ax.set_ylabel("wage")
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        if firm_id == num_firms - 1:
            ax.set_xlabel("timestep")
        ax.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    out_path = Path(__file__).with_name("post_hiring_wage_demo.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[demo] Saved firm wage trajectories to {out_path}")

    # Per-worker two-panel plots: profit & wage (left) and sigmas (right)
    per_worker_dir = Path(__file__).with_name("post_hiring_worker_profit_wage_sigma")
    per_worker_dir.mkdir(exist_ok=True)
    for worker_id in range(num_workers):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        # Left: profit & wage per firm (align wages to periods using wages_hist[1:])
        axes[0].plot(profits_hist[:, 0, worker_id], label="profit firm0", color="tab:blue")
        axes[0].plot(profits_hist[:, 1, worker_id], label="profit firm1", color="tab:orange")
        axes[0].plot(wages_hist[1:, 0, worker_id], label="wage firm0", color="tab:blue", linestyle="--")
        axes[0].plot(wages_hist[1:, 1, worker_id], label="wage firm1", color="tab:orange", linestyle="--")
        axes[0].axhline(0.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6)
        axes[0].set_title(f"Worker {worker_id}: profit & wage")
        axes[0].set_xlabel("timestep")
        axes[0].set_ylabel("value")
        axes[0].legend(fontsize=8)

        # Right: sigmas
        axes[1].plot(sigma_hat_hist[:, worker_id], label="sigma_hat", color="tab:purple")
        axes[1].plot(sigma_tilde_hist[:, 0, worker_id], label="sigma_tilde firm0", color="tab:blue")
        axes[1].plot(sigma_tilde_hist[:, 1, worker_id], label="sigma_tilde firm1", color="tab:orange")
        axes[1].axhline(sigma_true[worker_id], color="black", linestyle="--", linewidth=1.0, label="sigma_true")
        axes[1].set_title(f"Worker {worker_id}: sigma paths")
        axes[1].set_xlabel("timestep")
        axes[1].set_ylabel("sigma")
        axes[1].legend(fontsize=8)

        plt.tight_layout()
        out_path_worker = per_worker_dir / f"worker_{worker_id}_profit_wage_sigma.png"
        plt.savefig(out_path_worker)
        plt.close()

    print(f"[demo] Saved per-worker profit/wage/sigma plots to {per_worker_dir}")


if __name__ == "__main__":
    _demo_wage_trajectories_two_firms()
