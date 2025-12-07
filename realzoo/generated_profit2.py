

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

def update_experience(
    exp_t: np.ndarray,
    sigma_true: np.ndarray, # of each worker
    employed_by: np.ndarray,
    g0: float = 0.1,
    g1: float = 0.5,
    theta: float = 0.05
):
    return exp_t + (g0 + g1 * sigma_true) * employed_by * np.exp(-theta * exp_t)


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
    lr: float = 0.3
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
    sigma_update = (sigma_tilde_t1 - sigma_tilde_interview) * lr
    sigma_tilde_t1 = sigma_tilde_interview + sigma_update

    return sigma_tilde_t1, sigma_update, vx


def update_sigma_hat_accepted(sigma_tilde: np.ndarray, #?
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

    sigma_hat_t1 = sigma_tilde + 0.5 * sigma_update


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


# ----------------------------------------------------------------------
# Inline test / demo: two firms, ten workers, 20 timesteps
# ----------------------------------------------------------------------

def normal_between_0_1():
    x = np.random.normal(0.5, 1)
    while x < 0 or x > 1:   # reject negative values
        x = np.random.normal(0.5, 1)
    return x


def _demo_two_firms_twenty_steps():
    import matplotlib.pyplot as plt
    from pathlib import Path

    rng = np.random.RandomState(0)
    num_workers = 10
    num_firms = 2
    timesteps = 20

    sigma_true = np.array([normal_between_0_1() for _ in range(num_workers)]).astype(np.float32)
    sigma_hat = sigma_true + rng.normal(0.0, 0.3, size=num_workers).astype(np.float32)
    sigma_tilde = np.stack(
        [
            sigma_hat,
            sigma_hat,
        ]
    ).astype(np.float32)

    employed_by = np.array([0] * 5 + [1] * 3 + [-1] * 2, dtype=int)
    experience = np.ones(num_workers, dtype=np.float32)

    history_sigma_hat = [sigma_hat.copy()]
    history_sigma_tilde_f0 = [sigma_tilde[0].copy()]
    history_sigma_tilde_f1 = [sigma_tilde[1].copy()]
    history_profit_f0 = []
    history_profit_f1 = []

    delta_interview_sq = 0.4
    delta_eps_sq = 0.1

    for t in range(1, timesteps + 1):
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

            # this only adds to expierence if they're employed, and each worker only works at one company, so this is fine
            experience = update_experience(experience, sigma_true, mask, 0.1, 0.5, 0.05)

        history_profit_f0.append(float(np.mean(profits[0][employed_by == 0])) if np.any(employed_by == 0) else 0.0)
        history_profit_f1.append(float(np.mean(profits[1][employed_by == 1])) if np.any(employed_by == 1) else 0.0)

        sigma_hat_next = sigma_hat.copy()
        sigma_tilde_next = sigma_tilde.copy()

        for worker_id in range(num_workers):
            firm_id = employed_by[worker_id]
            if firm_id < 0:
                sigma_hat_no_offer, sigma_tilde_no_offer = update_sigma_no_offer(sigma_hat[worker_id])
                sigma_hat_next[worker_id] = float(sigma_hat_no_offer)
                sigma_tilde_next[:, worker_id] = float(sigma_tilde_no_offer)
                continue

            exp_t = experience[worker_id]
            if exp_t < 1:
                raise ValueError(f"Bad exper. value, less than 1: {exp_t}")
            sigma_tilde_new, sigma_update, vx = update_sigma_tilde_from_profit(
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

        sigma_hat = sigma_hat_next
        sigma_tilde = sigma_tilde_next

        history_sigma_hat.append(sigma_hat.copy())
        history_sigma_tilde_f0.append(sigma_tilde[0].copy())
        history_sigma_tilde_f1.append(sigma_tilde[1].copy())

    history_sigma_hat = np.vstack(history_sigma_hat)
    history_sigma_tilde_f0 = np.vstack(history_sigma_tilde_f0)
    history_sigma_tilde_f1 = np.vstack(history_sigma_tilde_f1)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    axes[0].set_title("Sigma_hat trajectories (10 workers)")
    for worker_id in range(num_workers):
        axes[0].plot(history_sigma_hat[:, worker_id], label=f"worker {worker_id}")
    axes[0].axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)
    axes[0].set_ylabel("sigma_hat")

    axes[1].set_title("Firm 0 sigma_tilde trajectories")
    for worker_id in range(num_workers):
        axes[1].plot(history_sigma_tilde_f0[:, worker_id], label=f"w{worker_id}")
    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[1].set_ylabel("sigma_tilde firm0")

    axes[2].set_title("Average profit per firm per timestep")
    axes[2].plot(history_profit_f0, label="firm 0 avg profit")
    axes[2].plot(history_profit_f1, label="firm 1 avg profit")
    axes[2].axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[2].set_xlabel("timestep")
    axes[2].set_ylabel("profit")
    axes[2].legend()

    axes[3].set_title("Sigma true")
    for worker_id in range(num_workers):
        line = [sigma_true[worker_id]] * len(history_sigma_hat)
        axes[3].plot(line, label=f"worker {worker_id}")
    axes[3].axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    axes[3].legend(loc="upper right", ncol=2, fontsize=8)
    axes[3].set_ylabel("sigma")

    plt.tight_layout()
    out_path = Path(__file__).with_name("generated_profit2_two_firms_demo.png")
    plt.savefig(out_path)
    print(f"[demo] Saved trajectories plot to {out_path}")

    # Per-worker trajectories: sigma_hat vs sigma_tilde (firm 0/1) with sigma_true line
    per_worker_dir = Path(__file__).with_name("generated_profit2_worker_plots")
    per_worker_dir.mkdir(exist_ok=True)
    for worker_id in range(num_workers):
        plt.figure(figsize=(8, 4))
        plt.plot(history_sigma_hat[:, worker_id], label="sigma_hat", color="tab:blue")
        plt.plot(history_sigma_tilde_f0[:, worker_id], label="sigma_tilde firm0", color="tab:orange")
        plt.plot(history_sigma_tilde_f1[:, worker_id], label="sigma_tilde firm1", color="tab:green")
        plt.axhline(sigma_true[worker_id], color="black", linestyle="--", linewidth=1.0, label="sigma_true")
        plt.title(f"Worker {worker_id}: sigma trajectories")
        plt.xlabel("timestep")
        plt.ylabel("sigma")
        plt.legend()
        plt.tight_layout()
        out_path_worker = per_worker_dir / f"worker_{worker_id}_sigma.png"
        plt.savefig(out_path_worker)
        plt.close()
    print(f"[demo] Saved per-worker plots to {per_worker_dir}")

if __name__ == "__main__":
    _demo_two_firms_twenty_steps()
