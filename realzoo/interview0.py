

"""Interview phase module: pre-hire screening and private signals.

Implements the interview/screening part of the model described in the paper:
    - Firms can invest c_{interview,ij} >= 0 in screening
    - This generates a private signal
          \tilde{σ}_{ij} = σ_j + η_{ij},   η_{ij} ~ N(0, δ^2(c_{interview,ij}))
      where
          δ^2(c) = δ0^2 * exp(-λ c)

The ScreeningMechanism class below handles:
    - Mapping interview cost -> signal noise variance δ^2(c)
    - Drawing private signals given true ability σ_j

This file is intentionally limited to the *interview phase* only.
Post-hire learning (belief updating from profits, experience accumulation,
profit functions, etc.) should live in a separate module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class ScreeningMechanism:
    """Interview/screening technology.

    This class encapsulates how interview cost translates into signal precision.

    Core paper equations:
        \tilde{σ}_{ij} = σ_j + η_{ij},   η_{ij} ~ N(0, δ^2(c_{interview,ij}))
        δ^2(c) = δ0^2 * exp(-λ c),   δ0^2 > 0, λ > 0.

    Attributes
    ----------
    delta0_sq:
        Baseline interview noise variance at zero cost (δ0^2).
    lam:
        Cost–precision decay parameter (λ). Higher λ means cost reduces
        noise variance more quickly.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(self, delta0_sq: float = 0.4, lam: float = 1.0, seed: Optional[int] = None):
        if delta0_sq <= 0:
            raise ValueError("delta0_sq must be > 0")
        if lam <= 0:
            raise ValueError("lam must be > 0")

        self.delta0_sq = float(delta0_sq)
        self.lam = float(lam)

        if seed is None:
            self._rng = np.random.mtrand._rand  # type: ignore[attr-defined]
        else:
            self._rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Public signal (sigma_hat) helpers up to interview time
    # ------------------------------------------------------------------
    @staticmethod
    def init_sigma_hat(
        sigma_true: np.ndarray,
        noise_std: float = 1.0,
        sigma_hat_min: float = -2.0,
        sigma_hat_max: float = 2.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> np.ndarray:
        """
        Initialize public signal: sigma_hat_0 = sigma_true + N(0, noise_std^2), clipped.
        """
        r = rng if rng is not None else np.random.mtrand._rand  # type: ignore[attr-defined]
        noise = r.randn(*sigma_true.shape).astype(np.float32) * noise_std
        sigma_hat_0 = sigma_true + noise
        return np.clip(sigma_hat_0, sigma_hat_min, sigma_hat_max).astype(np.float32)

    def interview_var(self, cost: float | np.ndarray) -> np.ndarray:
        """
        Cost-to-variance mapping δ^2(c) = δ0^2 * exp(-λ c).

        This small helper keeps the core `screen_worker` logic untouched while
        exposing the paper's variance schedule for downstream modules that need
        δ^2(c) explicitly (e.g., wage updates based on interview precision).
        """
        c_arr = np.asarray(cost, dtype=np.float32)
        return (self.delta0_sq * np.exp(-self.lam * c_arr)).astype(np.float32)


    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------
    def screen_worker(
        self,
        sigma_true: np.ndarray,
        interview_costs: np.ndarray,
        sigma_hat: np.ndarray,
        lr: float = 0.5
    ) -> np.ndarray:
        """Generate a private interview signal for an array of workers.

        Implements:
            \tilde{σ}_{ij} = σ_j + η_{ij},  η_{ij} ~ N(0, δ^2(c))

        Args
        ----
        sigma_true:
            True ability σ_j of the worker (can be scalar or vector np.ndarray).
        sigma_hat_0:
            Public initial signal \hat{σ}_{j,0}. Included for interface
            compatibility but not used in the current paper-consistent
            specification, where the interview signal is centered on σ_j.
        interview_costs:
            Interview cost c_{interview,ij} >= 0. If cost is 0, they were not interviewed.

        Returns
        -------
        sigma_tilde: np.ndarray
            The private interview signal \tilde{σ}_{ij} with the same shape
            as `sigma_true`.
        precision: float
            A convenient [0,1] summary of informativeness:
                precision = 1 - δ^2(c) / δ0^2
            where 0 means "no extra information" (cost=0) and 1 means
            "maximal precision" (cost → ∞).
        """
        if sigma_hat is None:
            raise ValueError("sigma_hat_0 must be provided to align sigma_tilde with public signal.")

        c_arr = np.asarray(interview_costs, dtype=float).reshape(-1)
        sigma_hat = np.asarray(sigma_hat, dtype=np.float32).reshape(-1)
        sigma_true_arr = np.asarray(sigma_true, dtype=np.float32).reshape(-1)

        # Where cost=0, skip and keep base; else add noise with cost-based std capped below public noise
        noise_sigma_hat = sigma_hat - sigma_true_arr
        print("Difference between sigma_hat and sigma_true: ", noise_sigma_hat)

        if c_arr * lr > 2.7 or lr <= 0 or lr > 1 :
            raise ValueError("cost and learning rate cannot be larger then 2.7 by math")

        thing = c_arr * lr
        moving_step_size = ((noise_sigma_hat**2)/(1+noise_sigma_hat**2))*((thing**2)/(1+thing))
        print("Step size: ", moving_step_size)
        # this only help move one time step.
        sigma_tilde = sigma_hat - moving_step_size * np.sign(noise_sigma_hat)

        print("simga_tilde after step: ", sigma_tilde)

        # For zero-cost entries, keep base exactly
        zero_mask = c_arr == 0
        if np.any(zero_mask):
            sigma_tilde[zero_mask] = sigma_hat[zero_mask]

        return sigma_tilde.astype(np.float32)


# ----------------------------------------------------------------------
# Local test / demo (does not alter main logic)
# ----------------------------------------------------------------------
def demo_interview_signals(
    num_workers: int = 10,
    timesteps: int = 2,
    seed: int = 123,
    out_path: Optional[Path] = None,
):
    """
    Quick simulation to visualize sigma, sigma_hat, and sigma_tilde trajectories.

    - 10 workers with distinct true abilities sampled on (0, 1)
    - Interview costs drawn from a small set of levels in [0, 2.7] (some workers share a level)
    - Two points in time: before interview (public sigma_hat) and after applying cost (sigma_tilde)
    """
    rng = np.random.RandomState(seed)
    sigma_true = rng.uniform(0.05, 0.95, size=num_workers).astype(np.float32)

    # Five cost levels spanning [0, 2.7], repeated so some workers share the same level.
    base_cost_levels = np.linspace(0.0, 2.7, 5, dtype=np.float32)
    interview_costs = np.tile(base_cost_levels, int(np.ceil(num_workers / base_cost_levels.size)))[:num_workers]
    cost_levels = interview_costs  # Keep naming used below for plotting

    mech = ScreeningMechanism(delta0_sq=0.4, lam=1.0, seed=seed)
    # Public signal before interviews
    sigma_hat_pre = mech.init_sigma_hat(
        sigma_true=sigma_true,
        noise_std=1.0,
        rng=rng,
    )
    # Private signal after interviewing with chosen costs
    sigma_tilde_post = mech.screen_worker(
        sigma_true=sigma_true,
        interview_costs=interview_costs,
        sigma_hat=sigma_hat_pre,
        lr=1,
    )

    # Shape to (timesteps, num_workers) for uniform handling in plotting/return.
    sigma_hat_series = np.stack([sigma_hat_pre, sigma_hat_pre], axis=0)
    sigma_tilde_series = np.stack([sigma_hat_pre, sigma_tilde_post], axis=0)

    print("sigma_true:", sigma_true)
    print("interview_costs:", interview_costs)
    print("sigma_hat shape:", sigma_hat_series.shape)
    print("sigma_hat:", sigma_hat_series)
    print("sigma_tilde shape:", sigma_tilde_series.shape)
    print("sigma_tilde:", sigma_tilde_series)

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional plotting
        print(f"matplotlib not available; skipping plot ({exc})")
        return {
            "sigma_true": sigma_true,
            "sigma_hat": sigma_hat_series,
            "sigma_tilde": sigma_tilde_series,
            "interview_costs": interview_costs,
            "cost_levels": cost_levels,
        }

    time_axis = np.arange(timesteps)
    fig, ax = plt.subplots(figsize=(10, 5))
    for w_idx, cost in enumerate(cost_levels):
        ax.plot(
            time_axis,
            np.repeat(sigma_true[w_idx], timesteps),
            "--",
            label=f"worker {w_idx} sigma_true (c={cost:.2f})",
        )
        ax.plot(
            time_axis,
            sigma_hat_series[:, w_idx],
            marker="o",
            label=f"worker {w_idx} sigma_hat (pre)",
        )
        ax.plot(
            time_axis,
            sigma_tilde_series[:, w_idx],
            marker="x",
            label=f"worker {w_idx} sigma_tilde (c={cost:.2f})",
        )

    ax.set_xlabel("Timestep")
    ax.set_ylabel("Signal value")
    ax.set_title("Interview signals per worker (distinct costs)")
    ax.legend(fontsize="xx-small", ncol=3)
    fig.tight_layout()

    if out_path is None:
        out_path = Path(__file__).with_name("interview0_signals.png")
    fig.savefig(out_path)
    plt.close(fig)

    # Per-worker trajectories to show all 10 workers explicitly.
    fig2, axes = plt.subplots(2, 5, figsize=(12, 5), sharex=True, sharey=True)
    axes = axes.flatten()
    for w_idx in range(num_workers):
        axes[w_idx].plot(time_axis, np.repeat(sigma_true[w_idx], timesteps), "k--", label="sigma_true")
        axes[w_idx].plot(time_axis, sigma_hat_series[:, w_idx], label="sigma_hat")
        axes[w_idx].plot(time_axis, sigma_tilde_series[:, w_idx], label="sigma_tilde")
        axes[w_idx].set_title(f"Worker {w_idx} (c={interview_costs[w_idx]:.2f})")
    axes[0].legend(fontsize="xx-small")
    fig2.tight_layout()

    per_worker_path = out_path.with_name("interview0_signals_per_worker.png")
    fig2.savefig(per_worker_path)
    plt.close(fig2)

    print(f"Saved plot to {out_path}")
    print(f"Saved per-worker plot to {per_worker_path}")
    return {
        "sigma_true": sigma_true,
        "sigma_hat": sigma_hat_series,
        "sigma_tilde": sigma_tilde_series,
        "interview_costs": interview_costs,
        "cost_levels": cost_levels,
        "plot_path": out_path,
        "per_worker_plot_path": per_worker_path,
    }


if __name__ == "__main__":  # pragma: no cover - manual demo
    demo_interview_signals()
