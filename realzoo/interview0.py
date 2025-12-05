

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
        sigma_hat: np.ndarray
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
        moving_step_size = ((noise_sigma_hat**2)/(1+noise_sigma_hat**2))*((c_arr**2)/(1+c_arr))
        sigma_tilde = sigma_true_arr - moving_step_size * np.sign(noise_sigma_hat)

        # For zero-cost entries, keep base exactly
        zero_mask = c_arr == 0
        if np.any(zero_mask):
            sigma_tilde[zero_mask] = sigma_hat[zero_mask]

        return sigma_tilde.astype(np.float32)
