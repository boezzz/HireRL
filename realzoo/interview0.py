

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

    delta0_sq: float = 1.0
    lam: float = 1.0
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.delta0_sq <= 0:
            raise ValueError("delta0_sq must be > 0")
        if self.lam <= 0:
            raise ValueError("lam must be > 0")

        # RNG for reproducible simulations
        self._rng: np.random.RandomState
        if self.seed is None:
            # Use global RNG
            self._rng = np.random.mtrand._rand  # type: ignore[attr-defined]
        else:
            self._rng = np.random.RandomState(self.seed)

    # ------------------------------------------------------------------
    # Cost -> variance map
    # ------------------------------------------------------------------
    def interview_var(self, cost: float) -> float:
        """Return interview signal variance δ^2(c) for a given cost.

        δ^2(c) = δ0^2 * exp(-λ c)

        Args
        ----
        cost:
            Interview/screening cost c >= 0.

        Returns
        -------
        float
            Signal noise variance δ^2(c).
        """
        c = max(0.001, float(cost))
        return float(self.delta0_sq * np.exp(-self.lam * c))


    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------
    def screen_worker(
        self,
        sigma_true: np.ndarray,
        sigma_hat_0: Optional[np.ndarray],  # kept for API compatibility; not used
        cost: float,
    ) -> Tuple[np.ndarray, float]:
        """Generate a private interview signal for a single worker.

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
        cost:
            Interview cost c_{interview,ij} >= 0.

        Returns
        -------
        tilde_sigma: np.ndarray
            The private interview signal \tilde{σ}_{ij} with the same shape
            as `sigma_true`.
        precision: float
            A convenient [0,1] summary of informativeness:
                precision = 1 - δ^2(c) / δ0^2
            where 0 means "no extra information" (cost=0) and 1 means
            "maximal precision" (cost → ∞).
        """
        var = self.interview_var(cost)
        std = float(np.sqrt(var))

        # Draw Gaussian noise with the target variance
        noise = self._rng.randn(*sigma_true.shape).astype(np.float32) * std
        tilde_sigma = sigma_true + noise

        # Precision summary in [0,1]
        precision = float(1.0 - var / self.delta0_sq) if self.delta0_sq > 0 else 0.0

        return tilde_sigma.astype(np.float32), precision

    # ------------------------------------------------------------------
    # Optional batch helper
    # ------------------------------------------------------------------
    def screen_batch(
        self,
        sigma_true_batch: np.ndarray,
        sigma_hat0_batch: Optional[np.ndarray],  # unused, kept for symmetry
        cost_batch: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized screening for a batch of workers.

        Args
        ----
        sigma_true_batch:
            Array of true abilities, shape (N, ...) where ... are ability dims.
        sigma_hat0_batch:
            Public initial signals (ignored in current implementation).
        cost_batch:
            Array of costs c_{interview,ij} with shape (N,).

        Returns
        -------
        tilde_sigma_batch: np.ndarray
            Interview signals for each worker, same shape as sigma_true_batch.
        precision_batch: np.ndarray
            Precision summary for each worker, shape (N,).
        """
        sigma_true_batch = np.asarray(sigma_true_batch)
        cost_batch = np.asarray(cost_batch, dtype=float)

        if sigma_true_batch.ndim == 1:
            # Make it (N,1) for broadcasting, then squeeze back
            sigma_true = sigma_true_batch[:, None]
            extra_dim = True
        else:
            sigma_true = sigma_true_batch
            extra_dim = False

        N = sigma_true.shape[0]
        if cost_batch.shape[0] != N:
            raise ValueError("cost_batch must have the same length as sigma_true_batch")

        vars_ = np.array([self.interview_var(c) for c in cost_batch], dtype=float)
        stds = np.sqrt(vars_)  # shape (N,)

        noise = self._rng.randn(*sigma_true.shape).astype(np.float32)
        # Broadcast stds along ability dimensions
        while stds.ndim < noise.ndim:
            stds = stds[:, None]
        noise *= stds

        tilde_sigma = sigma_true + noise
        precision = np.where(self.delta0_sq > 0.0,
                             1.0 - vars_ / self.delta0_sq,
                             0.0)

        if extra_dim:
            tilde_sigma = tilde_sigma[:, 0]

        return tilde_sigma.astype(np.float32), precision.astype(np.float32)