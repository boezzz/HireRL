"""
The screening mechanism returns a noisy estimate of true ability:
    σ_precision = σ̂_j,0 + (σ_j - σ̂_j,0) * precision(c)
where precision(c) ∈ [0, 1] increases with cost c

At c=0: σ_precision = σ̂_j,0 (no new information, just public signal)
At c=∞: σ_precision → σ_j (perfect information about true ability)
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum


class ScreeningTechnology(Enum):
    """[DEPRECATED] Legacy precision(c) forms kept for backward compatibility."""
    LINEAR = "linear"           # precision(c) = min(c / c_max, 1)
    LOGARITHMIC = "log"         # precision(c) = log(1 + c) / log(1 + c_max)
    SQRT = "sqrt"               # precision(c) = sqrt(c / c_max)
    SIGMOID = "sigmoid"         # precision(c) = 1 / (1 + exp(-k*(c - c_mid)))


class ScreeningMechanism:
    """
    Implements costly screening/interviewing mechanism for firms.

    Firms can invest cost c to get a more precise estimate of worker ability.
    This addresses the fundamental information asymmetry in labor markets.
    """

    def __init__(
        self,
        technology: ScreeningTechnology = ScreeningTechnology.SQRT,
        c_max: float = 1.0,  # Maximum meaningful cost
        noise_std: float = 0.2,  # Residual noise even with perfect screening
        seed: Optional[int] = None,
        delta0_sq: float = 1.0,   # Interview noise at zero cost: δ0^2
        lam: float = 1.0          # Cost–precision decay parameter λ
    ):
        """
        Initialize screening mechanism.

        Args:
            technology: Functional form of cost-precision relationship
            c_max: Cost level that gives near-perfect information
            noise_std: Residual noise (even expensive screening isn't perfect)
            seed: Random seed for reproducibility
        """
        self.technology = technology
        self.c_max = c_max
        self.noise_std = noise_std
        self.rng = np.random.RandomState(seed)
        self.delta0_sq = float(delta0_sq) # Interview noise at zero cost: δ0^2
        self.lam = float(lam) # Cost–precision decay parameter λ

    def get_precision(self, cost: float) -> float:
        """
        [DEPRECATED] Prefer using interview_var(cost) which implements δ²(c) from the paper.
        Compute precision level for a given screening cost.

        Precision ∈ [0, 1] determines how much of the gap between public signal
        and true ability is revealed.

        Args:
            cost: Screening/interview cost (non-negative)

        Returns:
            Precision level in [0, 1]
        """
        if cost < 0:
            cost = 0.0

        if self.technology == ScreeningTechnology.LINEAR:
            # Linear: precision increases linearly with cost
            precision = min(cost / self.c_max, 1.0)

        elif self.technology == ScreeningTechnology.LOGARITHMIC:
            # Logarithmic: diminishing returns to screening investment
            # Models: initial screening very valuable, but hard to perfect
            precision = np.log1p(cost) / np.log1p(self.c_max)

        elif self.technology == ScreeningTechnology.SQRT:
            # Square root: moderate diminishing returns
            # Balanced between linear and logarithmic
            precision = np.sqrt(min(cost / self.c_max, 1.0))

        elif self.technology == ScreeningTechnology.SIGMOID:
            # Sigmoid: threshold effect
            # Models: need minimum investment to learn anything useful
            k = 5.0  # Steepness parameter
            c_mid = self.c_max / 2  # Inflection point
            precision = 1.0 / (1.0 + np.exp(-k * (cost - c_mid)))
            # Normalize to [0, 1]
            precision = (precision - 1/(1 + np.exp(k*c_mid))) / (1/(1+np.exp(-k*c_mid)) - 1/(1+np.exp(k*c_mid)))

        else:
            raise ValueError(f"Unknown screening technology: {self.technology}")

        return float(np.clip(precision, 0.0, 1.0))

    def interview_var(self, cost: float) -> float:
        """
        Interview signal variance per the paper:
            δ^2(c) = δ0^2 * exp(-λ c)
        """
        c = max(0.01, float(cost))
        return float(self.delta0_sq * np.exp(-self.lam * c))

    def screen_worker(
        self,
        sigma_true: np.ndarray,
        sigma_hat_0: np.ndarray,
        cost: float
    ) -> Tuple[np.ndarray, float]:
        # Use paper-consistent private interview signal:
        # \tilde{σ}_{ij,t} = σ_j + η,  η ~ N(0, δ^2(c)) where δ^2(c) = δ0^2 * exp(-λ c)
        var = self.interview_var(cost)
        std = np.sqrt(var)
        noise = self.rng.randn(*sigma_true.shape).astype(np.float32) * std
        tilde_sigma = sigma_true + noise
        # Optional precision metric for logging/analysis: 1 - Var/Var0 ∈ [0,1]
        precision = float(1.0 - var / self.delta0_sq) if self.delta0_sq > 0 else 0.0
        return tilde_sigma, precision



    # def optimal_screening_cost(
    #     self,
    #     expected_profit_if_good: float,
    #     expected_profit_if_bad: float,
    #     cost_grid: Optional[np.ndarray] = None
    # ) -> float:
    #     """
    #     Compute optimal screening cost using simple decision-theoretic approach.

    #     Firm's problem:
    #         max_c { precision(c) * (V_good - V_bad) - c }

    #     where:
    #         - V_good: expected profit if worker is high ability
    #         - V_bad: expected profit if worker is low ability
    #         - precision(c): probability of correctly identifying type

    #     This is a simplified model. In practice, firms would use Bayesian updating
    #     and dynamic programming. #TODO: Implement Bayesian updating.

    #     Args:
    #         expected_profit_if_good: Profit from hiring good worker
    #         expected_profit_if_bad: Profit from hiring bad worker
    #         cost_grid: Grid of costs to search over (default: [0, 0.01, ..., c_max])

    #     Returns:
    #         Optimal screening cost
    #     """
    #     if cost_grid is None:
    #         cost_grid = np.linspace(0, self.c_max, 100)

    #     profit_diff = expected_profit_if_good - expected_profit_if_bad

    #     # Expected value for each screening level
    #     expected_values = []
    #     for c in cost_grid:
    #         precision = self.get_precision(c)
    #         # Value = information value - cost
    #         value = precision * profit_diff - c
    #         expected_values.append(value)

    #     # Find cost that maximizes expected value
    #     best_idx = np.argmax(expected_values)
    #     optimal_cost = cost_grid[best_idx]

    #     return float(optimal_cost)


class FirmBeliefs:
    """
    Maintains firm's beliefs about worker abilities.

    Firms perform Bayesian updating:
    1. Prior: Initial belief based on public signal σ̂_j,0
    2. Screening: Update based on interview/screening (if conducted)
    3. Performance: Update based on observed profit realizations

    This implements "employer learning" (Altonji & Pierret 2001):
    Firms gradually learn true ability by observing performance.
    """

    def __init__(self, num_workers: int, ability_dim: int = 1):
        """
        Initialize belief system.

        Args:
            num_workers: Total number of workers
            ability_dim: Dimensionality of ability
        """
        self.num_workers = num_workers
        self.ability_dim = ability_dim

        # Belief parameters (mean and variance for each worker)
        # Start with uninformative priors
        self.belief_mean = np.zeros((num_workers, ability_dim), dtype=np.float32)
        self.belief_var = np.ones((num_workers, ability_dim), dtype=np.float32) * 10.0  # High initial uncertainty

    def initialize_from_public_signals(self, sigma_hat_0: np.ndarray, signal_noise_var: float = 0.25):
        """
        Initialize beliefs from public signals (resumes).

        Prior: σ_j ~ N(σ̂_j,0, signal_noise_var)

        Args:
            sigma_hat_0: Public signals for all workers (num_workers, ability_dim)
            signal_noise_var: Known variance of public signal noise
        """
        self.belief_mean = sigma_hat_0.copy()
        self.belief_var = np.ones((self.num_workers, self.ability_dim), dtype=np.float32) * signal_noise_var

    def update_from_screening(
        self,
        worker_id: int,
        sigma_estimate: np.ndarray,
        screening_cost: float,
        screening_mechanism: ScreeningMechanism
    ):
        """
        Update beliefs using Bayesian updating after screening.

        Combines prior with screening signal using optimal weights.

        Bayesian update (for Gaussian distributions):
            μ_posterior = (μ_prior * σ²_signal + μ_signal * σ²_prior) / (σ²_prior + σ²_signal)
            σ²_posterior = (σ²_prior * σ²_signal) / (σ²_prior + σ²_signal)

        Args:
            worker_id: Worker who was screened
            sigma_estimate: Screening result
            screening_cost: Cost spent on screening
            screening_mechanism: Screening mechanism (to get precision/variance)
        """
        # Get variance of screening signal
        signal_var = screening_mechanism.interview_var(screening_cost)

        # Current belief
        prior_mean = self.belief_mean[worker_id]
        prior_var = self.belief_var[worker_id]

        # Bayesian update
        posterior_var = (prior_var * signal_var) / (prior_var + signal_var)
        posterior_mean = (prior_mean * signal_var + sigma_estimate * prior_var) / (prior_var + signal_var) ####??

        self.belief_mean[worker_id] = posterior_mean
        self.belief_var[worker_id] = posterior_var

    def interview_and_update(self,
                             worker_id: int,
                             sigma_true: np.ndarray,
                             sigma_hat_0: np.ndarray,
                             cost: float,
                             screening_mechanism: ScreeningMechanism) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Convenience pipeline: generate a private interview signal and immediately update beliefs.
        Returns (tilde_sigma, precision, posterior_mean, posterior_var).
        """
        tilde_sigma, precision = screening_mechanism.screen_worker(sigma_true, sigma_hat_0, cost)
        self.update_from_screening(worker_id, tilde_sigma, cost, screening_mechanism)
        return tilde_sigma, precision, self.belief_mean[worker_id].copy(), self.belief_var[worker_id].copy()

    def update_from_performance(
        self,
        worker_id: int,
        p_ijt: float,
        exp_t: float,
        tilde_sigma_interview: float,
        delta_interview_sq: float,
        delta_eps_sq: float,
    ):
        """
        [DEPRECATED signature change in progress] Paper-consistent update.
        This forwards to `update_from_performance_convex` to keep a single source of truth.
        """
        self.update_from_performance_convex(worker_id,
                                            p_ijt=p_ijt,
                                            tilde_sigma_interview=tilde_sigma_interview,
                                            exp_t=exp_t,
                                            delta_interview_sq=delta_interview_sq,
                                            delta_eps_sq=delta_eps_sq)

    def get_belief(self, worker_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current belief about a worker.

        Returns:
            Tuple of (mean, variance)
        """
        return self.belief_mean[worker_id], self.belief_var[worker_id]

    def get_expected_profit(self, worker_id: int, exp_tm1: float,
                            g0: float = 0.1, g1: float = 0.5, theta: float = 0.05,
                            f_type: str = 'linear') -> float:
        """
        Expected profit under current belief mean, following the paper's structure (without noise):
            E[p_{ij,t}] = f[ exp_{j,t-1} + (g0 + g1 * E[σ_j]) * exp(-θ * exp_{j,t-1}) ]
        """
        mean_ability = float(self.belief_mean[worker_id, 0])
        core = float(exp_tm1) + (g0 + g1 * mean_ability) * np.exp(-theta * float(exp_tm1))
        if f_type == 'linear':
            val = core
        elif f_type == 'log':
            val = np.log1p(core)
        elif f_type == 'diminishing':
            val = core / (1.0 + 0.1 * core)
        else:
            raise ValueError(f"Unknown profit function type: {f_type}")
        return float(val)

    def public_signal_next(self, sigma_hat_t: np.ndarray, employed: bool, gamma: float) -> np.ndarray:
        """
        Deterministic public-signal drift with tenure (per paper):
            \hat{σ}_{j,t+1} = \hat{σ}_{j,t} + γ·1{employed at t}
        """
        return sigma_hat_t + (gamma if employed else 0.0)

    def update_from_performance_convex(self,
                                       worker_id: int,
                                       p_ijt: float,
                                       tilde_sigma_interview: float,
                                       exp_t: float,
                                       delta_interview_sq: float,
                                       delta_eps_sq: float):
        """
        Paper-style convex combination update:
            \tilde{σ}_{t+1} = (1 - v_x)\,\tilde{σ}_{interview} + v_x\, p_{ij,t}
        with v_x = (exp * K1) / (1 + (exp - 1) * K1),  K1 = δ_interview^2 / (δ_interview^2 + δ_ε^2).
        This method stores the new score in belief_mean[worker_id, 0].
        """
        exp_clamped = max(0.0, float(exp_t))
        K1 = float(delta_interview_sq) / float(delta_interview_sq + delta_eps_sq) if (delta_interview_sq + delta_eps_sq) > 0 else 0.0
        vx = (exp_clamped * K1) / (1.0 + (exp_clamped - 1.0) * K1) if K1 > 0 else 0.0
        new_score = (1.0 - vx) * float(tilde_sigma_interview) + vx * float(p_ijt)
        self.belief_mean[worker_id, 0] = new_score


# ===============================================================
# Profit Function Definitions (Following Schönberg & Farber setup)
# ===============================================================

def generate_profit(exp_tm1: float,
                    sigma_j: float,
                    employed_tm1: bool,
                    g0: float = 0.1,
                    g1: float = 0.5,
                    theta: float = 0.05,
                    delta_eps_sq: float = 0.1,
                    f_type: str = 'linear',
                    rng: Optional[np.random.RandomState] = None) -> float:
    """
    Generate realized profit p_{ij,t} following the paper's structure:

        p_{ij,t} = f[ exp_{j,t-1} + (g0 + g1*sigma_j)*1{employed}*exp(-theta*exp_{j,t-1}) ] + epsilon

    where epsilon ~ N(0, delta_eps_sq).

    Args:
        exp_tm1: Experience level at t-1
        sigma_j: True ability of the worker
        employed_tm1: Whether the worker was employed at t-1
        g0, g1, theta: Parameters controlling experience and ability contributions
        delta_eps_sq: Variance of performance noise ε
        f_type: Form of profit function ('linear', 'log', 'diminishing')
        rng: Optional RNG for reproducibility

    Returns:
        Realized profit value p_{ij,t}.
    """
    if rng is None:
        rng = np.random

    # Core deterministic term
    core = exp_tm1 + (g0 + g1 * sigma_j) * (1.0 if employed_tm1 else 0.0) * np.exp(-theta * exp_tm1)

    # Select profit function form f(x)
    if f_type == 'linear':
        val = core  # Linear production: f(x) = x
    elif f_type == 'log':
        val = np.log1p(core)  # Concave returns: f(x) = log(1+x)
    elif f_type == 'diminishing':
        val = core / (1.0 + 0.1 * core)  # Diminishing marginal returns
    else:
        raise ValueError(f"Unknown profit function type: {f_type}")

    # Add random noise ε ~ N(0, δ_ε²)
    eps = rng.normal(0.0, np.sqrt(delta_eps_sq))
    return float(val + eps)


class ProfitFunctionExamples:
    """Convenience access to example profit functions used in calibration or testing."""

    @staticmethod
    def linear(x: float) -> float:
        return x

    @staticmethod
    def log(x: float) -> float:
        return np.log1p(x)

    @staticmethod
    def diminishing(x: float) -> float:
        return x / (1.0 + 0.1 * x)


# ===============================================================
# Experience Accumulation (per paper)
# ===============================================================
def update_experience(exp_t: float,
                      sigma_j: float,
                      employed_t: bool,
                      g0: float,
                      g1: float,
                      theta: float) -> float:
    """
    Deterministic on-the-job experience accumulation:
        exp_{t+1} = exp_t + (g0 + g1*sigma_j) * 1{employed at t} * exp(-theta * exp_t)
    with exp_t >= 0 and theta > 0.
    """
    exp_t = max(0.0, float(exp_t))
    theta = float(theta)
    if theta <= 0:
        raise ValueError("theta must be > 0")
    increment = (g0 + g1 * sigma_j) * (1.0 if employed_t else 0.0) * np.exp(-theta * exp_t)
    return float(exp_t + increment)

def update_experience_vec(exp_t: np.ndarray,
                          sigma_j: np.ndarray,
                          employed_t: np.ndarray,
                          g0: float,
                          g1: float,
                          theta: float) -> np.ndarray:
    """
    Vectorized version for batch updates over workers.
    Shapes:
      - exp_t: (N,)
      - sigma_j: (N,)
      - employed_t: (N,) boolean or {0,1}
    """
    exp_t = np.maximum(0.0, exp_t.astype(float))
    if theta <= 0:
        raise ValueError("theta must be > 0")
    employed_float = employed_t.astype(float)
    increment = (g0 + g1 * sigma_j) * employed_float * np.exp(-theta * exp_t)
    return (exp_t + increment).astype(float)
