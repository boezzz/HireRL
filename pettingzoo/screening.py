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
    """Different screening technology specifications."""
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
        seed: Optional[int] = None
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

    def get_precision(self, cost: float) -> float:
        """
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

    def screen_worker(
        self,
        sigma_true: np.ndarray,
        sigma_hat_0: np.ndarray,
        cost: float
    ) -> Tuple[np.ndarray, float]:
        """
        Perform screening/interview on a worker.

        Returns noisy estimate of true ability based on screening investment.

        Formula:
            σ_estimate = σ̂_0 + precision(c) * (σ_true - σ̂_0) + ε
        where:
            - σ̂_0 is the public signal (free information)
            - (σ_true - σ̂_0) is the hidden information gap
            - precision(c) determines how much of the gap is revealed
            - ε ~ N(0, noise_std²) is residual measurement error

        Args:
            sigma_true: Worker's true ability (private info)
            sigma_hat_0: Public signal (from resume)
            cost: Screening cost to invest

        Returns:
            Tuple of (estimated_ability, precision_achieved)
        """
        precision = self.get_precision(cost)

        # Reveal fraction of hidden information based on precision
        information_gap = sigma_true - sigma_hat_0
        revealed_info = precision * information_gap

        # Add measurement noise (screening isn't perfect)
        noise = self.rng.randn(*sigma_true.shape).astype(np.float32) * self.noise_std

        # Final estimate
        sigma_estimate = sigma_hat_0 + revealed_info + noise

        return sigma_estimate, precision

    def get_expected_variance(self, cost: float, sigma_variance: float = 1.0) -> float:
        """
        Compute expected variance of estimate given screening cost.

        Useful for firms to decide optimal screening investment.

        The variance of the estimate is:
            Var[σ_estimate] = (1 - precision)² * Var[σ_true] + noise_std²

        Args:
            cost: Screening cost
            sigma_variance: Variance of true ability distribution

        Returns:
            Expected variance of ability estimate
        """
        precision = self.get_precision(cost)

        # Variance from unrevealed information
        unrevealed_var = (1 - precision) ** 2 * sigma_variance

        # Variance from measurement noise
        noise_var = self.noise_std ** 2

        return unrevealed_var + noise_var

    def optimal_screening_cost(
        self,
        expected_profit_if_good: float,
        expected_profit_if_bad: float,
        cost_grid: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute optimal screening cost using simple decision-theoretic approach.

        Firm's problem:
            max_c { precision(c) * (V_good - V_bad) - c }

        where:
            - V_good: expected profit if worker is high ability
            - V_bad: expected profit if worker is low ability
            - precision(c): probability of correctly identifying type

        This is a simplified model. In practice, firms would use Bayesian updating
        and dynamic programming.

        Args:
            expected_profit_if_good: Profit from hiring good worker
            expected_profit_if_bad: Profit from hiring bad worker
            cost_grid: Grid of costs to search over (default: [0, 0.01, ..., c_max])

        Returns:
            Optimal screening cost
        """
        if cost_grid is None:
            cost_grid = np.linspace(0, self.c_max, 100)

        profit_diff = expected_profit_if_good - expected_profit_if_bad

        # Expected value for each screening level
        expected_values = []
        for c in cost_grid:
            precision = self.get_precision(c)
            # Value = information value - cost
            value = precision * profit_diff - c
            expected_values.append(value)

        # Find cost that maximizes expected value
        best_idx = np.argmax(expected_values)
        optimal_cost = cost_grid[best_idx]

        return float(optimal_cost)


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
        signal_var = screening_mechanism.get_expected_variance(screening_cost)

        # Current belief
        prior_mean = self.belief_mean[worker_id]
        prior_var = self.belief_var[worker_id]

        # Bayesian update
        posterior_var = (prior_var * signal_var) / (prior_var + signal_var)
        posterior_mean = (prior_mean * signal_var + sigma_estimate * prior_var) / (prior_var + signal_var)

        self.belief_mean[worker_id] = posterior_mean
        self.belief_var[worker_id] = posterior_var

    def update_from_performance(
        self,
        worker_id: int,
        observed_profit: float,
        worker_experience: float,
        profit_noise_var: float = 0.1
    ):
        """
        Update beliefs from observed performance (employer learning).

        Observation: profit = σ_j + β*log(1 + exp_j) + ε

        We infer σ_j from profit by subtracting experience component.

        Args:
            worker_id: Worker whose performance is observed
            observed_profit: Realized profit p_ij,t
            worker_experience: Worker's experience level exp_j,t
            profit_noise_var: Variance of profit noise
        """
        beta = 0.5  # Experience return parameter (should match worker_pool)

        # Infer ability signal from performance
        # profit = σ + β*log(1+exp) + ε
        # => σ_implied = profit - β*log(1+exp)
        experience_component = beta * np.log1p(worker_experience)
        sigma_implied = observed_profit - experience_component

        # Treat as noisy signal of true ability
        signal_var = profit_noise_var

        # Bayesian update (same as screening)
        prior_mean = self.belief_mean[worker_id, 0]  # Assuming ability_dim=1
        prior_var = self.belief_var[worker_id, 0]

        posterior_var = (prior_var * signal_var) / (prior_var + signal_var)
        posterior_mean = (prior_mean * signal_var + sigma_implied * prior_var) / (prior_var + signal_var)

        self.belief_mean[worker_id, 0] = posterior_mean
        self.belief_var[worker_id, 0] = posterior_var

    def get_belief(self, worker_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current belief about a worker.

        Returns:
            Tuple of (mean, variance)
        """
        return self.belief_mean[worker_id], self.belief_var[worker_id]

    def get_expected_profit(self, worker_id: int, experience: float, beta: float = 0.5) -> float:
        """
        Compute expected profit given current beliefs.

        E[profit | beliefs] = E[σ_j | beliefs] + β*log(1 + exp_j)

        Args:
            worker_id: Worker ID
            experience: Worker's current experience
            beta: Experience return parameter

        Returns:
            Expected profit
        """
        mean_ability = self.belief_mean[worker_id, 0]  # Assuming ability_dim=1
        experience_value = beta * np.log1p(experience)
        return float(mean_ability + experience_value)
