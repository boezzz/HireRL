"""
Worker Pool Management
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List


@dataclass
class WorkerState:
    """
    Complete state for a single worker.

    Attributes:
        worker_id: Unique identifier
        sigma_true: True ability σ_j ∈ R^d (PRIVATE - only worker knows)
        sigma_hat_0: Initial public signal σ̂_j,0 (PUBLIC - from resume)
        experience: On-the-job experience exp_j,t (accumulates while employed)
        tenure: Total time employed τ_j,t (PUBLIC - observable working years)
        employed_by: Company ID if employed, -1 if unemployed
        wage: Current wage if employed
        sigma_hat: Current public ability signal σ̂_j,t = φ(σ̂_j,0, τ_j,t)
    """
    worker_id: int
    sigma_true: np.ndarray  # True ability (d-dimensional)
    sigma_hat_0: np.ndarray  # Initial public signal (d-dimensional)
    experience: float  # Accumulated experience
    tenure: int  # Total periods employed
    employed_by: int  # Company index or -1
    wage: float  # Current wage
    sigma_hat: np.ndarray  # Current public signal


class WorkerPool:
    """
    Manages the population of workers in the labor market.

    The worker pool is treated as part of the environment (not strategic agents).
    Workers are greedy: they accept the highest-paying job offer.
    """

    def __init__(
        self,
        num_workers: int,
        ability_dim: int = 1,
        gamma: float = 0.1,  # Tenure signal growth rate
        g0: float = 0.1,     # Base experience growth
        g1: float = 0.05,    # Ability-dependent experience growth
        signal_noise_std: float = 0.5,  # Noise in initial public signal
        seed: Optional[int] = None
    ):
        """
        Initialize worker pool.

        Args:
            num_workers: Total number of workers in the market
            ability_dim: Dimensionality of ability vector (default 1 for simplicity)
            gamma: Rate at which public signal grows with tenure (σ̂_j,t = σ̂_j,0 + γ*τ_j,t)
            g0: Base rate of experience accumulation
            g1: Ability-dependent experience growth (g(σ_j) = g0 + g1*σ_j)
            signal_noise_std: Standard deviation of noise in initial public signal
            seed: Random seed for reproducibility
        """
        self.num_workers = num_workers
        self.ability_dim = ability_dim
        self.gamma = gamma
        self.g0 = g0
        self.g1 = g1
        self.signal_noise_std = signal_noise_std
        self.rng = np.random.RandomState(seed)

        # Worker states
        self.workers: List[WorkerState] = []

    def reset(self, seed: Optional[int] = None) -> List[WorkerState]:
        """
        Reset worker pool to initial state.

        Initializes:
        - True abilities σ_j ~ N(0, I) for each worker
        - Public signals σ̂_j,0 = σ_j + ε, where ε ~ N(0, noise_std²I)
        - All workers start unemployed with zero experience

        Returns:
            List of initialized worker states
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.workers = []

        for j in range(self.num_workers):
            # Sample true ability from standard normal
            sigma_true = self.rng.randn(self.ability_dim).astype(np.float32)

            # Create noisy public signal (resume quality doesn't perfectly reveal ability)
            noise = self.rng.randn(self.ability_dim).astype(np.float32) * self.signal_noise_std
            sigma_hat_0 = sigma_true + noise

            worker = WorkerState(
                worker_id=j,
                sigma_true=sigma_true,
                sigma_hat_0=sigma_hat_0.copy(),
                experience=0.0,
                tenure=0,
                employed_by=-1,
                wage=0.0,
                sigma_hat=sigma_hat_0.copy()
            )
            self.workers.append(worker)

        return self.workers

    def update_experience_and_tenure(self):
        """
        Update worker experience and tenure at end of period.

        Experience accumulation (only while employed):
            exp_{j,t+1} = exp_{j,t} + g(σ_j) * 1{j employed at t}
            where g(σ_j) = g0 + g1 * σ_j (higher ability → faster learning)

        Tenure accumulation (only while employed):
            τ_{j,t+1} = τ_{j,t} + 1{j employed at t}

        Public signal update (linear model):
            σ̂_{j,t+1} = σ̂_{j,0} + γ * τ_{j,t+1}
        """
        for worker in self.workers:
            if worker.employed_by >= 0:  # If employed
                # Experience grows faster for higher ability workers (learning-by-doing)
                # This creates heterogeneous returns to experience
                sigma_scalar = worker.sigma_true[0] if self.ability_dim == 1 else np.mean(worker.sigma_true)
                experience_growth = self.g0 + self.g1 * sigma_scalar
                worker.experience += experience_growth

                # Tenure increments by 1 period
                worker.tenure += 1

                # Update public signal (employers observe tenure growth)
                worker.sigma_hat = worker.sigma_hat_0 + self.gamma * worker.tenure

    def get_unemployed_workers(self) -> List[int]:
        """
        Get list of unemployed worker IDs.

        Returns:
            List of worker IDs who are currently unemployed
        """
        return [w.worker_id for w in self.workers if w.employed_by == -1]

    def get_employed_by_company(self, company_id: int) -> List[int]:
        """
        Get workers employed by a specific company.

        Args:
            company_id: Company index

        Returns:
            List of worker IDs employed by this company
        """
        return [w.worker_id for w in self.workers if w.employed_by == company_id]

    def hire_worker(self, worker_id: int, company_id: int, wage: float):
        """
        Hire a worker to a company.

        Args:
            worker_id: Worker to hire
            company_id: Hiring company
            wage: Offered wage
        """
        worker = self.workers[worker_id]
        worker.employed_by = company_id
        worker.wage = wage

    def fire_worker(self, worker_id: int):
        """
        Fire a worker (they become unemployed).

        When fired, the worker's public signal σ̂_j,0 for next employer
        becomes their current signal σ̂_j,t (includes tenure growth).

        Args:
            worker_id: Worker to fire
        """
        worker = self.workers[worker_id]
        worker.employed_by = -1
        worker.wage = 0.0
        # Update their "resume" signal for next employer (Per PDF: σ̂_j,0^(next) := σ̂_j,t)
        worker.sigma_hat_0 = worker.sigma_hat.copy()

    def apply_deterministic_quit_rule(self) -> List[int]:
        """
        Apply deterministic quit rule: workers quit if they see comparable workers
        earning higher wages.

        Rule: If worker j finds another worker with σ̂_{-j} ≈ σ̂_j but w_{-j} > w_j,
        then worker j quits and rejoins the unemployment pool.

        This creates wage competition and forces firms to pay market rates.

        Returns:
            List of worker IDs who quit
        """
        quit_workers = []

        # Group workers by public signal (rounded to avoid floating point issues)
        # Using 3 decimal places for "comparable ability"
        signal_groups: Dict[int, List[WorkerState]] = {}

        for worker in self.workers:
            if worker.employed_by >= 0:  # Only employed workers can quit
                # Round public signal to group comparable workers
                signal_key = int(np.round(worker.sigma_hat[0] * 1000))
                if signal_key not in signal_groups:
                    signal_groups[signal_key] = []
                signal_groups[signal_key].append(worker)

        # Check each group for wage disparities
        for signal_key, group in signal_groups.items():
            if len(group) <= 1:
                continue

            # Find max wage in this group
            max_wage = max(w.wage for w in group)

            # Workers earning less than max quit
            for worker in group:
                if worker.wage < max_wage - 1e-6:  # Small epsilon for numerical stability
                    self.fire_worker(worker.worker_id)
                    quit_workers.append(worker.worker_id)

        return quit_workers

    def get_public_state(self) -> Dict[str, np.ndarray]:
        """
        Get publicly observable information about all workers.

        This is what firms can observe without screening/interviewing.

        Returns:
            Dictionary with arrays for:
            - sigma_hat: Public ability signals (n_workers, d)
            - experience: Observable experience (n_workers,)
            - tenure: Observable tenure (n_workers,)
            - employed_by: Employment status (n_workers,)
            - wages: Current wages (n_workers,)
        """
        return {
            'sigma_hat': np.array([w.sigma_hat for w in self.workers], dtype=np.float32),
            'experience': np.array([w.experience for w in self.workers], dtype=np.float32),
            'tenure': np.array([w.tenure for w in self.workers], dtype=np.int32),
            'employed_by': np.array([w.employed_by for w in self.workers], dtype=np.int32),
            'wages': np.array([w.wage for w in self.workers], dtype=np.float32),
        }

    def compute_match_profit(self, worker_id: int, company_id: int) -> float:
        """
        Compute match-specific profit p(σ, exp) for a worker-company pair.

        Profit function: p_ij = σ_j + β * log(1 + exp_j)

        This captures:
        - True ability σ_j determines base productivity
        - Experience has diminishing returns (log function)
        - Profit depends on TRUE ability (private info), not public signal

        Args:
            worker_id: Worker index
            company_id: Company index (unused for now, could add match-specific effects)

        Returns:
            Match-specific profit
        """
        worker = self.workers[worker_id]
        beta = 0.5  # Experience return parameter

        # Profit depends on TRUE ability (which firms must infer)
        sigma_scalar = worker.sigma_true[0] if self.ability_dim == 1 else np.mean(worker.sigma_true)
        # Clamp experience to non-negative to avoid log1p issues
        exp_clamped = max(0.0, worker.experience)
        profit = sigma_scalar + beta * np.log1p(exp_clamped)

        return float(profit)

    def get_average_wage(self) -> float:
        """Get average wage of employed workers."""
        employed_wages = [w.wage for w in self.workers if w.employed_by >= 0]
        return np.mean(employed_wages) if employed_wages else 0.0

    def get_unemployment_rate(self) -> float:
        """Get fraction of workers who are unemployed."""
        unemployed = sum(1 for w in self.workers if w.employed_by == -1)
        return unemployed / self.num_workers

    def get_worker_state(self, worker_id: int) -> WorkerState:
        """Get complete state of a specific worker."""
        return self.workers[worker_id]
