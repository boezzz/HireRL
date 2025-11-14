import numpy as np
from typing import Tuple, Optional




class FirmBeliefs:

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


    def initialize_from_interview_signal(
        self,
        worker_id: int,
        tilde_sigma_interview: float,
        signal_noise_var: float = 0.25,
    ) -> None:
        """Initialize this firm's belief for a specific worker from the interview signal.

        This should be called *after* the interview phase, when firm i has
        observed its private signal \tilde{σ}_{ij,t=interview}. It sets the
        firm's posterior mean to that signal and uses `signal_noise_var` as the
        initial variance for all ability dimensions of this worker.

        Args:
            worker_id: Index of the worker j.
            tilde_sigma_interview: Private interview signal \tilde{σ}_{ij,interview}.
            signal_noise_var: Initial variance assigned to this belief.
        """
        mean_val = float(tilde_sigma_interview)
        var_val = float(signal_noise_var)
        self.belief_mean[worker_id, :] = mean_val
        self.belief_var[worker_id, :] = var_val

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
        在当前的 after_hiring_update.py 中，这个函数主要是：

        ✔ 给 其它模块 访问最新 posterior belief 的接口

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
        new_tilde_sigma_performance = (1.0 - vx) * float(tilde_sigma_interview) + vx * float(p_ijt)
        self.belief_mean[worker_id, 0] = new_tilde_sigma_performance


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
