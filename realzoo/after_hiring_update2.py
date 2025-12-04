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


    def get_belief(self, worker_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current belief about a worker.
        在当前的 2after_hiring_update.py 中，这个函数主要是：

        ✔ 给 其它模块 访问最新 posterior belief 的接口

        Returns:
            Tuple of (mean, variance)
        """
        return self.belief_mean[worker_id], self.belief_var[worker_id]

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
# Note: generate_profit() function moved to generated_profit3.py
# ProfitFunctionExamples class also available there as _profit_core()
# ===============================================================


# ===============================================================
# Experience Accumulation (per paper)
# ===============================================================
def update_experience(
    exp_t: float,
    sigma_j: float,
    employed_t: bool,
    g0: float,
    g1: float,
    theta: float,
) -> float:
    """
    Experience accumulation (paper-style) without clipping or decay:
        - If employed: exp_{t+1} = exp_t + (g0 + g1*sigma_j) * exp(-theta * exp_t)
        - If unemployed: exp_{t+1} = exp_t
    """
    exp_t = float(exp_t)
    theta = float(theta)
    if theta <= 0:
        raise ValueError("theta must be > 0")
    if employed_t:
        increment = (g0 + g1 * sigma_j) * np.exp(-theta * exp_t)
        exp_next = exp_t + increment
    else:
        exp_next = exp_t
    return float(exp_next)

def update_experience_vec(
    exp_t: np.ndarray,
    sigma_j: np.ndarray,
    employed_t: np.ndarray,
    g0: float,
    g1: float,
    theta: float,
) -> np.ndarray:
    """
    Vectorized paper-style experience update without clipping; unemployed keep exp.
    Shapes:
      - exp_t: (N,)
      - sigma_j: (N,)
      - employed_t: (N,) boolean or {0,1}
    """
    exp_t = exp_t.astype(float)
    sigma_j = sigma_j.astype(float)
    employed_mask = employed_t.astype(bool)

    exp_next = exp_t.copy()
    exp_next[employed_mask] = exp_t[employed_mask] + (g0 + g1 * sigma_j[employed_mask]) * np.exp(-theta * exp_t[employed_mask])
    return exp_next.astype(float)
