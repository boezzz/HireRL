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
        
        # Track interview history: True if we've ever interviewed this worker
        self.has_interviewed = np.zeros(num_workers, dtype=bool)


    def initialize_from_interview_signal(
        self,
        worker_id: int,
        psi_interview: float,
        sigma_hat_public: float,
        cost: float,
        delta0_sq: float,
        lambda_param: float,
        nu_hat_public: float = 1.0,
    ) -> None:
        """Initialize firm's belief using Bayesian combination of public and interview signals.

        Implements the specification:
        tilde_σ_{ij,t} = [δ²(c) · σ̂_{j,t} + ν̂_t · ψ_{ij,t}] / [δ²(c) + ν̂_t]
        
        Where:
        - ψ_{ij,t} is the interview signal (psi_interview)
        - σ̂_{j,t} is the public signal (sigma_hat_public)  
        - δ²(c) = δ₀² exp(-λc) is interview noise variance
        - ν̂_t is public signal variance (nu_hat_public)

        Args:
            worker_id: Index of the worker j.
            psi_interview: Interview signal ψ_{ij,t} = σ_j + η_{ij,t}.
            sigma_hat_public: Public signal σ̂_{j,t}.
            cost: Interview cost c_{ij,t}.
            delta0_sq: Base interview noise variance δ₀².
            lambda_param: Cost effectiveness parameter λ.
            nu_hat_public: Public signal variance ν̂_t.
        """
        # Handle zero cost case per specification:
        # - If never interviewed: belief = public signal σ̂_{j,t}
        # - If previously interviewed: belief remains at previous tilde_σ_{ij,t-1}
        if cost <= 0:
            if not self.has_interviewed[worker_id]:
                # Never interviewed: use public signal
                self.belief_mean[worker_id, :] = float(sigma_hat_public)
                self.belief_var[worker_id, :] = float(nu_hat_public)
            # If previously interviewed: belief stays unchanged (do nothing)
            return

        # Compute interview noise variance: δ²(c) = δ₀² exp(-λc)
        delta_sq = delta0_sq * np.exp(-lambda_param * cost)
        
        # Bayesian combination: weighted average of public and interview signals
        numerator = delta_sq * sigma_hat_public + nu_hat_public * psi_interview
        denominator = delta_sq + nu_hat_public
        
        belief_mean = numerator / denominator
        
        # Posterior variance: 1 / (1/δ²(c) + 1/ν̂_t)
        belief_variance = 1.0 / (1.0/delta_sq + 1.0/nu_hat_public)
        
        self.belief_mean[worker_id, :] = float(belief_mean)
        self.belief_var[worker_id, :] = float(belief_variance)
        
        # Mark this worker as interviewed
        self.has_interviewed[worker_id] = True


    def get_belief(self, worker_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get current belief about a worker.
        在当前的 2after_hiring_update.py 中，这个函数主要是：

        ✔ 给 其它模块 访问最新 posterior belief 的接口

        Returns:
            Tuple of (mean, variance)
        """
        return self.belief_mean[worker_id], self.belief_var[worker_id]
    
    def get_effective_belief(self, worker_id: int, public_signal: float, 
                           public_variance: float = 1.0) -> float:
        """
        Get the effective belief for decision-making.
        
        Returns:
        - Private belief if worker was previously interviewed 
        - Public signal if worker was never interviewed
        
        Args:
            worker_id: Worker to get belief for
            public_signal: Current public signal σ̂_{j,t}
            public_variance: Variance of public signal
            
        Returns:
            Effective belief mean for this worker
        """
        if self.has_interviewed[worker_id]:
            # Use stored private belief from previous interview
            return float(self.belief_mean[worker_id, 0])
        else:
            # Never interviewed - use public signal
            return float(public_signal)
    
    def initialize_default_belief(self, worker_id: int, public_signal: float, 
                                 public_variance: float = 1.0) -> None:
        """
        Initialize default belief based on public signal (without marking as interviewed).
        
        This is used during environment setup, not after actual interviews.
        
        Args:
            worker_id: Worker to initialize belief for
            public_signal: Current public signal σ̂_{j,t}
            public_variance: Variance of public signal
        """
        self.belief_mean[worker_id, :] = float(public_signal)
        self.belief_var[worker_id, :] = float(public_variance)
        # Explicitly do NOT mark as interviewed - this is just default initialization

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
        Recursive Bayesian belief update using performance history:
            \tilde{σ}_{ij,t+1} = (1 - v_x) * \tilde{σ}_{ij,t} + v_x * p_{ij,t}
        
        This creates a convex combination of previous belief and new performance,
        gradually converging to true ability σ_j as experience accumulates.
        
        Args:
            worker_id: Worker being updated
            p_ijt: Current period profit observation
            tilde_sigma_interview: Original interview signal (unused in recursive update)
            exp_t: Current experience level (affects update weight)
            delta_interview_sq: Interview signal variance
            delta_eps_sq: Profit noise variance
        """
        exp_clamped = max(0.0, float(exp_t))
        K1 = float(delta_interview_sq) / float(delta_interview_sq + delta_eps_sq) if (delta_interview_sq + delta_eps_sq) > 0 else 0.0
        vx = (exp_clamped * K1) / (1.0 + (exp_clamped - 1.0) * K1) if K1 > 0 else 0.0
        
        # Recursive update: use current belief, not original interview signal
        current_belief = float(self.belief_mean[worker_id, 0])
        new_belief = (1.0 - vx) * current_belief + vx * float(p_ijt)
        self.belief_mean[worker_id, 0] = new_belief


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
