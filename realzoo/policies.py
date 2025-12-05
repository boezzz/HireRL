"""
Baseline Policies for Job Market Environment

This module implements simple baseline policies for testing and comparison:
1. Random Policy: Random actions (80% greedy, 20% random)
2. Greedy Policy: Always hire best available worker, fire worst performer
3. stable macthing?
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


def _quantize_action(cost: float, action_mode: str, cost_levels: np.ndarray, max_cost: float):
    """Map target cost to env action."""
    cost_clamped = float(np.clip(cost, 0.0, max_cost))
    if action_mode == "discrete":
        idx = int(np.argmin(np.abs(cost_levels - cost_clamped)))
        return int(idx)
    return cost_clamped


class RandomPolicy:
    """Randomly sample an interview cost (continuous or discrete)."""

    def __init__(
        self,
        num_workers: int,
        max_interview_cost: float = 2.0,
        num_interview_cost_levels: int = 5,
        action_mode: str = "continuous",
        seed: Optional[int] = None,
    ):
        self.num_workers = num_workers
        self.max_interview_cost = float(max_interview_cost)
        self.action_mode = action_mode.lower()
        self.cost_levels = np.linspace(
            0.0,
            self.max_interview_cost,
            max(2, num_interview_cost_levels),
            dtype=np.float32,
        )
        self.rng = np.random.RandomState(seed)

    def get_action(self, observation: Dict, agent: str) -> float:
        """Sample a random cost or discrete level."""
        if self.action_mode == "discrete":
            return int(self.rng.randint(0, len(self.cost_levels)))
        return float(self.rng.rand() * self.max_interview_cost)


class GreedyPolicy:
    """
    Greedy hiring and firing based on expected profit.

    Strategy:
    - If workforce has capacity: hire best available unemployed worker
    - If workforce is full: fire worst worker if can replace with better
    - Use firm's beliefs to estimate expected profit
    """

    def __init__(
        self,
        num_workers: int,
        ability_dim: int = 1,
        max_interview_cost: float = 2.0,
        num_interview_cost_levels: int = 5,
        action_mode: str = "continuous",
    ):
        self.num_workers = num_workers
        self.ability_dim = ability_dim
        self.max_interview_cost = float(max_interview_cost)
        self.action_mode = action_mode.lower()
        self.cost_levels = np.linspace(
            0.0,
            self.max_interview_cost,
            max(2, num_interview_cost_levels),
            dtype=np.float32,
        )

    def _compute_expected_profit(self, worker_id: int, parsed: Dict[str, np.ndarray]) -> float:
        """
        Simple heuristic for expected profit based on beliefs, wages, and experience.
        """
        expected_signal = parsed['belief_mean'][worker_id, 0]
        wage = parsed['wages'][worker_id]
        experience = parsed['experience'][worker_id]
        tenure = parsed['tenure'][worker_id]
        # Reward high signals, penalize wages, and reward workers with low tenure (less costly)
        return float(expected_signal - wage + 0.1 * experience - 0.05 * tenure)

    def _parse_observation(
        self, obs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Parse observation vector into components.

        Observation structure:
        - sigma_hat: (N * d)
        - experience: (N,)
        - tenure: (N,)
        - employed_by: (N,)
        - wages: (N,)
        - belief_mean: (N * d)
        - belief_var: (N * d)
        - own_workforce: (N,)
        - own_profit: (1,)
        """
        N = self.num_workers
        d = self.ability_dim

        idx = 0
        sigma_hat = obs[idx:idx + N*d].reshape(N, d)
        idx += N * d

        experience = obs[idx:idx + N]
        idx += N

        tenure = obs[idx:idx + N]
        idx += N

        employed_by = obs[idx:idx + N]
        idx += N

        wages = obs[idx:idx + N]
        idx += N

        belief_mean = obs[idx:idx + N*d].reshape(N, d)
        idx += N * d

        belief_var = obs[idx:idx + N*d].reshape(N, d)
        idx += N * d

        own_workforce = obs[idx:idx + N]
        idx += N

        own_profit = obs[idx]

        return {
            'sigma_hat': sigma_hat,
            'experience': experience,
            'tenure': tenure,
            'employed_by': employed_by,
            'wages': wages,
            'belief_mean': belief_mean,
            'belief_var': belief_var,
            'own_workforce': own_workforce,
            'own_profit': own_profit
        }


    def get_action(self, observation, agent: str) -> float:
        """
        Select a greedy interview-cost action.

        Logic:
        1. Parse observation and compute expected profit proxies for workers.
        2. Look at unemployed workers: if there is high expected surplus, invest
           more in screening; otherwise, invest little or nothing.
        3. Map this intensity choice into a continuous cost in [0, max_interview_cost].
        """
        # Handle dict observation format and extract action_mask
        if isinstance(observation, dict):
            obs_array = observation['observation']
            action_mask = observation.get('action_mask', None)
        else:
            obs_array = observation
            action_mask = None

        parsed = self._parse_observation(obs_array)

        # Identify unemployed workers (employed_by < 0)
        unemployed_ids = [
            i for i in range(self.num_workers)
            if parsed['employed_by'][i] < 0
        ]

        if unemployed_ids:
            expected_profits = {
                i: self._compute_expected_profit(i, parsed)
                for i in unemployed_ids
            }
            max_exp_profit = max(expected_profits.values())
        else:
            max_exp_profit = 0.0

        if max_exp_profit <= 0.0:
            cost_choice = 0.0
        elif max_exp_profit < 1.0:
            cost_choice = 0.5 * self.max_interview_cost
        else:
            cost_choice = self.max_interview_cost

        return _quantize_action(cost_choice, self.action_mode, self.cost_levels, self.max_interview_cost)


class NoScreeningPolicy:
    """
    Policy that never spends on interviews (relies on public signals only).
    """

    def __init__(
        self,
        num_workers: int,
        ability_dim: int = 1,
        max_interview_cost: float = 2.0,
        num_interview_cost_levels: int = 5,
        action_mode: str = "continuous",
    ):
        self.num_workers = num_workers
        self.ability_dim = ability_dim
        self.max_interview_cost = float(max_interview_cost)
        self.action_mode = action_mode.lower()
        self.cost_levels = np.linspace(
            0.0,
            self.max_interview_cost,
            max(2, num_interview_cost_levels),
            dtype=np.float32,
        )

    def get_action(self, observation, agent: str) -> float:
        return _quantize_action(0.0, self.action_mode, self.cost_levels, self.max_interview_cost)


class HighScreeningPolicy:
    """
    Policy that frequently allocates the maximum interview cost.
    """

    def __init__(
        self,
        num_workers: int,
        ability_dim: int = 1,
        max_interview_cost: float = 2.0,
        num_interview_cost_levels: int = 5,
        action_mode: str = "continuous",
        seed: Optional[int] = None
    ):
        self.greedy_policy = GreedyPolicy(
            num_workers, ability_dim, max_interview_cost, num_interview_cost_levels, action_mode
        )
        self.max_interview_cost = float(max_interview_cost)
        self.action_mode = action_mode.lower()
        self.cost_levels = np.linspace(
            0.0,
            self.max_interview_cost,
            max(2, num_interview_cost_levels),
            dtype=np.float32,
        )
        self.rng = np.random.RandomState(seed)

    def get_action(self, observation, agent: str) -> float:
        if self.rng.rand() < 0.3:
            return _quantize_action(self.max_interview_cost, self.action_mode, self.cost_levels, self.max_interview_cost)
        return self.greedy_policy.get_action(observation, agent)


class NeverFirePolicy:
    """
    Policy that never fires workers (only hires).

    Tests the value of firing decisions.
    """

    def __init__(
        self,
        num_workers: int,
        ability_dim: int = 1,
        max_interview_cost: float = 2.0,
        num_interview_cost_levels: int = 5,
        action_mode: str = "continuous",
    ):
        self.greedy_policy = GreedyPolicy(
            num_workers, ability_dim, max_interview_cost, num_interview_cost_levels, action_mode
        )
        self.num_workers = num_workers

    def get_action(self, observation, agent: str) -> float:
        """Get action; firing is now handled by the environment's rule."""
        return self.greedy_policy.get_action(observation, agent)


class HeuristicPolicy:
    """
    Heuristic policy combining multiple strategies.

    Rules:
    1. Screen workers with high public signal but high uncertainty
    2. Hire workers with good expected profit
    3. Fire workers with low realized profit
    4. Maintain target workforce size
    """

    def __init__(
        self,
        num_workers: int,
        ability_dim: int = 1,
        max_interview_cost: float = 2.0,
        num_interview_cost_levels: int = 5,
        action_mode: str = "continuous",
        target_workforce_ratio: float = 0.8,
        screening_threshold: float = 0.5
    ):
        self.num_workers = num_workers
        self.ability_dim = ability_dim
        self.max_interview_cost = float(max_interview_cost)
        self.action_mode = action_mode.lower()
        self.cost_levels = np.linspace(
            0.0,
            self.max_interview_cost,
            max(2, num_interview_cost_levels),
            dtype=np.float32,
        )
        self.target_workforce = target_workforce_ratio
        self.screening_threshold = screening_threshold

        self.greedy_policy = GreedyPolicy(
            num_workers, ability_dim, max_interview_cost, num_interview_cost_levels, action_mode
        )

    def get_action(self, observation, agent: str) -> float:
        """
        Heuristic decision making over interview cost levels.
        """
        if isinstance(observation, dict):
            obs_array = observation['observation']
        else:
            obs_array = observation

        parsed = self.greedy_policy._parse_observation(obs_array)

        unemployed_ids = [i for i in range(self.num_workers) if parsed['employed_by'][i] < 0]

        high_uncertainty_candidates = []
        for i in unemployed_ids:
            variance = parsed['belief_var'][i, 0]  # Assuming d=1
            sigma_hat = parsed['sigma_hat'][i, 0]
            if sigma_hat > 0.5 and variance > self.screening_threshold:
                high_uncertainty_candidates.append(i)

        if high_uncertainty_candidates:
            if len(high_uncertainty_candidates) >= 2:
                return _quantize_action(self.max_interview_cost, self.action_mode, self.cost_levels, self.max_interview_cost)
            mid_cost = 0.5 * self.max_interview_cost
            return _quantize_action(mid_cost, self.action_mode, self.cost_levels, self.max_interview_cost)

        return self.greedy_policy.get_action(observation, agent)


def create_policy(policy_name: str, env_config: Dict) -> object:
    """
    Factory function to create policies by name.

    Args:
        policy_name: One of ['random', 'greedy', 'no_screening', 'high_screening',
                              'never_fire', 'heuristic']
        env_config: Environment configuration dict with keys:
                    num_workers, num_companies, max_workers_per_company, ability_dim

    Returns:
        Policy object with get_action(observation, agent) method
    """
    num_workers = env_config['num_workers']
    ability_dim = env_config.get('ability_dim', 1)
    max_interview_cost = env_config.get('max_interview_cost', 2.0)
    num_cost_levels = env_config.get('num_interview_cost_levels', 5)
    action_mode = env_config.get('action_mode', 'continuous')

    if policy_name == 'random':
        return RandomPolicy(num_workers, max_interview_cost, num_cost_levels, action_mode)

    elif policy_name == 'greedy':
        return GreedyPolicy(num_workers, ability_dim, max_interview_cost, num_cost_levels, action_mode)

    elif policy_name == 'no_screening':
        return NoScreeningPolicy(num_workers, ability_dim, max_interview_cost, num_cost_levels, action_mode)

    elif policy_name == 'high_screening':
        return HighScreeningPolicy(num_workers, ability_dim, max_interview_cost, num_cost_levels, action_mode)

    elif policy_name == 'never_fire':
        return NeverFirePolicy(num_workers, ability_dim, max_interview_cost, num_cost_levels, action_mode)

    elif policy_name == 'heuristic':
        return HeuristicPolicy(num_workers, ability_dim, max_interview_cost, num_cost_levels, action_mode)

    else:
        raise ValueError(f"Unknown policy: {policy_name}")
