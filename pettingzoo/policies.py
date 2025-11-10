"""
Baseline Policies for Job Market Environment

This module implements simple baseline policies for testing and comparison:
1. Random Policy: Random actions
2. Greedy Policy: Always hire best available worker, fire worst performer
3. No-Screening Policy: Never interview, rely on public signals
4. High-Screening Policy: Always interview before hiring
5. Never-Fire Policy: Never fire workers
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class RandomPolicy:
    """Random action selection (uniform over valid actions with action masking)."""

    def __init__(self, num_workers: int, seed: Optional[int] = None):
        self.num_workers = num_workers
        self.action_space_size = 1 + 3 * num_workers
        self.rng = np.random.RandomState(seed)

    def get_action(self, observation: Dict, agent: str) -> int:
        """Sample random action from valid actions only."""
        # Handle dict observation format (with action masking)
        if isinstance(observation, dict):
            action_mask = observation['action_mask']
            valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
            return self.rng.choice(valid_actions) if valid_actions else 0
        else:
            # Fallback for old format
            return self.rng.randint(0, self.action_space_size)


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
        num_companies: int,
        max_workers_per_company: int,
        ability_dim: int = 1
    ):
        self.num_workers = num_workers
        self.num_companies = num_companies
        self.max_workers_per_company = max_workers_per_company
        self.ability_dim = ability_dim

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

    def _compute_expected_profit(
        self, worker_id: int, parsed_obs: Dict[str, np.ndarray]
    ) -> float:
        """Compute expected profit based on beliefs."""
        beta = 0.5
        belief_mean = parsed_obs['belief_mean'][worker_id, 0]  # Assuming d=1
        experience = parsed_obs['experience'][worker_id]

        return belief_mean + beta * np.log1p(experience)

    def get_action(self, observation, agent: str) -> int:
        """
        Select greedy action.

        Logic:
        1. Compute expected profit for all workers
        2. If have capacity and unemployed workers available: hire best
        3. If at capacity: check if can upgrade (fire worst, hire better)
        4. Otherwise: no-op
        """
        # Handle dict observation format
        if isinstance(observation, dict):
            obs_array = observation['observation']
        else:
            obs_array = observation

        parsed = self._parse_observation(obs_array)

        # Get current workforce
        own_workforce = parsed['own_workforce']
        workforce_ids = [i for i in range(self.num_workers) if own_workforce[i] > 0.5]
        workforce_size = len(workforce_ids)

        # Get unemployed workers
        unemployed_ids = [
            i for i in range(self.num_workers)
            if parsed['employed_by'][i] < 0  # -1 means unemployed
        ]

        # Compute expected profits
        expected_profits = {
            i: self._compute_expected_profit(i, parsed)
            for i in range(self.num_workers)
        }

        # Case 1: Have capacity, hire best unemployed
        if workforce_size < self.max_workers_per_company and unemployed_ids:
            best_unemployed = max(unemployed_ids, key=lambda i: expected_profits[i])
            # Offer action: N+1 to 2N
            return self.num_workers + 1 + best_unemployed

        # Case 2: At capacity, try to upgrade
        if workforce_size >= self.max_workers_per_company and unemployed_ids:
            worst_employee = min(workforce_ids, key=lambda i: expected_profits[i])
            best_unemployed = max(unemployed_ids, key=lambda i: expected_profits[i])

            # Fire if upgrade is beneficial
            if expected_profits[best_unemployed] > expected_profits[worst_employee] + 0.5:  # Margin for costs
                # Fire action: 1 to N
                return 1 + worst_employee

        # Default: no-op
        return 0


class NoScreeningPolicy:
    """
    Greedy policy that never screens (relies on public signals only).

    This tests the value of screening.
    """

    def __init__(
        self,
        num_workers: int,
        num_companies: int,
        max_workers_per_company: int,
        ability_dim: int = 1
    ):
        self.greedy_policy = GreedyPolicy(
            num_workers, num_companies, max_workers_per_company, ability_dim
        )
        self.num_workers = num_workers

    def get_action(self, observation, agent: str) -> int:
        """Get greedy action, but never interview."""
        action = self.greedy_policy.get_action(observation, agent)

        # If action is interview (2N+1 to 3N), convert to no-op
        if action > 2 * self.num_workers:
            return 0  # No-op

        return action


class HighScreeningPolicy:
    """
    Policy that always screens workers before hiring.

    Strategy:
    - Interview unemployed workers to update beliefs
    - Then use greedy hiring based on updated beliefs
    """

    def __init__(
        self,
        num_workers: int,
        num_companies: int,
        max_workers_per_company: int,
        ability_dim: int = 1,
        seed: Optional[int] = None
    ):
        self.greedy_policy = GreedyPolicy(
            num_workers, num_companies, max_workers_per_company, ability_dim
        )
        self.num_workers = num_workers
        self.rng = np.random.RandomState(seed)

        # Track which workers have been screened (per agent)
        self.screened_workers: Dict[str, set] = {}

    def get_action(self, observation, agent: str) -> int:
        """
        With some probability, interview an unscreened worker.
        Otherwise, use greedy policy.
        """
        if agent not in self.screened_workers:
            self.screened_workers[agent] = set()

        # Handle dict observation format
        if isinstance(observation, dict):
            obs_array = observation['observation']
        else:
            obs_array = observation

        parsed = self.greedy_policy._parse_observation(obs_array)

        # Find unemployed workers who haven't been screened
        unemployed_unscreened = [
            i for i in range(self.num_workers)
            if parsed['employed_by'][i] < 0 and i not in self.screened_workers[agent]
        ]

        # 30% chance to screen an unscreened worker
        if unemployed_unscreened and self.rng.rand() < 0.3:
            worker_to_screen = self.rng.choice(unemployed_unscreened)
            self.screened_workers[agent].add(worker_to_screen)
            # Interview action: 2N+1 to 3N
            return 2 * self.num_workers + 1 + worker_to_screen

        # Otherwise use greedy policy
        return self.greedy_policy.get_action(observation, agent)


class NeverFirePolicy:
    """
    Policy that never fires workers (only hires).

    Tests the value of firing decisions.
    """

    def __init__(
        self,
        num_workers: int,
        num_companies: int,
        max_workers_per_company: int,
        ability_dim: int = 1
    ):
        self.greedy_policy = GreedyPolicy(
            num_workers, num_companies, max_workers_per_company, ability_dim
        )
        self.num_workers = num_workers

    def get_action(self, observation, agent: str) -> int:
        """Get action, but never fire."""
        action = self.greedy_policy.get_action(observation, agent)

        # If action is fire (1 to N), convert to no-op
        if 1 <= action <= self.num_workers:
            return 0  # No-op

        return action


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
        num_companies: int,
        max_workers_per_company: int,
        ability_dim: int = 1,
        target_workforce_ratio: float = 0.8,
        screening_threshold: float = 0.5
    ):
        self.num_workers = num_workers
        self.num_companies = num_companies
        self.max_workers_per_company = max_workers_per_company
        self.ability_dim = ability_dim
        self.target_workforce = int(max_workers_per_company * target_workforce_ratio)
        self.screening_threshold = screening_threshold

        self.greedy_policy = GreedyPolicy(
            num_workers, num_companies, max_workers_per_company, ability_dim
        )

    def get_action(self, observation, agent: str) -> int:
        """
        Heuristic decision making.

        Priority:
        1. Fire if workforce too large or low-performing workers
        2. Screen high-uncertainty workers
        3. Hire if below target
        4. No-op
        """
        # Handle dict observation format
        if isinstance(observation, dict):
            obs_array = observation['observation']
        else:
            obs_array = observation

        parsed = self.greedy_policy._parse_observation(obs_array)

        workforce_ids = [i for i in range(self.num_workers) if parsed['own_workforce'][i] > 0.5]
        workforce_size = len(workforce_ids)

        unemployed_ids = [i for i in range(self.num_workers) if parsed['employed_by'][i] < 0]

        # Priority 1: Fire if overcapacity
        if workforce_size > self.max_workers_per_company:
            # Fire random worker (overcapacity shouldn't happen, but handle it)
            return 1 + workforce_ids[0]

        # Priority 2: Screen high-uncertainty unemployed workers
        for i in unemployed_ids:
            variance = parsed['belief_var'][i, 0]  # Assuming d=1
            sigma_hat = parsed['sigma_hat'][i, 0]

            # Screen if: high public signal AND high uncertainty
            if sigma_hat > 0.5 and variance > self.screening_threshold:
                return 2 * self.num_workers + 1 + i

        # Priority 3: Hire if below target
        if workforce_size < self.target_workforce and unemployed_ids:
            # Hire best available
            expected_profits = {
                i: self.greedy_policy._compute_expected_profit(i, parsed)
                for i in unemployed_ids
            }
            best_worker = max(unemployed_ids, key=lambda i: expected_profits[i])
            return self.num_workers + 1 + best_worker

        # Priority 4: Fire worst performer if can upgrade
        if workforce_size > 0 and unemployed_ids:
            expected_profits_all = {
                i: self.greedy_policy._compute_expected_profit(i, parsed)
                for i in range(self.num_workers)
            }

            worst_employee = min(workforce_ids, key=lambda i: expected_profits_all[i])
            best_unemployed = max(unemployed_ids, key=lambda i: expected_profits_all[i])

            if expected_profits_all[best_unemployed] > expected_profits_all[worst_employee] + 0.5:
                return 1 + worst_employee

        # Default: no-op
        return 0


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
    num_companies = env_config['num_companies']
    max_workers = env_config['max_workers_per_company']
    ability_dim = env_config.get('ability_dim', 1)

    if policy_name == 'random':
        return RandomPolicy(num_workers)

    elif policy_name == 'greedy':
        return GreedyPolicy(num_workers, num_companies, max_workers, ability_dim)

    elif policy_name == 'no_screening':
        return NoScreeningPolicy(num_workers, num_companies, max_workers, ability_dim)

    elif policy_name == 'high_screening':
        return HighScreeningPolicy(num_workers, num_companies, max_workers, ability_dim)

    elif policy_name == 'never_fire':
        return NeverFirePolicy(num_workers, num_companies, max_workers, ability_dim)

    elif policy_name == 'heuristic':
        return HeuristicPolicy(num_workers, num_companies, max_workers, ability_dim)

    else:
        raise ValueError(f"Unknown policy: {policy_name}")
