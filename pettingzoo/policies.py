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
        ability_dim: int = 1,
    ):
        self.num_workers = num_workers
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


    def get_action(self, observation, agent: str) -> int:
        """
        Select a greedy interview-cost action.

        In the current environment, firms do not directly choose which worker
        to hire or fire. Instead, each firm's discrete action controls how much
        to invest in interviewing (screening) the worker assigned to it by the
        deterministic interview mechanism in the environment.

        Logic:
        1. Parse observation and compute expected profit proxies for workers.
        2. Look at unemployed workers: if there is high expected surplus, invest
           more in screening; otherwise, invest little or nothing.
        3. Map this intensity choice into a discrete action using the action_mask:
           - smallest valid index   -> interpreted as "no-op" / lowest cost
           - largest valid index    -> interpreted as "highest cost"
           - a middle valid index   -> medium cost (if available)
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

        # Determine valid discrete actions
        if action_mask is not None:
            valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
        else:
            # Fallback: assume a small fixed action space: {0,1,2,3}
            valid_actions = list(range(4))

        if not valid_actions:
            return 0

        valid_actions_sorted = sorted(valid_actions)
        no_op_action = valid_actions_sorted[0]
        highest_cost_action = valid_actions_sorted[-1]
        if len(valid_actions_sorted) >= 3:
            mid_idx = len(valid_actions_sorted) // 2
            mid_cost_action = valid_actions_sorted[mid_idx]
        else:
            mid_cost_action = highest_cost_action

        # Heuristic mapping from expected profit to interview intensity
        if max_exp_profit <= 0.0:
            # No promising unemployed workers: do nothing / lowest cost
            return no_op_action
        elif max_exp_profit < 1.0:
            # Moderate opportunity: invest medium screening cost
            return mid_cost_action
        else:
            # High expected surplus: invest highest available screening cost
            return highest_cost_action


class NoScreeningPolicy:
    """
    Greedy policy that never screens (relies on public signals only).

    This tests the value of screening.
    """

    def __init__(
        self,
        num_workers: int,
        ability_dim: int = 1
    ):
        self.greedy_policy = GreedyPolicy(
            num_workers, ability_dim
        )
        self.num_workers = num_workers

    def get_action(self, observation, agent: str) -> int:
        """Never invest in screening: always choose the lowest-cost/no-op action."""
        if isinstance(observation, dict):
            action_mask = observation.get('action_mask', None)
        else:
            action_mask = None

        if action_mask is not None:
            valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
            if valid_actions:
                return min(valid_actions)
            return 0
        # Fallback: assume action 0 is no-op
        return 0


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
        ability_dim: int = 1,
        seed: Optional[int] = None
    ):
        self.greedy_policy = GreedyPolicy(
            num_workers, ability_dim
        )
        self.num_workers = num_workers
        self.rng = np.random.RandomState(seed)

    def get_action(self, observation, agent: str) -> int:
        """
        With some probability, choose a high screening cost.
        Otherwise, use the greedy cost-based policy.
        """
        # Handle dict observation format
        if isinstance(observation, dict):
            action_mask = observation.get('action_mask', None)
        else:
            action_mask = None

        if action_mask is not None:
            valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
        else:
            valid_actions = list(range(4))

        if not valid_actions:
            return 0

        valid_actions_sorted = sorted(valid_actions)
        no_op_action = valid_actions_sorted[0]
        highest_cost_action = valid_actions_sorted[-1]

        # 30% chance to choose the highest screening cost
        if self.rng.rand() < 0.3:
            return highest_cost_action

        # Otherwise, fall back to greedy cost choice
        return self.greedy_policy.get_action(observation, agent)


class NeverFirePolicy:
    """
    Policy that never fires workers (only hires).

    Tests the value of firing decisions.
    """

    def __init__(
        self,
        num_workers: int,
        ability_dim: int = 1
    ):
        self.greedy_policy = GreedyPolicy(
            num_workers, ability_dim
        )
        self.num_workers = num_workers

    def get_action(self, observation, agent: str) -> int:
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
        target_workforce_ratio: float = 0.8,
        screening_threshold: float = 0.5
    ):
        self.num_workers = num_workers
        self.ability_dim = ability_dim
        self.target_workforce = target_workforce_ratio
        self.screening_threshold = screening_threshold

        self.greedy_policy = GreedyPolicy(
            num_workers, ability_dim
        )

    def get_action(self, observation, agent: str) -> int:
        """
        Heuristic decision making over interview cost levels.

        Priority:
        1. If there exist unemployed workers with high public signal and high
           uncertainty, invest in higher-cost screening.
        2. Otherwise, use the greedy cost-based policy.
        """
        # Handle dict observation format
        if isinstance(observation, dict):
            obs_array = observation['observation']
            action_mask = observation.get('action_mask', None)
        else:
            obs_array = observation
            action_mask = None

        parsed = self.greedy_policy._parse_observation(obs_array)

        unemployed_ids = [i for i in range(self.num_workers) if parsed['employed_by'][i] < 0]

        # Determine candidate workers for extra screening
        high_uncertainty_candidates = []
        for i in unemployed_ids:
            variance = parsed['belief_var'][i, 0]  # Assuming d=1
            sigma_hat = parsed['sigma_hat'][i, 0]
            if sigma_hat > 0.5 and variance > self.screening_threshold:
                high_uncertainty_candidates.append(i)

        # Determine valid actions
        if action_mask is not None:
            valid_actions = [i for i, v in enumerate(action_mask) if v == 1]
        else:
            valid_actions = list(range(4))

        if not valid_actions:
            return 0

        valid_actions_sorted = sorted(valid_actions)
        no_op_action = valid_actions_sorted[0]
        highest_cost_action = valid_actions_sorted[-1]
        if len(valid_actions_sorted) >= 3:
            mid_idx = len(valid_actions_sorted) // 2
            mid_cost_action = valid_actions_sorted[mid_idx]
        else:
            mid_cost_action = highest_cost_action

        if high_uncertainty_candidates:
            # If there are promising but uncertain workers, use a higher screening cost
            return mid_cost_action if len(high_uncertainty_candidates) == 1 else highest_cost_action

        # Otherwise use greedy cost-based policy
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
    num_companies = env_config['num_companies']
    max_workers = env_config['max_workers_per_company']
    ability_dim = env_config.get('ability_dim', 1)

    if policy_name == 'random':
        return RandomPolicy(num_workers)

    elif policy_name == 'greedy':
        return GreedyPolicy(num_workers, ability_dim)

    elif policy_name == 'no_screening':
        return NoScreeningPolicy(num_workers, ability_dim)

    elif policy_name == 'high_screening':
        return HighScreeningPolicy(num_workers, ability_dim)

    elif policy_name == 'never_fire':
        return NeverFirePolicy(num_workers, ability_dim)

    elif policy_name == 'heuristic':
        return HeuristicPolicy(num_workers, ability_dim)

    else:
        raise ValueError(f"Unknown policy: {policy_name}")
