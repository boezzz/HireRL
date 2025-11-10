"""
helpers
- Logging and tracking
- Visualization helpers
- Statistical analysis
- Environment verification
"""

import numpy as np
from typing import Dict, List, Any, Optional
import json


class EpisodeLogger:
    """
    Logs episode data for analysis.

    Tracks:
    - Rewards over time
    - Workforce sizes
    - Unemployment rates
    - Wages
    - Screening decisions
    """

    def __init__(self, agents: List[str]):
        self.agents = agents
        self.reset()

    def reset(self):
        """Reset logger for new episode."""
        self.timesteps = []
        self.rewards = {agent: [] for agent in self.agents}
        self.workforce_sizes = {agent: [] for agent in self.agents}
        self.unemployment_rates = []
        self.average_wages = []
        self.actions_taken = {agent: [] for agent in self.agents}

    def log_step(
        self,
        timestep: int,
        rewards: Dict[str, float],
        infos: Dict[str, dict],
        actions: Dict[str, int]
    ):
        """Log data from one timestep."""
        self.timesteps.append(timestep)

        for agent in self.agents:
            self.rewards[agent].append(rewards[agent])
            self.workforce_sizes[agent].append(infos[agent]['workforce_size'])
            self.actions_taken[agent].append(actions[agent])

        # Log market-wide stats (same for all agents)
        first_agent = self.agents[0]
        self.unemployment_rates.append(infos[first_agent]['unemployment_rate'])
        self.average_wages.append(infos[first_agent]['avg_wage'])

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for episode."""
        summary = {
            'total_timesteps': len(self.timesteps),
            'agents': {}
        }

        for agent in self.agents:
            summary['agents'][agent] = {
                'total_reward': sum(self.rewards[agent]),
                'mean_reward': np.mean(self.rewards[agent]) if self.rewards[agent] else 0,
                'mean_workforce_size': np.mean(self.workforce_sizes[agent]) if self.workforce_sizes[agent] else 0,
                'final_workforce_size': self.workforce_sizes[agent][-1] if self.workforce_sizes[agent] else 0,
            }

        summary['market'] = {
            'final_unemployment_rate': self.unemployment_rates[-1] if self.unemployment_rates else 0,
            'mean_unemployment_rate': np.mean(self.unemployment_rates) if self.unemployment_rates else 0,
            'final_average_wage': self.average_wages[-1] if self.average_wages else 0,
            'mean_average_wage': np.mean(self.average_wages) if self.average_wages else 0,
        }

        return summary

    def save_to_json(self, filepath: str):
        """Save episode data to JSON file."""
        data = {
            'timesteps': self.timesteps,
            'rewards': {agent: list(r) for agent, r in self.rewards.items()},
            'workforce_sizes': {agent: list(w) for agent, w in self.workforce_sizes.items()},
            'unemployment_rates': self.unemployment_rates,
            'average_wages': self.average_wages,
            'summary': self.get_summary()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


def compute_market_efficiency(
    total_matches: int,
    num_workers: int,
    num_companies: int,
    max_workers_per_company: int
) -> float:
    """
    Compute market efficiency: fraction of maximum possible matches.

    Maximum matches = min(num_workers, num_companies * max_workers_per_company)

    Args:
        total_matches: Number of workers currently employed
        num_workers: Total workers
        num_companies: Total companies
        max_workers_per_company: Capacity per company

    Returns:
        Efficiency ratio in [0, 1]
    """
    max_possible = min(num_workers, num_companies * max_workers_per_company)
    return total_matches / max(max_possible, 1)


def analyze_screening_value(
    logger_with_screening: EpisodeLogger,
    logger_without_screening: EpisodeLogger
) -> Dict[str, float]:
    """
    Analyze value of screening by comparing episodes.

    Args:
        logger_with_screening: Episode log with screening
        logger_without_screening: Episode log without screening

    Returns:
        Dictionary with comparison metrics
    """
    summary_with = logger_with_screening.get_summary()
    summary_without = logger_without_screening.get_summary()

    # Compute aggregate rewards
    total_reward_with = sum(
        summary_with['agents'][agent]['total_reward']
        for agent in summary_with['agents']
    )
    total_reward_without = sum(
        summary_without['agents'][agent]['total_reward']
        for agent in summary_without['agents']
    )

    screening_value = total_reward_with - total_reward_without

    return {
        'total_reward_with_screening': total_reward_with,
        'total_reward_without_screening': total_reward_without,
        'screening_value': screening_value,
        'screening_value_pct': (screening_value / abs(total_reward_without)) * 100 if total_reward_without != 0 else 0,
        'unemployment_diff': (
            summary_with['market']['final_unemployment_rate'] -
            summary_without['market']['final_unemployment_rate']
        )
    }


def verify_environment(env, num_steps: int = 10, verbose: bool = True):
    """
    Verify environment correctness.

    DEPRECATED: Use pettingzoo.test.parallel_api_test instead for comprehensive testing.

    This function provides basic checks for:
    - Reset works
    - Step works with random actions
    - Observations are in declared space
    - Rewards are numeric
    - Info dicts contain required keys

    Args:
        env: Environment instance
        num_steps: Number of steps to test
        verbose: Print detailed output

    Returns:
        True if all checks pass, False otherwise
    """
    if verbose:
        print("Verifying environment (use pettingzoo.test.parallel_api_test for full compliance)...")

    try:
        # Test reset
        observations, infos = env.reset()

        if verbose:
            print("✓ Reset successful")

        # Check observations are in space
        for agent in env.agents:
            obs = observations[agent]
            expected_space = env.observation_space(agent)

            if not expected_space.contains(obs):
                print(f"✗ Observation not in space for {agent}")
                return False

        if verbose:
            print(f"✓ Observations in declared spaces")

        # Test steps with random actions
        for step in range(num_steps):
            actions = {
                agent: env.action_space(agent).sample()
                for agent in env.agents
            }

            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Check rewards
            for agent in env.agents:
                if not isinstance(rewards[agent], (int, float)):
                    print(f"✗ Invalid reward type for {agent}: {type(rewards[agent])}")
                    return False

            # Check infos
            required_info_keys = ['workforce_size', 'total_profit', 'unemployment_rate', 'avg_wage', 'timestep']
            for agent in env.agents:
                for key in required_info_keys:
                    if key not in infos[agent]:
                        print(f"✗ Missing info key '{key}' for {agent}")
                        return False

        if verbose:
            print(f"✓ Successfully ran {num_steps} steps")

        if verbose:
            print("✓ All checks passed!")

        return True

    except Exception as e:
        print(f"✗ Error during verification: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_environment_info(env):
    """Print detailed information about environment configuration."""
    print("\n" + "="*60)
    print("ENVIRONMENT CONFIGURATION")
    print("="*60)
    print(f"Number of Companies: {env.num_companies}")
    print(f"Number of Workers: {env.num_workers}")
    print(f"Ability Dimension: {env.ability_dim}")
    print(f"Max Workers per Company: {env.max_workers_per_company}")
    print(f"")
    print(f"Worker Dynamics:")
    print(f"  Tenure signal growth (γ): {env.gamma}")
    print(f"  Base experience growth (g0): {env.g0}")
    print(f"  Ability-dependent growth (g1): {env.g1}")
    print(f"")
    print(f"Costs:")
    print(f"  Firing cost: {env.base_firing_cost}")
    print(f"  Hiring cost: {env.base_hiring_cost}")
    print(f"  Screening cost: {env.base_screening_cost}")
    print(f"")
    print(f"Market Parameters:")
    print(f"  Worker bargaining power: {env.worker_bargaining_power}")
    print(f"  Max timesteps: {env.max_timesteps}")
    print(f"")
    print(f"Spaces:")
    print(f"  Action space size: {env.action_size}")
    obs_space = env.observation_space(env.agents[0])
    if hasattr(obs_space, 'spaces'):  # Dict space
        print(f"  Observation space: Dict with keys {list(obs_space.spaces.keys())}")
        print(f"    - observation shape: {obs_space.spaces['observation'].shape}")
        print(f"    - action_mask shape: {obs_space.spaces['action_mask'].shape}")
    else:
        print(f"  Observation space shape: {obs_space.shape}")
    print("="*60 + "\n")


def compute_gini_coefficient(wages: np.ndarray) -> float:
    """
    Compute Gini coefficient for wage inequality.

    Gini = 0: Perfect equality
    Gini = 1: Perfect inequality

    Args:
        wages: Array of wages

    Returns:
        Gini coefficient
    """
    # Filter out zero wages (unemployed)
    employed_wages = wages[wages > 0]

    if len(employed_wages) == 0:
        return 0.0

    # Sort wages
    sorted_wages = np.sort(employed_wages)
    n = len(sorted_wages)

    # Compute Gini
    cumsum = np.cumsum(sorted_wages)
    gini = (2 * np.sum((np.arange(1, n+1)) * sorted_wages)) / (n * cumsum[-1]) - (n + 1) / n

    return float(gini)


class PerformanceMetrics:
    """
    Compute comprehensive performance metrics for job market.

    Metrics include:
    - Efficiency: Match rate, unemployment
    - Equity: Wage inequality (Gini)
    - Stability: Quit rate, firing rate
    - Profitability: Total surplus, firm profits
    """

    def __init__(self):
        self.history = []

    def compute_metrics(
        self,
        worker_pool,
        company_profits: Dict[str, List[float]],
        timestep: int
    ) -> Dict[str, float]:
        """
        Compute metrics for current state.

        Args:
            worker_pool: WorkerPool instance
            company_profits: Dict of profit histories
            timestep: Current timestep

        Returns:
            Dictionary of metrics
        """
        public_state = worker_pool.get_public_state()

        unemployment_rate = worker_pool.get_unemployment_rate()
        avg_wage = worker_pool.get_average_wage()
        gini = compute_gini_coefficient(public_state['wages'])

        total_matches = np.sum(public_state['employed_by'] >= 0)

        # Total surplus (sum of all profits + wages)
        total_firm_profit = sum(
            profits[-1] if profits else 0
            for profits in company_profits.values()
        )
        total_wages = np.sum(public_state['wages'])
        total_surplus = total_firm_profit + total_wages

        metrics = {
            'timestep': timestep,
            'unemployment_rate': unemployment_rate,
            'total_matches': int(total_matches),
            'avg_wage': avg_wage,
            'wage_gini': gini,
            'total_firm_profit': total_firm_profit,
            'total_wages': total_wages,
            'total_surplus': total_surplus,
        }

        self.history.append(metrics)
        return metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all timesteps."""
        if not self.history:
            return {}

        return {
            'mean_unemployment_rate': np.mean([m['unemployment_rate'] for m in self.history]),
            'mean_matches': np.mean([m['total_matches'] for m in self.history]),
            'mean_wage': np.mean([m['avg_wage'] for m in self.history]),
            'mean_wage_gini': np.mean([m['wage_gini'] for m in self.history]),
            'total_surplus': sum([m['total_surplus'] for m in self.history]),
        }
