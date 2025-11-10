"""
tests different baseline policies in the job market environment:
1. Random policy
2. Greedy policy (no screening)
3. High screening policy
4. Heuristic policy
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pettingzoo'))

import numpy as np
from typing import Dict, List

from hirerl import JobMarketEnv
from policies import create_policy


def run_episode(env: JobMarketEnv, policy, max_steps: int = 100, render: bool = False):
    """
    Run one episode with given policy.

    Args:
        env: Environment instance
        policy: Policy object with get_action(obs, agent) method
        max_steps: Maximum episode length
        render: Whether to render

    Returns:
        Dictionary with episode statistics
    """
    observations, infos = env.reset()

    total_rewards = {agent: 0.0 for agent in env.agents}
    episode_length = 0

    workforce_history = {agent: [] for agent in env.agents}
    profit_history = {agent: [] for agent in env.agents}

    for step in range(max_steps):
        # Get actions from policy
        actions = {}
        for agent in env.agents:
            if hasattr(policy, '__getitem__'):  # Dict of policies
                actions[agent] = policy[agent].get_action(observations[agent], agent)
            else:  # Single policy for all agents
                actions[agent] = policy.get_action(observations[agent], agent)

        # Step environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Track statistics
        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
            workforce_history[agent].append(infos[agent]['workforce_size'])
            profit_history[agent].append(rewards[agent])

        episode_length += 1

        if render and step % 10 == 0:
            env.render()

        # Check if done
        if all(terminations.values()) or all(truncations.values()):
            break

    return {
        'total_rewards': total_rewards,
        'episode_length': episode_length,
        'final_unemployment': infos[env.agents[0]]['unemployment_rate'],
        'final_avg_wage': infos[env.agents[0]]['avg_wage'],
        'workforce_history': workforce_history,
        'profit_history': profit_history
    }


def compare_policies(
    env_config: Dict,
    policy_names: List[str],
    num_episodes: int = 5,
    max_steps: int = 100
):
    """
    Compare multiple policies.

    Args:
        env_config: Environment configuration
        policy_names: List of policy names to test
        num_episodes: Number of episodes per policy
        max_steps: Maximum steps per episode

    Returns:
        Dictionary with comparison results
    """
    results = {policy_name: [] for policy_name in policy_names}

    for policy_name in policy_names:
        print(f"\n{'='*60}")
        print(f"Testing Policy: {policy_name.upper()}")
        print(f"{'='*60}")

        for episode in range(num_episodes):
            # Create fresh environment
            env = JobMarketEnv(**env_config)

            # Create policy
            policy = create_policy(policy_name, env_config)

            # Run episode
            render = (episode == 0)  # Render first episode
            episode_result = run_episode(env, policy, max_steps, render=render)

            results[policy_name].append(episode_result)

            # Print summary
            avg_reward = np.mean(list(episode_result['total_rewards'].values()))
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, "
                  f"Length = {episode_result['episode_length']}, "
                  f"Unemployment = {episode_result['final_unemployment']:.2%}")

    return results


def print_comparison_summary(results: Dict):
    """Print summary statistics comparing policies."""
    print(f"\n{'='*60}")
    print("POLICY COMPARISON SUMMARY")
    print(f"{'='*60}\n")

    policy_stats = {}

    for policy_name, episodes in results.items():
        # Aggregate statistics across episodes
        avg_rewards = []
        episode_lengths = []
        final_unemployments = []

        for ep in episodes:
            avg_rewards.append(np.mean(list(ep['total_rewards'].values())))
            episode_lengths.append(ep['episode_length'])
            final_unemployments.append(ep['final_unemployment'])

        policy_stats[policy_name] = {
            'mean_reward': np.mean(avg_rewards),
            'std_reward': np.std(avg_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_unemployment': np.mean(final_unemployments)
        }

    # Print table
    print(f"{'Policy':<20} {'Mean Reward':<15} {'Std Reward':<15} {'Avg Length':<12} {'Unemployment':<12}")
    print(f"{'-'*80}")

    for policy_name, stats in policy_stats.items():
        print(f"{policy_name:<20} "
              f"{stats['mean_reward']:<15.2f} "
              f"{stats['std_reward']:<15.2f} "
              f"{stats['mean_length']:<12.1f} "
              f"{stats['mean_unemployment']:<12.2%}")

    print(f"{'-'*80}\n")


def main():
    """Main test function."""
    print("="*60)
    print("HIRERL BASELINE POLICY TESTING")
    print("="*60)

    # Environment configuration
    env_config = {
        'num_companies': 3,
        'num_workers': 10,
        'ability_dim': 1,
        'max_workers_per_company': 5,
        'gamma': 0.1,
        'g0': 0.1,
        'g1': 0.05,
        'base_firing_cost': 0.1,
        'base_hiring_cost': 0.2,
        'base_screening_cost': 0.5,
        'worker_bargaining_power': 0.6,
        'max_timesteps': 50,
        'seed': 42
    }

    # Policies to test
    policy_names = [
        'random',
        'greedy',
        'no_screening',
        'high_screening',
        'heuristic'
    ]

    # Run comparison
    results = compare_policies(
        env_config=env_config,
        policy_names=policy_names,
        num_episodes=3,
        max_steps=50
    )

    # Print summary
    print_comparison_summary(results)

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print("="*60)


if __name__ == '__main__':
    main()
