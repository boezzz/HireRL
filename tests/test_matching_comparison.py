"""
Compare baseline policies with different matching mechanisms:
- Greedy matching (default)
- Stable matching (Gale-Shapley)

This allows us to answer: How does greedy profit-maximizing matching compare to stable matching?
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'realzoo'))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path

from hirerl import JobMarketEnv
from hirerl_stable import JobMarketEnvStableMatching
from policies import create_policy


def run_episode(env, policy, max_steps: int = 100):
    """Run one episode with given policy."""
    observations, infos = env.reset()

    total_rewards = {agent: 0.0 for agent in env.agents}
    episode_length = 0

    workforce_history = {agent: [] for agent in env.agents}
    profit_history = {agent: [] for agent in env.agents}

    for step in range(max_steps):
        actions = {}
        for agent in env.agents:
            if hasattr(policy, '__getitem__'):
                actions[agent] = policy[agent].get_action(observations[agent], agent)
            else:
                actions[agent] = policy.get_action(observations[agent], agent)

        observations, rewards, terminations, truncations, infos = env.step(actions)

        for agent in env.agents:
            total_rewards[agent] += rewards[agent]
            workforce_history[agent].append(infos[agent]['workforce_size'])
            profit_history[agent].append(rewards[agent])

        episode_length += 1

        if all(terminations.values()) or all(truncations.values()):
            break

    return {
        'total_rewards': total_rewards,
        'episode_length': episode_length,
        'final_unemployment': infos[env.possible_agents[0]]['unemployment_rate'],
        'workforce_history': workforce_history,
        'profit_history': profit_history
    }


def compare_matching_mechanisms(
    env_config: Dict,
    policy_name: str,
    num_episodes: int = 10,
    max_steps: int = 50
):
    """
    Compare greedy vs stable matching for a given policy.

    Returns:
        Dictionary with results for both matching mechanisms
    """
    results = {
        'greedy_matching': [],
        'stable_matching': []
    }

    for mechanism in ['greedy_matching', 'stable_matching']:
        print(f"\n{'='*60}")
        print(f"Testing: {mechanism.upper()} with {policy_name} policy")
        print(f"{'='*60}")

        for episode in range(num_episodes):
            # Create environment with appropriate matching mechanism
            if mechanism == 'greedy_matching':
                env = JobMarketEnv(**env_config)
            else:
                env = JobMarketEnvStableMatching(**env_config)

            # Create policy
            policy = create_policy(policy_name, env_config)

            # Run episode
            episode_result = run_episode(env, policy, max_steps)
            results[mechanism].append(episode_result)

            avg_reward = np.mean(list(episode_result['total_rewards'].values()))
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}, "
                  f"Unemployment = {episode_result['final_unemployment']:.2%}")

    return results


def plot_matching_comparison(results: Dict, policy_name: str, save_path: str = None):
    """
    Create plots comparing greedy vs stable matching.

    Args:
        results: Dictionary with greedy_matching and stable_matching results
        policy_name: Name of policy used
        save_path: Optional path to save plots
    """
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    mechanisms = ['greedy_matching', 'stable_matching']
    colors = ['#3498db', '#e74c3c']

    # 1. Average Reward Comparison
    ax = axes[0, 0]
    mean_rewards = []
    std_rewards = []

    for mechanism in mechanisms:
        episodes = results[mechanism]
        rewards = [np.mean(list(ep['total_rewards'].values())) for ep in episodes]
        mean_rewards.append(np.mean(rewards))
        std_rewards.append(np.std(rewards))

    bars = ax.bar(['Greedy Matching', 'Stable Matching'], mean_rewards,
                   yerr=std_rewards, capsize=5, color=colors, alpha=0.7)
    ax.set_ylabel('Average Reward')
    ax.set_title('Mean Episode Reward by Matching Mechanism')
    ax.grid(axis='y', alpha=0.3)

    # 2. Unemployment Rate Comparison
    ax = axes[0, 1]
    mean_unemployment = []
    std_unemployment = []

    for mechanism in mechanisms:
        episodes = results[mechanism]
        unemployment = [ep['final_unemployment'] for ep in episodes]
        mean_unemployment.append(np.mean(unemployment))
        std_unemployment.append(np.std(unemployment))

    ax.bar(['Greedy Matching', 'Stable Matching'], mean_unemployment,
           yerr=std_unemployment, capsize=5, color=colors, alpha=0.7)
    ax.set_ylabel('Unemployment Rate')
    ax.set_title('Final Unemployment Rate by Matching Mechanism')
    ax.grid(axis='y', alpha=0.3)

    # 3. Workforce Evolution
    ax = axes[1, 0]

    for mechanism, color in zip(mechanisms, colors):
        episodes = results[mechanism]

        # Average workforce trajectory
        max_len = max(max(len(wh) for wh in ep['workforce_history'].values())
                      for ep in episodes)
        workforce_matrix = np.zeros((len(episodes), max_len))

        for i, ep in enumerate(episodes):
            workforce_avg = np.mean([wh for wh in ep['workforce_history'].values()], axis=0)
            workforce_matrix[i, :len(workforce_avg)] = workforce_avg
            if len(workforce_avg) < max_len:
                workforce_matrix[i, len(workforce_avg):] = workforce_avg[-1]

        mean_workforce = np.mean(workforce_matrix, axis=0)
        std_workforce = np.std(workforce_matrix, axis=0)

        timesteps = np.arange(max_len)
        label = 'Greedy' if mechanism == 'greedy_matching' else 'Stable'
        ax.plot(timesteps, mean_workforce, label=label, linewidth=2, color=color, alpha=0.8)
        ax.fill_between(timesteps,
                        mean_workforce - std_workforce,
                        mean_workforce + std_workforce,
                        color=color, alpha=0.2)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Average Workforce Size')
    ax.set_title('Workforce Evolution')
    ax.legend()
    ax.grid(alpha=0.3)

    # 4. Cumulative Profit
    ax = axes[1, 1]

    for mechanism, color in zip(mechanisms, colors):
        episodes = results[mechanism]

        max_len = max(max(len(ph) for ph in ep['profit_history'].values())
                      for ep in episodes)
        profit_matrix = np.zeros((len(episodes), max_len))

        for i, ep in enumerate(episodes):
            profit_avg = np.mean([ph for ph in ep['profit_history'].values()], axis=0)
            profit_matrix[i, :len(profit_avg)] = profit_avg

        cumulative_profit = np.cumsum(profit_matrix, axis=1)
        mean_profit = np.mean(cumulative_profit, axis=0)
        std_profit = np.std(cumulative_profit, axis=0)

        timesteps = np.arange(max_len)
        label = 'Greedy' if mechanism == 'greedy_matching' else 'Stable'
        ax.plot(timesteps, mean_profit, label=label, linewidth=2, color=color, alpha=0.8)
        ax.fill_between(timesteps,
                        mean_profit - std_profit,
                        mean_profit + std_profit,
                        color=color, alpha=0.2)

    ax.set_xlabel('Timestep')
    ax.set_ylabel('Cumulative Profit')
    ax.set_title('Cumulative Profit Evolution')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle(f'Greedy vs Stable Matching Comparison\n({policy_name} policy)',
                 fontsize=14, fontweight='bold')

    if save_path:
        filepath = os.path.join(save_path, f'matching_comparison_{policy_name}.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {filepath}")

    plt.tight_layout()
    return fig


def print_matching_summary(results: Dict, policy_name: str):
    """Print summary comparing matching mechanisms."""
    print(f"\n{'='*60}")
    print(f"MATCHING MECHANISM COMPARISON - {policy_name.upper()} POLICY")
    print(f"{'='*60}\n")

    summary = {}

    for mechanism in ['greedy_matching', 'stable_matching']:
        episodes = results[mechanism]

        avg_rewards = [np.mean(list(ep['total_rewards'].values())) for ep in episodes]
        episode_lengths = [ep['episode_length'] for ep in episodes]
        final_unemployments = [ep['final_unemployment'] for ep in episodes]

        summary[mechanism] = {
            'mean_reward': np.mean(avg_rewards),
            'std_reward': np.std(avg_rewards),
            'mean_length': np.mean(episode_lengths),
            'mean_unemployment': np.mean(final_unemployments)
        }

    print(f"{'Mechanism':<20} {'Mean Reward':<15} {'Std Reward':<15} {'Avg Length':<12} {'Unemployment':<12}")
    print(f"{'-'*80}")

    for mechanism, stats in summary.items():
        name = 'Greedy Matching' if mechanism == 'greedy_matching' else 'Stable Matching'
        print(f"{name:<20} "
              f"{stats['mean_reward']:<15.2f} "
              f"{stats['std_reward']:<15.2f} "
              f"{stats['mean_length']:<12.1f} "
              f"{stats['mean_unemployment']:<12.2%}")

    print(f"{'-'*80}\n")

    # Compute performance difference
    greedy_reward = summary['greedy_matching']['mean_reward']
    stable_reward = summary['stable_matching']['mean_reward']
    diff = greedy_reward - stable_reward
    pct_diff = (diff / abs(stable_reward)) * 100 if stable_reward != 0 else 0

    print(f"Performance Difference:")
    print(f"  Greedy vs Stable Reward: {diff:+.2f} ({pct_diff:+.1f}%)")

    if diff > 0:
        print(f"  Result: Greedy matching achieves {abs(pct_diff):.1f}% higher reward")
    else:
        print(f"  Result: Stable matching achieves {abs(pct_diff):.1f}% higher reward")


def main():
    """Main comparison test."""
    print("="*60)
    print("GREEDY vs STABLE MATCHING COMPARISON")
    print("="*60)

    # Environment configuration
    env_config = {
        'num_companies': 3,
        'num_workers': 10,
        'ability_dim': 1,
        'max_workers_per_company': 5,
        'g0': 0.1,
        'g1': 0.05,
        'max_interview_cost': 2.0,
        'num_interview_cost_levels': 5,
        'action_mode': 'continuous',
        'max_timesteps': 50,
        'seed': 42
    }

    # Test with greedy policy
    policy_name = 'greedy'

    results = compare_matching_mechanisms(
        env_config=env_config,
        policy_name=policy_name,
        num_episodes=10,
        max_steps=50
    )

    # Print summary
    print_matching_summary(results, policy_name)

    # Create plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'baseline_results')
    plot_matching_comparison(results, policy_name, save_path=output_dir)

    print("\n" + "="*60)
    print("COMPARISON COMPLETE")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print("="*60)

    # Comment out plt.show() to avoid blocking in automated tests
    # plt.show()


if __name__ == '__main__':
    main()
