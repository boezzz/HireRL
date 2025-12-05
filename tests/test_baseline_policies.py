"""
tests different baseline policies in the job market environment:
1. Random policy
2. Greedy policy (no screening)
3. High screening policy
4. Heuristic policy
5. Stable matching (environment variant)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'realzoo'))

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from pathlib import Path

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

    # Use possible_agents instead of agents (which may be empty after episode ends)
    agent_key = env.possible_agents[0]

    return {
        'total_rewards': total_rewards,
        'episode_length': episode_length,
        'final_unemployment': infos[agent_key]['unemployment_rate'],
        'final_avg_wage': infos[agent_key]['avg_wage'],
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

    return policy_stats


def plot_policy_comparison(results: Dict, save_path: str = None):
    """
    Create comprehensive visualization comparing baseline policies.

    Args:
        results: Dictionary with policy results from compare_policies
        save_path: Optional path to save plots
    """
    # Create output directory if needed
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)

    # Aggregate statistics
    policy_stats = {}
    for policy_name, episodes in results.items():
        avg_rewards = []
        episode_lengths = []
        final_unemployments = []
        workforce_histories = []
        profit_histories = []

        for ep in episodes:
            avg_rewards.append(np.mean(list(ep['total_rewards'].values())))
            episode_lengths.append(ep['episode_length'])
            final_unemployments.append(ep['final_unemployment'])

            # Average across companies for each timestep
            workforce_avg = np.mean([wh for wh in ep['workforce_history'].values()], axis=0)
            profit_avg = np.mean([ph for ph in ep['profit_history'].values()], axis=0)
            workforce_histories.append(workforce_avg)
            profit_histories.append(profit_avg)

        policy_stats[policy_name] = {
            'avg_rewards': avg_rewards,
            'episode_lengths': episode_lengths,
            'final_unemployments': final_unemployments,
            'workforce_histories': workforce_histories,
            'profit_histories': profit_histories
        }

    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Average Reward Comparison (Bar plot with error bars)
    ax1 = fig.add_subplot(gs[0, 0])
    policy_names = list(policy_stats.keys())
    mean_rewards = [np.mean(stats['avg_rewards']) for stats in policy_stats.values()]
    std_rewards = [np.std(stats['avg_rewards']) for stats in policy_stats.values()]

    bars = ax1.bar(policy_names, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Mean Episode Reward by Policy')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', alpha=0.3)

    # Color bars by performance
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(mean_rewards)))
    sorted_indices = np.argsort(mean_rewards)
    for i, (bar, idx) in enumerate(zip(bars, sorted_indices)):
        bar.set_color(colors[np.where(sorted_indices == i)[0][0]])

    # 2. Unemployment Rate Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    mean_unemployment = [np.mean(stats['final_unemployments']) for stats in policy_stats.values()]
    std_unemployment = [np.std(stats['final_unemployments']) for stats in policy_stats.values()]

    ax2.bar(policy_names, mean_unemployment, yerr=std_unemployment, capsize=5, alpha=0.7, color='coral')
    ax2.set_ylabel('Unemployment Rate')
    ax2.set_title('Final Unemployment Rate by Policy')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', alpha=0.3)

    # 3. Workforce Evolution over Time
    ax3 = fig.add_subplot(gs[1, :])
    for policy_name, stats in policy_stats.items():
        # Average workforce trajectory across episodes
        max_len = max(len(wh) for wh in stats['workforce_histories'])
        workforce_matrix = np.zeros((len(stats['workforce_histories']), max_len))

        for i, wh in enumerate(stats['workforce_histories']):
            workforce_matrix[i, :len(wh)] = wh
            if len(wh) < max_len:
                workforce_matrix[i, len(wh):] = wh[-1]  # Pad with last value

        mean_workforce = np.mean(workforce_matrix, axis=0)
        std_workforce = np.std(workforce_matrix, axis=0)

        timesteps = np.arange(max_len)
        ax3.plot(timesteps, mean_workforce, label=policy_name, linewidth=2, alpha=0.8)
        ax3.fill_between(timesteps,
                         mean_workforce - std_workforce,
                         mean_workforce + std_workforce,
                         alpha=0.2)

    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Average Workforce Size')
    ax3.set_title('Workforce Evolution Across Policies')
    ax3.legend(loc='best')
    ax3.grid(alpha=0.3)

    # 4. Profit Evolution over Time
    ax4 = fig.add_subplot(gs[2, :])
    for policy_name, stats in policy_stats.items():
        # Average profit trajectory across episodes
        max_len = max(len(ph) for ph in stats['profit_histories'])
        profit_matrix = np.zeros((len(stats['profit_histories']), max_len))

        for i, ph in enumerate(stats['profit_histories']):
            profit_matrix[i, :len(ph)] = ph
            if len(ph) < max_len:
                profit_matrix[i, len(ph):] = 0  # Pad with zeros

        # Cumulative profit
        cumulative_profit = np.cumsum(profit_matrix, axis=1)
        mean_profit = np.mean(cumulative_profit, axis=0)
        std_profit = np.std(cumulative_profit, axis=0)

        timesteps = np.arange(max_len)
        ax4.plot(timesteps, mean_profit, label=policy_name, linewidth=2, alpha=0.8)
        ax4.fill_between(timesteps,
                         mean_profit - std_profit,
                         mean_profit + std_profit,
                         alpha=0.2)

    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Cumulative Profit')
    ax4.set_title('Cumulative Profit Evolution Across Policies')
    ax4.legend(loc='best')
    ax4.grid(alpha=0.3)

    plt.suptitle('Baseline Policy Comparison - HireRL Environment', fontsize=16, fontweight='bold')

    # Save or show
    if save_path:
        filepath = os.path.join(save_path, 'baseline_policy_comparison.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {filepath}")

    plt.tight_layout()
    return fig


def plot_detailed_metrics(results: Dict, save_path: str = None):
    """
    Create detailed metric plots for each policy.

    Args:
        results: Dictionary with policy results
        save_path: Optional path to save plots
    """
    if save_path:
        Path(save_path).mkdir(parents=True, exist_ok=True)

    num_policies = len(results)
    fig, axes = plt.subplots(num_policies, 3, figsize=(18, 4 * num_policies))

    if num_policies == 1:
        axes = axes.reshape(1, -1)

    for idx, (policy_name, episodes) in enumerate(results.items()):
        # Extract data
        total_rewards_per_company = {f'company_{i}': [] for i in range(3)}
        workforce_sizes = []
        unemployment_rates = []

        for ep in episodes:
            for company, reward in ep['total_rewards'].items():
                total_rewards_per_company[company].append(reward)

            # Get final metrics
            workforce_sizes.append(np.mean([wh[-1] if len(wh) > 0 else 0
                                          for wh in ep['workforce_history'].values()]))
            unemployment_rates.append(ep['final_unemployment'])

        # Plot 1: Reward distribution per company
        ax = axes[idx, 0]
        company_data = [total_rewards_per_company[f'company_{i}'] for i in range(3)]
        bp = ax.boxplot(company_data, labels=[f'C{i}' for i in range(3)], patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_ylabel('Total Reward')
        ax.set_title(f'{policy_name}: Reward Distribution')
        ax.grid(axis='y', alpha=0.3)

        # Plot 2: Final workforce size distribution
        ax = axes[idx, 1]
        ax.hist(workforce_sizes, bins=10, alpha=0.7, color='green', edgecolor='black')
        ax.axvline(np.mean(workforce_sizes), color='red', linestyle='--',
                   label=f'Mean: {np.mean(workforce_sizes):.2f}')
        ax.set_xlabel('Final Workforce Size')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{policy_name}: Workforce Size')
        ax.legend()
        ax.grid(alpha=0.3)

        # Plot 3: Unemployment rate distribution
        ax = axes[idx, 2]
        ax.hist(unemployment_rates, bins=10, alpha=0.7, color='coral', edgecolor='black')
        ax.axvline(np.mean(unemployment_rates), color='red', linestyle='--',
                   label=f'Mean: {np.mean(unemployment_rates):.2%}')
        ax.set_xlabel('Final Unemployment Rate')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{policy_name}: Unemployment')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle('Detailed Policy Metrics', fontsize=16, fontweight='bold')

    if save_path:
        filepath = os.path.join(save_path, 'detailed_policy_metrics.png')
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Detailed plot saved to: {filepath}")

    plt.tight_layout()
    return fig


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
        'g0': 0.1,
        'g1': 0.05,
        'max_interview_cost': 2.0,
        'num_interview_cost_levels': 5,
        'action_mode': 'continuous',
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

    # Run comparison with more episodes for better statistics
    results = compare_policies(
        env_config=env_config,
        policy_names=policy_names,
        num_episodes=10,
        max_steps=50
    )

    # Print summary
    policy_stats = print_comparison_summary(results)

    # Create plots
    print("\n" + "="*60)
    print("GENERATING PLOTS")
    print("="*60)

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'baseline_results')

    # Generate comparison plots
    print("\nGenerating policy comparison plots...")
    plot_policy_comparison(results, save_path=output_dir)

    # Generate detailed metrics
    print("Generating detailed metric plots...")
    plot_detailed_metrics(results, save_path=output_dir)

    print("\n" + "="*60)
    print("TESTING COMPLETE")
    print(f"Results saved to: {os.path.abspath(output_dir)}")
    print("="*60)

    # Show plots only in interactive mode
    # Comment out plt.show() to avoid blocking in automated tests
    # plt.show()


if __name__ == '__main__':
    main()
