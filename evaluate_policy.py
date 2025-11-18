"""
Policy Evaluation Script for HireRL

This script evaluates trained PPO policies in unseen environments.
It supports flexible configuration to evaluate models trained with different
environment setups (varying number of agents, workers, etc.).

Features:
- Load policies from checkpoints
- Evaluate on new unseen environments
- Support different environment configurations
- Comprehensive metrics and visualization
- Deterministic and stochastic evaluation modes
- Episode recording and analysis

Usage:
    python evaluate_policy.py --checkpoint_dir checkpoints --env_config config.json
    python evaluate_policy.py --model company_0_final.pt --n_episodes 20
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pettingzoo'))

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from hirerl import JobMarketEnv
from utils import EpisodeLogger, PerformanceMetrics, compute_gini_coefficient
from train_ppo import ActorCritic


class PolicyEvaluator:
    """
    Evaluator for trained PPO policies.

    Supports:
    - Loading models with different architectures
    - Evaluating on new unseen environments
    - Flexible environment configuration
    - Comprehensive metric tracking
    """

    def __init__(
        self,
        checkpoint_paths: Dict[str, str],
        env_config: Dict[str, Any],
        device: str = 'cpu',
        seed: Optional[int] = None
    ):
        """
        Initialize policy evaluator.

        Args:
            checkpoint_paths: Dict mapping agent_name -> checkpoint_path
            env_config: Environment configuration dictionary
            device: Device to run evaluation on
            seed: Random seed for evaluation
        """
        self.checkpoint_paths = checkpoint_paths
        self.env_config = env_config
        self.device = device
        self.seed = seed if seed is not None else 42

        # Create evaluation environment
        self.env = self._create_environment(env_config, self.seed)

        # Load policies
        self.policies = self._load_policies(checkpoint_paths)

    def _create_environment(self, config: Dict[str, Any], seed: int) -> JobMarketEnv:
        """Create environment from configuration."""
        return JobMarketEnv(
            num_companies=config.get('num_companies', 3),
            num_workers=config.get('num_workers', 10),
            max_workers_per_company=config.get('max_workers_per_company', 5),
            ability_dim=config.get('ability_dim', 1),
            gamma=config.get('gamma', 0.1),
            g0=config.get('g0', 0.1),
            g1=config.get('g1', 0.05),
            base_firing_cost=config.get('base_firing_cost', 0.1),
            base_hiring_cost=config.get('base_hiring_cost', 0.2),
            base_screening_cost=config.get('base_screening_cost', 0.5),
            worker_bargaining_power=config.get('worker_bargaining_power', 0.6),
            max_timesteps=config.get('max_timesteps', 100),
            seed=seed
        )

    def _load_policies(self, checkpoint_paths: Dict[str, str]) -> Dict[str, ActorCritic]:
        """
        Load policies from checkpoints.

        IMPORTANT: The evaluation environment must have the same observation
        and action space dimensions as the training environment. This means:
        - Same number of workers (num_workers)
        - Same number of companies (num_companies)
        - Same ability dimension (ability_dim)

        You CAN vary other parameters like costs, dynamics, max_timesteps, etc.

        Raises:
            RuntimeError: If checkpoint dimensions don't match environment
        """
        policies = {}

        for agent_name, checkpoint_path in checkpoint_paths.items():
            # Get observation and action dimensions from environment
            obs_space = self.env.observation_space(agent_name)
            obs_dim = obs_space.spaces['observation'].shape[0]
            action_dim = self.env.action_space(agent_name).n

            # Create network
            network = ActorCritic(obs_dim, action_dim).to(self.device)

            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            try:
                network.load_state_dict(checkpoint)
            except RuntimeError as e:
                error_msg = (
                    f"\n{'='*70}\n"
                    f"ERROR: Model dimension mismatch for {agent_name}\n"
                    f"{'='*70}\n\n"
                    f"The checkpoint was trained with different environment dimensions.\n\n"
                    f"COMMON CAUSES:\n"
                    f"  - Different number of workers (num_workers)\n"
                    f"  - Different number of companies (num_companies)\n"
                    f"  - Different ability dimension (ability_dim)\n\n"
                    f"SOLUTION:\n"
                    f"  The evaluation environment must have the SAME:\n"
                    f"  - num_workers\n"
                    f"  - num_companies\n"
                    f"  - ability_dim\n\n"
                    f"  You CAN change other parameters like:\n"
                    f"  - Costs (firing, hiring, screening)\n"
                    f"  - Dynamics (gamma, g0, g1)\n"
                    f"  - Worker bargaining power\n"
                    f"  - Max timesteps\n\n"
                    f"Original error: {str(e)}\n"
                    f"{'='*70}\n"
                )
                raise RuntimeError(error_msg) from e

            network.eval()
            policies[agent_name] = network

        return policies

    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluate policies for multiple episodes.

        Args:
            n_episodes: Number of episodes to evaluate
            deterministic: Use deterministic (argmax) actions
            render: Render environment during evaluation
            verbose: Print detailed progress

        Returns:
            Dictionary containing evaluation metrics
        """
        if verbose:
            mode = "deterministic" if deterministic else "stochastic"
            print(f"\nEvaluating {n_episodes} episodes ({mode})...")

        # Track metrics across episodes
        episode_rewards = {agent: [] for agent in self.env.possible_agents}
        episode_lengths = []
        episode_metrics = []

        # Track worker employment histories across all episodes
        worker_histories = []  # List of dicts, one per episode

        # Logger for detailed tracking
        logger = EpisodeLogger(self.env.possible_agents)
        perf_metrics = PerformanceMetrics()

        for episode in range(n_episodes):

            # Reset for new episode
            observations, _ = self.env.reset(seed=self.seed + episode)
            logger.reset()

            ep_rewards = {agent: 0.0 for agent in self.env.possible_agents}
            ep_length = 0
            done = False

            # Track worker states each timestep for this episode
            num_workers = self.env.num_workers
            worker_trajectories = {
                w_id: {
                    'employment_history': [],  # List of (timestep, company_id, wage)
                    'actions_received': [],     # List of (timestep, company_id, action_type)
                    'ability': float(self.env.worker_pool.workers[w_id].sigma_true[0]) if hasattr(self.env.worker_pool.workers[w_id].sigma_true, '__len__') else float(self.env.worker_pool.workers[w_id].sigma_true),
                    'initial_signal': float(self.env.worker_pool.workers[w_id].sigma_hat_0[0]) if hasattr(self.env.worker_pool.workers[w_id].sigma_hat_0, '__len__') else float(self.env.worker_pool.workers[w_id].sigma_hat_0)
                }
                for w_id in range(num_workers)
            }

            while not done:
                # Track current worker states
                for w_id in range(num_workers):
                    worker = self.env.worker_pool.workers[w_id]
                    worker_trajectories[w_id]['employment_history'].append({
                        'timestep': ep_length,
                        'employed_by': worker.employed_by,
                        'wage': worker.wage,
                        'experience': float(worker.experience),
                        'tenure': int(worker.tenure),
                        'public_signal': float(worker.sigma_hat[0]) if hasattr(worker.sigma_hat, '__len__') else float(worker.sigma_hat)
                    })

                # Get actions from policies
                actions = {}
                for agent_name in self.env.agents:
                    if agent_name in self.policies:
                        action = self.policies[agent_name].get_action(
                            observations[agent_name],
                            deterministic=deterministic
                        )
                        actions[agent_name] = action
                    else:
                        # Random action for agents without loaded policy
                        actions[agent_name] = self.env.action_space(agent_name).sample()

                # Decode and track actions
                for agent_name, action_id in actions.items():
                    company_idx = int(agent_name.split("_")[1])
                    action_type, worker_id, value = self.env._decode_action(agent_name, action_id)

                    if worker_id is not None:
                        worker_trajectories[worker_id]['actions_received'].append({
                            'timestep': ep_length,
                            'company_id': company_idx,
                            'action_type': action_type,
                            'value': value
                        })

                # Step environment
                next_observations, rewards, terminations, truncations, infos = self.env.step(actions)

                # Log step
                logger.log_step(ep_length, rewards, infos, actions)

                # Track rewards
                for agent_name in self.env.possible_agents:
                    if agent_name in rewards:
                        ep_rewards[agent_name] += rewards[agent_name]

                ep_length += 1

                # Render if requested
                if render:
                    self.env.render()

                observations = next_observations
                done = all(terminations.values()) or all(truncations.values())

            # Store worker histories for this episode
            worker_histories.append(worker_trajectories)

            # Store episode results
            for agent_name in self.env.possible_agents:
                episode_rewards[agent_name].append(ep_rewards[agent_name])
            episode_lengths.append(ep_length)

            # Compute episode metrics
            episode_summary = logger.get_summary()
            episode_metrics.append(episode_summary)

        # Compute aggregate statistics
        results = self._compute_statistics(episode_rewards, episode_lengths, episode_metrics, worker_histories)

        if verbose:
            self._print_summary(results)
            self._print_worker_summary(results, n_workers=min(5, num_workers))  # Show top 5 workers

        return results

    def _compute_statistics(
        self,
        episode_rewards: Dict[str, List[float]],
        episode_lengths: List[int],
        episode_metrics: List[Dict[str, Any]],
        worker_histories: List[Dict[int, Dict]]
    ) -> Dict[str, Any]:
        """Compute aggregate statistics from episode data."""
        results = {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'agent_statistics': {},
            'market_statistics': {},
            'aggregate': {}
        }

        # Per-agent statistics
        for agent_name, rewards in episode_rewards.items():
            results['agent_statistics'][agent_name] = {
                'mean_reward': float(np.mean(rewards)),
                'std_reward': float(np.std(rewards)),
                'min_reward': float(np.min(rewards)),
                'max_reward': float(np.max(rewards)),
                'total_reward': float(np.sum(rewards))
            }

        # Market statistics
        unemployment_rates = [m['market']['final_unemployment_rate'] for m in episode_metrics]
        avg_wages = [m['market']['final_average_wage'] for m in episode_metrics]

        results['market_statistics'] = {
            'mean_unemployment_rate': float(np.mean(unemployment_rates)),
            'std_unemployment_rate': float(np.std(unemployment_rates)),
            'mean_wage': float(np.mean(avg_wages)),
            'std_wage': float(np.std(avg_wages))
        }

        # Aggregate statistics
        total_rewards = [sum(episode_rewards[a][i] for a in episode_rewards.keys())
                        for i in range(len(episode_lengths))]

        results['aggregate'] = {
            'mean_episode_length': float(np.mean(episode_lengths)),
            'std_episode_length': float(np.std(episode_lengths)),
            'mean_total_reward': float(np.mean(total_rewards)),
            'std_total_reward': float(np.std(total_rewards))
        }

        # Analyze worker histories
        results['worker_histories'] = worker_histories
        results['worker_statistics'] = self._analyze_worker_histories(worker_histories)

        return results

    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary."""
        print("\nResults:")
        for agent_name, stats in results['agent_statistics'].items():
            print(f"  {agent_name}: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")

        market = results['market_statistics']
        agg = results['aggregate']
        print(f"  Unemployment: {market['mean_unemployment_rate']:.2%}")
        print(f"  Avg Wage: {market['mean_wage']:.2f}")
        print(f"  Episode Length: {agg['mean_episode_length']:.1f}")

    def _analyze_worker_histories(self, worker_histories: List[Dict[int, Dict]]) -> Dict[str, Any]:
        """Analyze worker employment trajectories across episodes."""
        if not worker_histories:
            return {}

        num_episodes = len(worker_histories)
        num_workers = len(worker_histories[0])

        worker_stats = {}
        for worker_id in range(num_workers):
            # Aggregate stats across all episodes for this worker
            total_employed_time = 0
            companies_worked_for = set()
            total_wages_earned = 0.0
            num_hires = 0
            num_fires = 0
            num_interviews = 0
            num_offers_received = 0

            for ep_idx, episode_history in enumerate(worker_histories):
                worker_data = episode_history[worker_id]

                # Count employment time
                for state in worker_data['employment_history']:
                    if state['employed_by'] != -1:
                        total_employed_time += 1
                        companies_worked_for.add(state['employed_by'])
                        total_wages_earned += state['wage']

                # Count actions received
                prev_employed_by = -1
                for i, state in enumerate(worker_data['employment_history']):
                    if state['employed_by'] != -1 and prev_employed_by == -1:
                        num_hires += 1
                    elif state['employed_by'] == -1 and prev_employed_by != -1:
                        num_fires += 1
                    prev_employed_by = state['employed_by']

                # Count interview and offer actions
                for action in worker_data['actions_received']:
                    if action['action_type'] == 'interview':
                        num_interviews += 1
                    elif action['action_type'] == 'offer':
                        num_offers_received += 1

            # Compute averages
            total_timesteps = sum(len(worker_histories[ep][worker_id]['employment_history'])
                                 for ep in range(num_episodes))

            worker_stats[worker_id] = {
                'ability': float(worker_histories[0][worker_id]['ability']),
                'initial_signal': float(worker_histories[0][worker_id]['initial_signal']),
                'avg_employment_rate': total_employed_time / total_timesteps if total_timesteps > 0 else 0,
                'companies_worked_for': sorted(list(companies_worked_for)),
                'num_different_companies': len(companies_worked_for),
                'avg_wage_when_employed': total_wages_earned / total_employed_time if total_employed_time > 0 else 0,
                'total_hires': num_hires,
                'total_fires': num_fires,
                'total_interviews': num_interviews,
                'total_offers_received': num_offers_received,
                'job_stability': num_hires / num_episodes if num_episodes > 0 else 0  # Jobs per episode
            }

        return worker_stats

    def _print_worker_summary(self, results: Dict[str, Any], n_workers: int = 5):
        """Print summary of worker employment histories."""
        if 'worker_statistics' not in results or not results['worker_statistics']:
            return

        worker_stats = results['worker_statistics']

        # Sort workers by employment rate (most employed first)
        sorted_workers = sorted(
            worker_stats.items(),
            key=lambda x: x[1]['avg_employment_rate'],
            reverse=True
        )

        print(f"\n\nWorker Employment Summaries (Top {n_workers} by employment rate):")
        print("=" * 80)

        for worker_id, stats in sorted_workers[:n_workers]:
            print(f"\nWorker {worker_id}:")
            print(f"  Ability: {stats['ability']:.2f} | Initial Signal: {stats['initial_signal']:.2f}")
            print(f"  Employment Rate: {stats['avg_employment_rate']:.1%}")
            print(f"  Avg Wage: ${stats['avg_wage_when_employed']:.2f}")
            print(f"  Companies Worked For: {stats['companies_worked_for']} ({stats['num_different_companies']} total)")
            print(f"  Hires: {stats['total_hires']} | Fires: {stats['total_fires']} | Interviews: {stats['total_interviews']}")
            print(f"  Job Stability: {stats['job_stability']:.2f} jobs/episode")

    def save_results(self, results: Dict[str, Any], filepath: str):
        """Save evaluation results to JSON."""
        # Create a copy without the full worker_histories (too large for JSON)
        # Keep worker_statistics which is the analyzed summary
        results_to_save = {k: v for k, v in results.items() if k != 'worker_histories'}

        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        results_to_save = convert_numpy(results_to_save)

        with open(filepath, 'w') as f:
            json.dump(results_to_save, f, indent=2)

    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Create visualization of evaluation results.

        Args:
            results: Evaluation results dictionary
            save_path: Path to save figure (if None, displays interactively)
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Policy Evaluation Results', fontsize=16, fontweight='bold')

        # 1. Episode rewards by agent
        ax = axes[0, 0]
        episode_rewards = results['episode_rewards']
        for agent_name, rewards in episode_rewards.items():
            ax.plot(rewards, marker='o', label=agent_name, alpha=0.7)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards by Agent')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Reward distribution
        ax = axes[0, 1]
        reward_data = [rewards for rewards in episode_rewards.values()]
        positions = range(len(episode_rewards))
        bp = ax.boxplot(reward_data, positions=positions, labels=episode_rewards.keys(), patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
        ax.set_xlabel('Agent')
        ax.set_ylabel('Reward')
        ax.set_title('Reward Distribution Across Episodes')
        ax.grid(True, alpha=0.3, axis='y')

        # 3. Episode lengths
        ax = axes[1, 0]
        episode_lengths = results['episode_lengths']
        ax.plot(episode_lengths, marker='s', color='green', alpha=0.7)
        ax.axhline(y=np.mean(episode_lengths), color='red', linestyle='--',
                   label=f'Mean: {np.mean(episode_lengths):.1f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Episode Length')
        ax.set_title('Episode Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Summary statistics
        ax = axes[1, 1]
        ax.axis('off')

        summary_text = "Summary Statistics\n" + "="*40 + "\n\n"
        for agent_name, stats in results['agent_statistics'].items():
            summary_text += f"{agent_name}:\n"
            summary_text += f"  Mean: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}\n"
        summary_text += "\nMarket:\n"
        market = results['market_statistics']
        summary_text += f"  Unemployment: {market['mean_unemployment_rate']:.2%}\n"
        summary_text += f"  Avg Wage: {market['mean_wage']:.2f}\n"

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontfamily='monospace', fontsize=10, verticalalignment='top')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

        plt.close()


def auto_discover_checkpoints(checkpoint_dir: str, update_num: Optional[int] = None) -> Dict[str, str]:
    """
    Automatically discover checkpoint files in directory.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        update_num: Specific update number to load (None for final models)

    Returns:
        Dictionary mapping agent_name -> checkpoint_path
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoints = {}

    if update_num is None:
        # Look for final models
        pattern = "*_final.pt"
    else:
        # Look for specific update
        pattern = f"*_update_{update_num}.pt"

    for checkpoint_path in checkpoint_dir.glob(pattern):
        # Extract agent name from filename
        filename = checkpoint_path.stem  # e.g., "company_0_final"
        if update_num is None:
            agent_name = filename.replace("_final", "")
        else:
            agent_name = filename.replace(f"_update_{update_num}", "")

        checkpoints[agent_name] = str(checkpoint_path)

    return checkpoints


def create_env_config_from_training(
    num_companies: Optional[int] = None,
    num_workers: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Create environment configuration for evaluation.

    This allows you to evaluate policies on environments with different
    configurations than training (e.g., more/fewer workers, different costs).

    Args:
        num_companies: Number of companies (if None, inferred from checkpoints)
        num_workers: Number of workers
        **kwargs: Additional environment parameters

    Returns:
        Environment configuration dictionary
    """
    config = {
        'num_companies': num_companies or 3,
        'num_workers': num_workers or 10,
        'max_workers_per_company': kwargs.get('max_workers_per_company', 5),
        'ability_dim': kwargs.get('ability_dim', 1),
        'gamma': kwargs.get('gamma', 0.1),
        'g0': kwargs.get('g0', 0.1),
        'g1': kwargs.get('g1', 0.05),
        'base_firing_cost': kwargs.get('base_firing_cost', 0.1),
        'base_hiring_cost': kwargs.get('base_hiring_cost', 0.2),
        'base_screening_cost': kwargs.get('base_screening_cost', 0.5),
        'worker_bargaining_power': kwargs.get('worker_bargaining_power', 0.6),
        'max_timesteps': kwargs.get('max_timesteps', 100)
    }

    return config


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate trained PPO policies')

    # Run specification
    parser.add_argument('--run_name', type=str, default=None,
                       help='Run name (will load from runs/{run_name}/)')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                       help='Directory containing checkpoint files (overrides --run_name)')
    parser.add_argument('--update_num', type=int, default=None,
                       help='Specific update number to load (default: final models)')

    # Evaluation arguments
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--deterministic', action='store_true', default=True,
                       help='Use deterministic actions (argmax)')
    parser.add_argument('--stochastic', action='store_false', dest='deterministic',
                       help='Use stochastic actions (sampling)')
    parser.add_argument('--render', action='store_true',
                       help='Render environment during evaluation')
    parser.add_argument('--seed', type=int, default=999,
                       help='Random seed for evaluation')

    # Environment configuration (optional overrides)
    parser.add_argument('--max_timesteps', type=int, default=None,
                       help='Override episode length')
    parser.add_argument('--env_config', type=str, default=None,
                       help='Path to JSON file with environment configuration')

    # Output arguments
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to output directory')

    args = parser.parse_args()

    # Determine checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir
        config_path = None
    elif args.run_name:
        checkpoint_dir = f"runs/{args.run_name}/checkpoints"
        config_path = f"runs/{args.run_name}/config.json"
    else:
        print("ERROR: Must specify either --run_name or --checkpoint_dir")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Discover checkpoints
    print(f"\nLoading from: {checkpoint_dir}")
    checkpoints = auto_discover_checkpoints(checkpoint_dir, args.update_num)

    if not checkpoints:
        print(f"ERROR: No checkpoints found in {checkpoint_dir}")
        return

    print(f"Found {len(checkpoints)} checkpoint(s)")

    # Load environment configuration
    if args.env_config:
        # User-provided config takes precedence
        with open(args.env_config, 'r') as f:
            env_config = json.load(f)
    elif config_path and Path(config_path).exists():
        # Load from run's config.json
        with open(config_path, 'r') as f:
            full_config = json.load(f)
            env_config = full_config['env']
            # Apply overrides
            if args.max_timesteps is not None:
                env_config['max_timesteps'] = args.max_timesteps
        print(f"Loaded config from {config_path}")
    else:
        # Fallback to defaults
        env_config = create_env_config_from_training(
            num_companies=len(checkpoints),
            num_workers=10,
            max_timesteps=args.max_timesteps or 100
        )

    # Create evaluator
    evaluator = PolicyEvaluator(
        checkpoint_paths=checkpoints,
        env_config=env_config,
        device='cpu',
        seed=args.seed
    )

    # Run evaluation
    results = evaluator.evaluate(
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
        render=args.render,
        verbose=True
    )

    # Save results
    results_path = output_dir / f"evaluation_results_seed{args.seed}.json"
    evaluator.save_results(results, str(results_path))

    # Create plots
    if args.save_plots:
        plot_path = output_dir / f"evaluation_plots_seed{args.seed}.png"
        evaluator.plot_results(results, save_path=str(plot_path))
    else:
        evaluator.plot_results(results)

    print(f"\nSaved to {output_dir}/")


if __name__ == '__main__':
    main()
