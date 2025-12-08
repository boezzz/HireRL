"""
Run all experiments for HireRL paper.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../realzoo'))

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from realzoo.hirerl import JobMarketEnv
from train_ppo import IPPOTrainer, set_global_seed

# Import plotting utilities
from plot_utils import (
    plot_learning_curves,
    plot_learned_policy_analysis,
    plot_firing_cost_comparison,
    create_market_structure_table,
    create_firing_cost_table,
)

# Import empirical data initialization
try:
    from real_data_init import sde as sde_data
except Exception:
    sde_data = None  # Fallback to synthetic if unavailable


def get_empirical_env_params(
    num_companies: int,
    num_workers: int,
    seed: int,
    employees_per_agent: int = 20_000,
    auto_calibrate_noise: bool = True
) -> Dict[str, Any]:
    """
    Get empirical environment parameters from real data.

    Returns dict with:
    - firm_capacities, firm_types, firm_type_premia
    - profit_noise_var, wage_scale
    - delta_interview0_sq (for calibration)
    """
    if sde_data is None:
        print("[warn] SDE data not available, using synthetic defaults")
        return {}

    params = {}

    try:
        # Initialize firms from real data
        firms_df = sde_data.initialize_firms(
            num_firms=num_companies,
            type_config=sde_data.firms_type_config,
            random_state=seed,
            sample_strategy="empirical",
        )
        params['firm_capacities'] = sde_data.to_env_capacities(
            firms_df, employees_per_agent=employees_per_agent
        )
        params['firm_types'] = firms_df["firm_type"].tolist()
        params['firm_type_premia'] = sde_data.estimate_size_wage_premia()

        print(f"[info] Empirical firm init: {len(params['firm_capacities'])} firms, "
              f"types={params['firm_types']}, capacities={params['firm_capacities']}")

    except Exception as e:
        print(f"[warn] Empirical firm init failed: {e}")

    # Calibrate noise parameters
    if auto_calibrate_noise:
        try:
            wage_df = sde_data.load_wage_exp()
            # Use yearly buckets (bin_step=1) so each timestep aligns to ~1 year
            path = sde_data.wage_variance_ratio_path(wage_df, bin_step=1)
            if path:
                periods = max(path.keys())
                best = sde_data.calibrate_signal_noise(
                    target_path=path, periods=periods, n_workers=5000, seed=0
                )
            else:
                ratio = sde_data.wage_variance_ratio(wage_df)
                best = None if ratio is None else sde_data.calibrate_signal_noise(
                    target_ratio=ratio, periods=3, n_workers=5000, seed=0
                )

            if best:
                delta_interview0_sq, profit_noise_var, _, _, _ = best
                params['delta_interview0_sq'] = delta_interview0_sq
                params['profit_noise_var'] = profit_noise_var
                print(f"[info] Calibrated noise: delta_interview0_sq={delta_interview0_sq:.4f}, "
                      f"profit_noise_var={profit_noise_var:.4f}")

            # Set wage scale to match empirical magnitudes
            wage_mean = float(wage_df["salary"].mean())
            if wage_mean > 0:
                params['wage_scale'] = wage_mean
                print(f"[info] Wage scale: {wage_mean:.2f}")

        except Exception as e:
            print(f"[warn] Noise calibration failed: {e}")

    return params


def extract_evaluation_metrics(
    env: JobMarketEnv,
    trainer: IPPOTrainer,
    n_episodes: int = 100
) -> Dict[str, Any]:
    """
    Evaluate trained policy and extract metrics for plotting.

    Returns:
        Dictionary with:
        - episodes: episode numbers
        - hiring_rates: hiring rate per episode
        - firing_rates: firing rate per episode
        - true_abilities: worker true abilities (σ)
        - avg_interview_costs: avg interview cost per worker across episodes
        - unemployment_rate: average unemployment rate
        - avg_wage: average wage
        - avg_profit: average profit
        - avg_employment_duration: average tenure
    """
    print(f"  Evaluating policy over {n_episodes} episodes...")

    # Track per-episode metrics
    episode_data = {
        'hiring_rates': [],
        'firing_rates': [],
        'profits': [],
        'unemployment_rates': [],
        'avg_wages': [],
    }

    # Track per-worker interview costs
    worker_interview_costs = {w: [] for w in range(env.num_workers)}
    worker_true_abilities = None  # Will be set on first episode

    for episode in range(n_episodes):
        observations, _ = env.reset()

        # Store true abilities (constant across episodes)
        if worker_true_abilities is None:
            worker_true_abilities = env.sigma_true.copy()

        # Track this episode
        ep_hires = 0
        ep_fires = 0
        ep_profits = []
        prev_employed = env.employed_by.copy()

        done = False
        while not done:
            # Get actions
            actions = {}
            for agent_name in env.agents:
                action = trainer.agents[agent_name].network.get_action(
                    observations[agent_name],
                    deterministic=True
                )
                actions[agent_name] = action

                # Track interview costs per worker
                for worker_id in range(env.num_workers):
                    cost = action[worker_id]
                    if cost > 0:
                        worker_interview_costs[worker_id].append(cost)

            # Step
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Track hires/fires
            new_employed = env.employed_by.copy()
            for w in range(env.num_workers):
                if prev_employed[w] < 0 and new_employed[w] >= 0:
                    ep_hires += 1
                elif prev_employed[w] >= 0 and new_employed[w] < 0:
                    ep_fires += 1
            prev_employed = new_employed

            # Track profits
            for agent in env.agents:
                if agent in rewards:
                    ep_profits.append(rewards[agent])

            done = all(terminations.values()) or all(truncations.values())

        # Store episode metrics
        episode_length = env.timestep
        episode_data['hiring_rates'].append(ep_hires / episode_length if episode_length > 0 else 0)
        episode_data['firing_rates'].append(ep_fires / episode_length if episode_length > 0 else 0)
        episode_data['profits'].append(np.mean(ep_profits) if ep_profits else 0)
        episode_data['unemployment_rates'].append(np.mean(env.employed_by < 0))
        employed_workers = env.employed_by >= 0
        episode_data['avg_wages'].append(
            np.mean(env.wages[employed_workers]) if np.any(employed_workers) else 0
        )

    # Compute per-worker average interview costs
    avg_interview_costs = np.array([
        np.mean(worker_interview_costs[w]) if worker_interview_costs[w] else 0
        for w in range(env.num_workers)
    ])

    # Compute employment duration from last evaluation
    employed_mask = env.employed_by >= 0
    avg_employment_duration = np.mean(env.tenure[employed_mask]) if np.any(employed_mask) else 0

    return {
        'episodes': np.arange(n_episodes),
        'hiring_rates': np.array(episode_data['hiring_rates']),
        'firing_rates': np.array(episode_data['firing_rates']),
        'true_abilities': worker_true_abilities,
        'avg_interview_costs': avg_interview_costs,
        'unemployment_rate': np.mean(episode_data['unemployment_rates']),
        'avg_wage': np.mean(episode_data['avg_wages']),
        'avg_profit': np.mean(episode_data['profits']),
        'avg_employment_duration': avg_employment_duration,
        'firing_rate': np.mean(episode_data['firing_rates']),
        'avg_interview_cost': np.mean(avg_interview_costs),
    }


def experiment_1_baseline(seed: int = 42, use_empirical: bool = False, run_name: Optional[str] = None):
    """
    Learning Dynamics

    Train baseline policy and generate:
    - Learning curves (during training)
    - Learned policy analysis (after training)

    Args:
        seed: Random seed
        use_empirical: If True, use empirical data from SDE for initialization
        run_name: Optional custom run name for the trainer
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Learning Dynamics")
    print("="*60)

    set_global_seed(seed)

    # Base parameters
    num_companies = 2
    num_workers = 10
    max_workers_per_company = 5
    max_timesteps = 20
    firing_cost_multiplier = 6.0

    # Get empirical or synthetic parameters
    env_params = {
        'num_companies': num_companies,
        'num_workers': num_workers,
        'max_workers_per_company': max_workers_per_company,
        'max_timesteps': max_timesteps,
        'firing_cost_multiplier': firing_cost_multiplier,
        'seed': seed,
    }

    if use_empirical:
        print("Using EMPIRICAL initialization from real data...")
        empirical_params = get_empirical_env_params(
            num_companies=num_companies,
            num_workers=num_workers,
            seed=seed,
        )
        env_params.update(empirical_params)
    else:
        print("Using SYNTHETIC initialization...")

    # Create environment
    env = JobMarketEnv(**env_params)

    # If empirical calibration provided delta_interview0_sq, apply it
    if use_empirical and 'delta_interview0_sq' in empirical_params:
        env.screening.delta0_sq = float(empirical_params['delta_interview0_sq'])

    # Train
    if run_name is None:
        run_name = "baseline_learning_dynamics"

    trainer = IPPOTrainer(
        env=env,
        lr=3e-3,
        run_name=run_name,
        seed=seed
    )

    print("\nTraining baseline policy...")
    trainer.train(
        total_timesteps=10_000,  # Much shorter training
        n_steps=256,  # Smaller rollouts
        n_epochs=4,  # Fewer epochs
        batch_size=32,  # Smaller batches
        log_interval=5,
        save_interval=50000,
        eval_interval=0  # No periodic eval during training
    )

    # Create output directories under this run
    figures_dir = os.path.join(trainer.run_dir, "figures")
    results_dir = os.path.join(trainer.run_dir, "results")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("\nGenerating learning curve plots...")
    plot_learning_curves(
        log_dir=trainer.run_dir,
        save_path=os.path.join(figures_dir, "fig1_learning_curves.png")
    )

    print("\nEvaluating learned policy...")
    eval_data = extract_evaluation_metrics(env, trainer, n_episodes=20)  # Fewer eval episodes

    print("\nGenerating learned policy analysis plots...")
    plot_learned_policy_analysis(
        eval_data=eval_data,
        save_path=os.path.join(figures_dir, "fig2_learned_policy.png")
    )

    # Save results
    with open(os.path.join(results_dir, "baseline_eval_data.pkl"), 'wb') as f:
        pickle.dump(eval_data, f)

    print(f"\n✓ All outputs saved to: {trainer.run_dir}/")
    return trainer.run_dir, eval_data


def experiment_2_firing_costs(seed: int = 42, use_empirical: bool = False, base_run_dir: str = "runs"):
    """
    Firing Cost Sensitivity

    Train separate policies for each firing cost multiplier.

    Args:
        seed: Random seed
        use_empirical: If True, use empirical data from SDE for initialization
        base_run_dir: Base directory for this experiment's outputs
    """ 
    print("\n" + "="*60)
    print("EXPERIMENT 2: Firing Cost Sensitivity")
    print("="*60)

    firing_costs = [0.0, 3.0, 6.0]
    results_by_cost = {}
    run_dirs = []

    for c_fire in firing_costs:
        print(f"\n--- Training with firing_cost_multiplier = {c_fire} ---")

        set_global_seed(seed)

        # Base parameters
        num_companies = 2
        num_workers = 10
        max_workers_per_company = 5
        max_timesteps = 20

        # Get empirical or synthetic parameters
        env_params = {
            'num_companies': num_companies,
            'num_workers': num_workers,
            'max_workers_per_company': max_workers_per_company,
            'max_timesteps': max_timesteps,
            'firing_cost_multiplier': c_fire,
            'seed': seed,
        }

        if use_empirical:
            print("  Using EMPIRICAL initialization...")
            empirical_params = get_empirical_env_params(
                num_companies=num_companies,
                num_workers=num_workers,
                seed=seed,
            )
            env_params.update(empirical_params)
        else:
            print("  Using SYNTHETIC initialization...")

        # Create environment
        env = JobMarketEnv(**env_params)

        # If empirical calibration provided delta_interview0_sq, apply it
        if use_empirical and 'delta_interview0_sq' in empirical_params:
            env.screening.delta0_sq = float(empirical_params['delta_interview0_sq'])

        run_name_prefix = base_run_dir.replace("runs/", "") if base_run_dir.startswith("runs/") else base_run_dir
        trainer = IPPOTrainer(
            env=env,
            lr=3e-3,
            run_name=f"{run_name_prefix}/firing_cost_{c_fire}",
            seed=seed
        )

        trainer.train(
            total_timesteps=10_000,  # Much shorter training
            n_steps=256,  # Smaller rollouts
            n_epochs=4,  # Fewer epochs
            batch_size=32,  # Smaller batches
            log_interval=10,
            save_interval=100000,
            eval_interval=0
        )

        # Evaluate
        eval_data = extract_evaluation_metrics(env, trainer, n_episodes=20)  # Fewer eval episodes
        results_by_cost[c_fire] = eval_data
        run_dirs.append(trainer.run_dir)

        print(f"  Firing rate: {eval_data['firing_rate']:.3f}")
        print(f"  Avg profit: {eval_data['avg_profit']:.3f}")
        print(f"  Unemployment: {eval_data['unemployment_rate']:.2%}")

    # Create combined output directory for comparison plots
    comparison_dir = os.path.join(base_run_dir, "firing_cost_comparison")
    figures_dir = os.path.join(comparison_dir, "figures")
    results_dir = os.path.join(comparison_dir, "results")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Generate plots
    print("\nGenerating firing cost comparison plots...")
    plot_firing_cost_comparison(
        results_by_cost=results_by_cost,
        save_path=os.path.join(figures_dir, "fig3_firing_cost_sensitivity.png")
    )

    # Generate table
    print("\nGenerating firing cost comparison table...")
    create_firing_cost_table(
        results_by_cost=results_by_cost,
        save_path=os.path.join(results_dir, "firing_cost_table.csv")
    )

    # Save results
    with open(os.path.join(results_dir, "firing_cost_results.pkl"), 'wb') as f:
        pickle.dump(results_by_cost, f)

    print(f"\n✓ Comparison outputs saved to: {comparison_dir}/")
    print(f"✓ Individual runs: {', '.join(run_dirs)}")
    return results_by_cost, comparison_dir


def experiment_3_market_structure(seed: int = 42, use_empirical: bool = False, base_run_dir: str = "runs"):
    """
    Market Structure

    Train policies for different market configurations.

    Args:
        seed: Random seed
        use_empirical: If True, use empirical data from SDE for initialization
        base_run_dir: Base directory for this experiment's outputs
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Market Structure")
    print("="*60)

    # Smaller configs for faster testing
    configs = [
        (2, 10),
        (2, 20),
        (3, 10),
        (3, 20),
    ]

    results = {}
    run_dirs = []

    for n_firms, n_workers in configs:
        print(f"\n--- Training with {n_firms} firms, {n_workers} workers ---")

        set_global_seed(seed)

        # Base parameters
        max_timesteps = 20
        firing_cost_multiplier = 6.0

        # Get empirical or synthetic parameters
        env_params = {
            'num_companies': n_firms,
            'num_workers': n_workers,
            'max_workers_per_company': max(5, n_workers // n_firms + 2),
            'max_timesteps': max_timesteps,
            'firing_cost_multiplier': firing_cost_multiplier,
            'seed': seed,
        }

        if use_empirical:
            print("  Using EMPIRICAL initialization...")
            empirical_params = get_empirical_env_params(
                num_companies=n_firms,
                num_workers=n_workers,
                seed=seed,
            )
            env_params.update(empirical_params)
        else:
            print("  Using SYNTHETIC initialization...")

        # Create environment
        env = JobMarketEnv(**env_params)

        # If empirical calibration provided delta_interview0_sq, apply it
        if use_empirical and 'delta_interview0_sq' in empirical_params:
            env.screening.delta0_sq = float(empirical_params['delta_interview0_sq'])

        run_name_prefix = base_run_dir.replace("runs/", "") if base_run_dir.startswith("runs/") else base_run_dir
        trainer = IPPOTrainer(
            env=env,
            lr=3e-3,
            run_name=f"{run_name_prefix}/market_{n_firms}f_{n_workers}w",
            seed=seed
        )

        # Shorter training for market structure experiments
        trainer.train(
            total_timesteps=8_000,  # Even shorter for market structure
            n_steps=256,  # Smaller rollouts
            n_epochs=4,  # Fewer epochs
            batch_size=32,  # Smaller batches
            log_interval=10,
            save_interval=100000,
            eval_interval=0
        )

        # Evaluate
        eval_data = extract_evaluation_metrics(env, trainer, n_episodes=15)  # Fewer eval episodes
        results[(n_firms, n_workers)] = eval_data
        run_dirs.append(trainer.run_dir)

        print(f"  Unemployment: {eval_data['unemployment_rate']:.2%}")
        print(f"  Avg wage: {eval_data['avg_wage']:.3f}")
        print(f"  Avg profit: {eval_data['avg_profit']:.3f}")

    # Create combined output directory for comparison tables
    comparison_dir = os.path.join(base_run_dir, "market_structure_comparison")
    results_dir = os.path.join(comparison_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Generate table
    print("\nGenerating market structure table...")
    create_market_structure_table(
        results=results,
        save_path=os.path.join(results_dir, "market_structure_table.csv")
    )

    # Save results
    with open(os.path.join(results_dir, "market_structure_results.pkl"), 'wb') as f:
        pickle.dump(results, f)

    print(f"\n✓ Comparison outputs saved to: {comparison_dir}/")
    print(f"✓ Individual runs: {', '.join(run_dirs)}")
    return results, comparison_dir


def main(use_empirical: bool = False):
    """
    Run all experiments.

    Args:
        use_empirical: If True, use empirical data from SDE for initialization.
                      If False, use synthetic defaults.
    """
    print("\n" + "="*80)
    print(" "*20 + "HireRL EXPERIMENT SUITE (FAST VERSION)")
    print("="*80)
    print("\nThis will run all experiments for the paper:")
    print("  1. Learning Dynamics (Section 3.2)")
    print("  2. Firing Cost Sensitivity (Section 3.3)")
    print("  3. Market Structure (Section 3.4)")
    print("\nNOTE: Using smaller environments for faster testing!")
    print("  - 2-3 companies (instead of 5)")
    print("  - 10-20 workers (instead of 50)")
    print("  - 20 timesteps (instead of 100)")
    print("  - 8k-10k training steps (instead of 50k-100k)")

    init_mode = "EMPIRICAL (real data)" if use_empirical else "SYNTHETIC"
    print(f"\nInitialization mode: {init_mode}")

    print("\nEstimated time: ~10-15 minutes on M1 MacBook Pro")
    print("="*80 + "\n")

    # Create base runs directory with timestamp
    import time
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    suite_name = f"experiment_suite_{timestamp}"
    run_suite_dir = f"runs/{suite_name}"

    print(f"\n📁 All outputs will be saved under: {run_suite_dir}/\n")

    baseline_run_name = f"{suite_name}/baseline_learning_dynamics"
    baseline_dir, baseline_data = experiment_1_baseline(seed=42, use_empirical=use_empirical, run_name=baseline_run_name)

    firing_results, firing_comparison_dir = experiment_2_firing_costs(seed=42, use_empirical=use_empirical, base_run_dir=run_suite_dir)
    market_results, market_comparison_dir = experiment_3_market_structure(seed=42, use_empirical=use_empirical, base_run_dir=run_suite_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run HireRL experiments')
    parser.add_argument(
        '--empirical',
        action='store_true',
        help='Use empirical data from SDE for initialization (default: synthetic)'
    )
    parser.add_argument(
        '--synthetic',
        action='store_true',
        help='Use synthetic initialization (default: True)'
    )
    args = parser.parse_args()

    # Default to synthetic unless --empirical is specified
    use_empirical = args.empirical and not args.synthetic

    main(use_empirical=use_empirical)
