"""
Plotting utilities for HireRL experiments.
Generates matplotlib figures for paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
from tensorboard.backend.event_processing import event_accumulator

# Set style for publication-quality plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.figsize'] = (6, 4)


def load_tensorboard_scalar(log_dir: str, tag: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load scalar data from TensorBoard logs."""
    ea = event_accumulator.EventAccumulator(log_dir)
    ea.Reload()

    if tag not in ea.Tags()['scalars']:
        print(f"Warning: Tag '{tag}' not found. Available: {ea.Tags()['scalars'][:5]}...")
        return np.array([]), np.array([])

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])

    return steps, values


def plot_learning_curves(
    log_dir: str,
    save_path: str = "figures/learning_curves.png"
):
    """
    Figure for Section 3.2: Learning Dynamics during training.

    Shows:
    - Episodic return over training
    - Average interview cost over training
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel A: Episodic return
    steps, returns = load_tensorboard_scalar(log_dir, "charts/total_episodic_return")
    if len(steps) > 0:
        axes[0].plot(steps, returns, linewidth=1.5, color='#2E86AB')
        axes[0].set_xlabel('Training Steps')
        axes[0].set_ylabel('Total Episodic Return')
        axes[0].set_title('(a) Learning Curve')
        axes[0].grid(True, alpha=0.3)

    # Panel B: Interview cost
    steps, costs = load_tensorboard_scalar(log_dir, "interview/avg_cost")
    if len(steps) > 0:
        axes[1].plot(steps, costs, linewidth=1.5, color='#A23B72')
        axes[1].set_xlabel('Training Steps')
        axes[1].set_ylabel('Average Interview Cost')
        axes[1].set_title('(b) Interview Cost Evolution')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_learned_policy_analysis(
    eval_data: Dict[str, Any],
    save_path: str = "figures/learned_policy.png"
):
    """
    Figure for Section 3.2: Analysis of learned policy behavior.

    Shows:
    - Hiring/firing stability across episodes
    - True ability vs interview cost correlation
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Panel A: Hiring/Firing rates per episode
    episodes = eval_data['episodes']
    hiring_rates = eval_data['hiring_rates']
    firing_rates = eval_data['firing_rates']

    axes[0].plot(episodes, hiring_rates, label='Hiring Rate',
                 linewidth=2, color='#06A77D', marker='o', markersize=3)
    axes[0].plot(episodes, firing_rates, label='Firing Rate',
                 linewidth=2, color='#D62246', marker='s', markersize=3)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Rate (workers/timestep)')
    axes[0].set_title('(a) Employment Dynamics Stability')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel B: True ability vs avg interview cost
    true_abilities = eval_data['true_abilities']
    avg_interview_costs = eval_data['avg_interview_costs']

    axes[1].scatter(true_abilities, avg_interview_costs,
                   alpha=0.6, s=50, color='#F18F01')

    # Add trend line
    z = np.polyfit(true_abilities, avg_interview_costs, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(true_abilities.min(), true_abilities.max(), 100)
    axes[1].plot(x_trend, p(x_trend),
                linestyle='--', color='#C73E1D', linewidth=2,
                label=f'Trend (r={np.corrcoef(true_abilities, avg_interview_costs)[0,1]:.3f})')

    axes[1].set_xlabel('Worker True Ability (σ)')
    axes[1].set_ylabel('Avg Interview Cost')
    axes[1].set_title('(b) Interview Cost vs True Ability')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def plot_firing_cost_comparison(
    results_by_cost: Dict[float, Dict[str, Any]],
    save_path: str = "figures/firing_cost_sensitivity.png"
):
    """
    Figure for Section 3.3: Firing cost sensitivity analysis.

    Shows comparison across different firing cost multipliers.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    costs = sorted(results_by_cost.keys())

    # Extract metrics
    firing_rates = [results_by_cost[c]['firing_rate'] for c in costs]
    profits = [results_by_cost[c]['avg_profit'] for c in costs]
    interview_costs = [results_by_cost[c]['avg_interview_cost'] for c in costs]
    employment_duration = [results_by_cost[c]['avg_employment_duration'] for c in costs]

    # Panel A: Firing rate
    axes[0, 0].plot(costs, firing_rates, marker='o', linewidth=2,
                    markersize=8, color='#D62246')
    axes[0, 0].set_xlabel('Firing Cost Multiplier (× wage)')
    axes[0, 0].set_ylabel('Firing Rate (per episode)')
    axes[0, 0].set_title('(a) Firing Rate vs Severance Cost')
    axes[0, 0].grid(True, alpha=0.3)

    # Panel B: Average profit
    axes[0, 1].plot(costs, profits, marker='s', linewidth=2,
                    markersize=8, color='#06A77D')
    axes[0, 1].set_xlabel('Firing Cost Multiplier (× wage)')
    axes[0, 1].set_ylabel('Average Profit')
    axes[0, 1].set_title('(b) Profit vs Severance Cost')
    axes[0, 1].grid(True, alpha=0.3)

    # Panel C: Interview spending
    axes[1, 0].plot(costs, interview_costs, marker='^', linewidth=2,
                    markersize=8, color='#F18F01')
    axes[1, 0].set_xlabel('Firing Cost Multiplier (× wage)')
    axes[1, 0].set_ylabel('Avg Interview Cost')
    axes[1, 0].set_title('(c) Interview Spending vs Severance Cost')
    axes[1, 0].grid(True, alpha=0.3)

    # Panel D: Employment duration
    axes[1, 1].plot(costs, employment_duration, marker='d', linewidth=2,
                    markersize=8, color='#A23B72')
    axes[1, 1].set_xlabel('Firing Cost Multiplier (× wage)')
    axes[1, 1].set_ylabel('Avg Employment Duration')
    axes[1, 1].set_title('(d) Employment Duration vs Severance Cost')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def create_market_structure_table(
    results: Dict[Tuple[int, int], Dict[str, Any]],
    save_path: str = "results/market_structure_table.csv"
):
    """
    Create Table for Section 3.4: Market structure comparison.

    Returns LaTeX table code and saves CSV.
    """
    # Create DataFrame
    rows = []
    for (n_firms, n_workers), metrics in sorted(results.items()):
        rows.append({
            'Firms': n_firms,
            'Workers': n_workers,
            'Unemp. Rate': f"{metrics['unemployment_rate']:.1%}",
            'Avg Wage': f"{metrics['avg_wage']:.2f}",
            'Interview Cost': f"{metrics['avg_interview_cost']:.3f}",
            'Avg Profit': f"{metrics['avg_profit']:.2f}",
        })

    df = pd.DataFrame(rows)

    # Save CSV
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved CSV: {save_path}")

    # Generate LaTeX table
    latex = df.to_latex(index=False, escape=False,
                       caption="Market Structure and Outcomes",
                       label="tab:market_summary")

    # Save LaTeX
    latex_path = save_path.replace('.csv', '.tex')
    with open(latex_path, 'w') as f:
        f.write(latex)
    print(f"Saved LaTeX: {latex_path}")

    # Print to terminal
    print("\n" + "="*60)
    print("LaTeX Table Code:")
    print("="*60)
    print(latex)
    print("="*60 + "\n")

    return df


def create_firing_cost_table(
    results_by_cost: Dict[float, Dict[str, Any]],
    save_path: str = "results/firing_cost_table.csv"
):
    """Create comparison table for firing cost experiments."""
    rows = []
    for cost, metrics in sorted(results_by_cost.items()):
        rows.append({
            'Firing Cost (× wage)': cost,
            'Firing Rate': f"{metrics['firing_rate']:.2f}",
            'Avg Profit': f"{metrics['avg_profit']:.2f}",
            'Interview Cost': f"{metrics['avg_interview_cost']:.3f}",
            'Employment Duration': f"{metrics['avg_employment_duration']:.1f}",
        })

    df = pd.DataFrame(rows)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Saved: {save_path}")

    latex = df.to_latex(index=False, escape=False,
                       caption="Firing Cost Sensitivity Analysis",
                       label="tab:firing_cost")

    latex_path = save_path.replace('.csv', '.tex')
    with open(latex_path, 'w') as f:
        f.write(latex)

    print("\n" + "="*60)
    print("Firing Cost Table (LaTeX):")
    print("="*60)
    print(latex)
    print("="*60 + "\n")

    return df
