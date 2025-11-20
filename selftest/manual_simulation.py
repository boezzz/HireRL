"""
Manual simulation of the HireRL environment without neural networks.

Run with:
    python selftest/manual_simulation.py
"""

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'realzoo'))

import numpy as np
from hirerl import JobMarketEnv


def choose_manual_action(action_space, action_mask, max_cost: float) -> float:
    """
    Select a deterministic action for debugging.

    For continuous actions: pick half of the max cost (if mask allows).
    For discrete actions: choose the highest valid index (max interview cost).
    """
    if isinstance(getattr(action_space, "n", None), int):
        valid_indices = [i for i, v in enumerate(action_mask) if v == 1]
        return float(valid_indices[-1]) if valid_indices else 0.0

    # Continuous Box action
    return float(0.5 * max_cost if action_mask[0] == 1 else 0.0)


def print_worker_metrics(infos: Dict[str, Dict], step: int):
    """Log sigma_hat, sigma_tilde, sigma_true per worker."""
    for agent, info in infos.items():
        metrics = info.get("worker_metrics", [])
        if not metrics:
            continue
        print(f"\n[Step {step}] Agent {agent} worker metrics:")
        for entry in metrics:
            print(
                f"  worker {entry['worker_id']:>2} | "
                f"sigma_true={entry['sigma_true']:+.3f} | "
                f"sigma_hat={entry['sigma_hat']:+.3f} | "
                f"sigma_tilde={entry['sigma_tilde']:+.3f} | "
                f"wage={entry['wage']:+.2f}"
            )


def run_manual_simulation():
    env = JobMarketEnv(
        num_companies=2,
        num_workers=5,
        ability_dim=1,
        max_workers_per_company=3,
        action_mode="continuous",
        seed=123,
    )

    observations, infos = env.reset()
    print_worker_metrics(infos, step=0)

    csv_rows: List[Dict] = []
    interview_events = defaultdict(lambda: {'t': [], 'value': []})
    firing_events = defaultdict(list)
    horizon = 10
    for step in range(1, horizon + 1):
        actions = {}
        for agent in env.agents:
            action_space = env.action_space(agent)
            action_mask = observations[agent]["action_mask"]
            actions[agent] = choose_manual_action(
                action_space, action_mask, env.max_interview_cost
            )

        observations, rewards, terminations, truncations, infos = env.step(actions)

        print(f"\nStep {step} rewards: {rewards}")
        print_worker_metrics(infos, step=step)
        for agent, info in infos.items():
            for entry in info.get('worker_metrics', []):
                csv_rows.append({
                    'timestep': step,
                    'agent': agent,
                    'worker_id': entry['worker_id'],
                    'sigma_true': entry['sigma_true'],
                    'sigma_hat': entry['sigma_hat'],
                    'sigma_tilde': entry['sigma_tilde'],
                    'wage': entry['wage'],
                    'action_value': float(actions.get(agent)) if actions.get(agent) is not None else None,
                    'interview_cost': entry['interview_cost'],
                })
                if entry['interview_cost'] > 0:
                    key = (agent, entry['worker_id'])
                    interview_events[key]['t'].append(step)
                    interview_events[key]['value'].append(entry['interview_cost'])
            for worker_id in info.get("firings", []):
                firing_events[(agent, worker_id)].append(step)

        if all(terminations.values()) or all(truncations.values()):
            print("Episode ended early.")
            break

    save_path = Path(__file__).with_name("manual_simulation_sigma.csv")
    with save_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestep', 'agent', 'worker_id',
            'sigma_true', 'sigma_hat', 'sigma_tilde', 'wage', 'action_value', 'interview_cost'
        ])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSaved sigma/action log to {save_path}")

    sigma_series = defaultdict(lambda: {'t': [], 'sigma_true': [], 'sigma_hat': [], 'sigma_tilde': []})
    action_series = defaultdict(lambda: {'t': [], 'action': []})
    for row in csv_rows:
        key = (row['agent'], row['worker_id'])
        sigma_series[key]['t'].append(row['timestep'])
        sigma_series[key]['sigma_true'].append(row['sigma_true'])
        sigma_series[key]['sigma_hat'].append(row['sigma_hat'])
        sigma_series[key]['sigma_tilde'].append(row['sigma_tilde'])
        if row['action_value'] is not None:
            action_series[row['agent']]['t'].append(row['timestep'])
            action_series[row['agent']]['action'].append(row['action_value'])

    plot_dir = Path(__file__).with_name("manual_plots")
    plot_dir.mkdir(exist_ok=True)
    for (agent, worker_id), data in sigma_series.items():
        plt.figure()
        plt.plot(data['t'], data['sigma_true'], label='sigma_true')
        plt.plot(data['t'], data['sigma_hat'], label='sigma_hat')
        plt.plot(data['t'], data['sigma_tilde'], label='sigma_tilde')
        plt.xlabel('timestep')
        plt.ylabel('sigma')
        plt.title(f'{agent} worker {worker_id}')
        plt.legend()
        event_t = interview_events[(agent, worker_id)]['t']
        if event_t:
            for t in event_t:
                plt.axvline(t, color='green', linestyle='--', alpha=0.3, label='interview')
        fire_t = firing_events.get((agent, worker_id), [])
        if fire_t:
            for t in fire_t:
                plt.axvline(t, color='red', linestyle=':', alpha=0.4, label='firing')
        handles, labels = plt.gca().get_legend_handles_labels()
        seen = set()
        uniq_handles, uniq_labels = [], []
        for h, l in zip(handles, labels):
            if l not in seen:
                uniq_handles.append(h)
                uniq_labels.append(l)
                seen.add(l)
        plt.legend(uniq_handles, uniq_labels)
        plt.tight_layout()
        plt.savefig(plot_dir / f'{agent}_worker{worker_id}_sigma.png')
        plt.close()

    for agent, data in action_series.items():
        if not data['t']:
            continue
        plt.figure()
        plt.step(data['t'], data['action'], where='post')
        plt.xlabel('timestep')
        plt.ylabel('action (interview cost)')
        plt.title(f'{agent} actions')
        plt.tight_layout()
        plt.savefig(plot_dir / f'{agent}_actions.png')
        plt.close()
    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    run_manual_simulation()
