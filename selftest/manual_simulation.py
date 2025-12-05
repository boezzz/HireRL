"""
Manual simulation of the HireRL environment without neural networks.

Run with:
    python selftest/manual_simulation.py
"""

import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add repo root so we can import both realzoo and real_data_init
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "realzoo"))
sys.path.insert(0, ROOT)  # allow `import real_data_init.sde`

import numpy as np
from hirerl import JobMarketEnv
from real_data_init.sde import (
    firms_type_config,
    initialize_firms,
    to_env_capacities,
    estimate_size_wage_premia,
    load_wage_exp,
    wage_variance_ratio,
    wage_variance_ratio_path,
    calibrate_signal_noise,
)


def normalize_action_mask(action_mask, action_size: int) -> np.ndarray:
    """Ensure action mask is a boolean vector of length action_size."""
    mask_arr = np.asarray(action_mask).astype(bool).flatten()
    if mask_arr.size != action_size:
        fill_val = bool(mask_arr[0]) if mask_arr.size > 0 else True
        mask_arr = np.full(action_size, fill_val, dtype=bool)
    return mask_arr


def choose_manual_action(action_space, action_mask, max_cost: float, num_workers: int) -> np.ndarray | int:
    """
    Select a deterministic action.

    Continuous: returns vector of costs (one per worker), each set to half max cost
    Discrete: choose highest valid index (scalar, applied to all workers by env)
    """
    mask_arr = normalize_action_mask(action_mask, getattr(action_space, "n", len(action_mask)))
    if isinstance(getattr(action_space, "n", None), int):
        # Discrete mode: return scalar index
        valid_indices = [i for i, v in enumerate(mask_arr) if v]
        return int(valid_indices[-1]) if valid_indices else 0
    # Continuous mode: return vector of per-worker costs
    return np.full(num_workers, 0.5 * max_cost, dtype=np.float32)


def print_worker_metrics(infos: Dict[str, Dict], step: int):
    """Log sigma_hat, sigma_tilde, sigma_true per worker."""
    for agent, info in infos.items():
        metrics = info.get("worker_metrics", [])
        if not metrics:
            continue
        print(f"\n[Step {step}] Agent {agent} worker metrics:")
        for entry in metrics:
            print(f"  worker {entry['worker_id']:>2} | "
                f"sigma_true={entry['sigma_true']:+.3f} | "
                f"sigma_hat={entry['sigma_hat']:+.3f} | "
                f"sigma_tilde={entry['sigma_tilde']:+.3f} | "
                f"wage={entry['wage']:+.2f}")


def compute_profit_signal(profit: float, sigma_true: float) -> float:
    """Diagnostic helper (currently unused hook)."""
    return float(sigma_true)


def load_realistic_env(num_firms: int = 5,
                       employees_per_agent: float = 1_000.0,
                       random_state: int | None = 42,
                       seed: int | None = 42) -> Tuple[JobMarketEnv, List[str]]:
    """
    Build an env seeded by real firm sizes and wage premia.
    employees_per_agent 就是一把比例尺。公式很简单：

  capacity_i = round(公司真实员工数_i / employees_per_agent)

  - 如果设得小（比如 500），真实 5,000 人的公司 → capacity ≈ 10；1,000,000 人的公司 → capacity ≈ 2,000，巨头在环境里也很大，总 worker 数会暴涨。
  - 如果设得大（比如 5,000），同样的 5,000 人公司 → capacity ≈ 1；1,000,000 人公司 → capacity ≈ 200，巨头被压缩得更紧，总 worker 数变小。
    """
    firms_df = initialize_firms(num_firms=num_firms,
                                type_config=firms_type_config,
                                random_state=random_state)
    capacities = to_env_capacities(firms_df, employees_per_agent=employees_per_agent)
    firm_types = firms_df["firm_type"].tolist()
    try:
        premia = estimate_size_wage_premia()
    except Exception as e:
        print(f"[warn] size premia estimation skipped ({e}); using neutral premia.")
        premia = {"small": 1.0, "medium": 1.0, "large": 1.0, "generic": 1.0}

    # Calibrate signal noise to empirical wage variance drop if possible
    profit_noise_var = 0.05
    delta_interview0_sq = 0.4
    wage_scale = 1.0
    try:
        wage_df = load_wage_exp()
        # Use yearly buckets (bin_step=1) so each timestep aligns to ~1 year
        path = wage_variance_ratio_path(wage_df, bin_step=1)
        if path:
            periods = max(path.keys())
            best = calibrate_signal_noise(
                target_path=path, periods=periods, n_workers=5000, seed=0
            )
        else:
            ratio = wage_variance_ratio(wage_df)
            best = None if ratio is None else calibrate_signal_noise(
                target_ratio=ratio, periods=3, n_workers=5000, seed=0
            )
        if best:
            delta_interview0_sq, profit_noise_var, _, _, _ = best
            print(f"[info] calibrated noise: delta_interview0_sq={delta_interview0_sq}, delta_profit_sq={profit_noise_var}")
        wage_mean = float(wage_df["salary"].mean())
        wage_scale = wage_mean if wage_mean > 0 else wage_scale
    except Exception as e:
        print(f"[warn] noise calibration skipped ({e}); using defaults.")

    num_workers = sum(capacities)
    env = JobMarketEnv(
        num_companies=len(capacities),
        num_workers=num_workers,
        max_workers_per_company=max(capacities) if capacities else 1,
        firm_capacities=capacities,
        firm_types=firm_types,
        firm_type_premia=premia,
        ability_dim=1,
        action_mode="continuous",
        max_interview_cost=2.0,
        profit_noise_var=profit_noise_var,
        wage_scale=wage_scale,
        max_timesteps=100,
        seed=seed,
    )
    env.screening.delta0_sq = float(delta_interview0_sq)
    env.screening._rng = np.random.RandomState(seed)
    return env, firm_types


def run_manual_simulation():
    env, firm_types = load_realistic_env(num_firms=5, employees_per_agent=4000, random_state=42, seed=42)

    observations, infos = env.reset(seed=42)
    print_worker_metrics(infos, step=0)
    print(f"Firm types: {firm_types}")

    csv_rows: List[Dict] = []
    interview_events = defaultdict(lambda: {'t': [], 'value': []})
    firing_events = defaultdict(list)
    hiring_events = defaultdict(list)
    hire_counts = defaultdict(list)
    fire_counts = defaultdict(list)
    finance_series = defaultdict(lambda: {'t': [], 'profit': [], 'wage': [], 'firing_cost': [], 'reward': []})
    wage_series = defaultdict(lambda: {'t': [], 'wage': []})
    workforce_sizes = defaultdict(lambda: {'t': [], 'size': []})
    profit_signal_series = defaultdict(lambda: {'t': [], 'profit_signal': []})
    total_hires_per_step: List[int] = []
    total_fires_per_step: List[int] = []
    step_indices: List[int] = []
    action_series = defaultdict(lambda: {'t': [], 'action': []})
    vx_series = defaultdict(lambda: {'t': [], 'vx': [], 'k1': []})
    horizon = 100

    # Log reset (t=0) state so sigma_hat and sigma_tilde starting points are visible
    step = 0
    for agent in env.possible_agents:
        info = infos.get(agent, {})
        if info:
            for entry in info.get('worker_metrics', []):
                profit_val = entry.get('profit')
                profit_signal = compute_profit_signal(profit=profit_val, sigma_true=entry['sigma_true'])
                if profit_signal is not None:
                    profit_signal_series[(agent, entry['worker_id'])]['t'].append(step)
                    profit_signal_series[(agent, entry['worker_id'])]['profit_signal'].append(profit_signal)
                vx_val = entry.get('vx')
                k1_val = entry.get('k1')
                if vx_val is not None and k1_val is not None:
                    key = (agent, entry['worker_id'])
                    vx_series[key]['t'].append(step)
                    vx_series[key]['vx'].append(vx_val)
                    vx_series[key]['k1'].append(k1_val)
                csv_rows.append({
                    'timestep': step,
                    'agent': agent,
                    'worker_id': entry['worker_id'],
                    'sigma_true': entry['sigma_true'],
                    'sigma_hat': entry['sigma_hat'],
                    'sigma_tilde': entry['sigma_tilde'],
                    'wage': entry['wage'],
                    'profit': profit_val,
                    'profit_signal': profit_signal,
                    'action_value': None,
                    'interview_cost': entry['interview_cost'],
                    'vx': vx_val,
                    'k1': k1_val,
                })
        hire_counts[agent].append(0)
        fire_counts[agent].append(0)
        # Workforce size at t=0
        company_idx = int(agent.split('_')[1])
        workforce_sizes[agent]['t'].append(step)
        workforce_sizes[agent]['size'].append(int(np.sum(env.employed_by == company_idx)))
    total_hires_per_step.append(0)
    total_fires_per_step.append(0)
    step_indices.append(0)
    prev_employed_by = env.employed_by.copy()

    base_interview_var = float(env.screening.interview_var(0.0))

    for step in range(1, horizon + 1):
        actions = {}
        for agent in env.agents:
            action_space = env.action_space(agent)
            action_mask = observations[agent]["action_mask"]
            actions[agent] = choose_manual_action(
                action_space, action_mask, env.max_interview_cost, env.num_workers
            )
            # Log average interview cost (mean of non-zero costs)
            action_vec = np.asarray(actions[agent]).reshape(-1)
            interviewed = action_vec > 0.0
            avg_cost = float(np.mean(action_vec[interviewed])) if interviewed.any() else 0.0
            action_series[agent]['t'].append(step)
            action_series[agent]['action'].append(avg_cost)

        observations, rewards, terminations, truncations, infos = env.step(actions)

        print(f"\nStep {step} rewards: {rewards}")
        for agent in env.possible_agents:
            info = infos.get(agent, {})
            if info:
                print(
                    f"  {agent}: profit={info.get('last_step_profit', 0.0):+.2f}, "
                    f"wages={info.get('last_step_wage', 0.0):+.2f}, "
                    f"firing_cost={info.get('last_step_firing_cost', 0.0):+.2f}, "
                    f"reward={info.get('last_step_reward', 0.0):+.2f}"
                )
                finance_series[agent]['t'].append(step)
                finance_series[agent]['profit'].append(info.get('last_step_profit', 0.0))
                finance_series[agent]['wage'].append(info.get('last_step_wage', 0.0))
                finance_series[agent]['firing_cost'].append(info.get('last_step_firing_cost', 0.0))
                finance_series[agent]['reward'].append(info.get('last_step_reward', 0.0))
                wage_series[agent]['t'].append(step)
                wage_series[agent]['wage'].append(info.get('last_step_wage', 0.0))
        print_worker_metrics(infos, step=step)
        step_total_hires = 0
        step_total_fires = 0
        agent_hires = {agent: 0 for agent in env.possible_agents}
        agent_fires = {agent: 0 for agent in env.possible_agents}

        for agent in env.possible_agents:
            info = infos.get(agent)
            if info:
                for entry in info.get('worker_metrics', []):
                    vx_val = entry.get('vx')
                    k1_val = entry.get('k1')
                    profit_val = entry.get('profit')
                    profit_signal = compute_profit_signal(profit=profit_val, sigma_true=entry['sigma_true']) if profit_val is not None else None
                    if profit_signal is not None:
                        profit_signal_series[(agent, entry['worker_id'])]['t'].append(step)
                        profit_signal_series[(agent, entry['worker_id'])]['profit_signal'].append(profit_signal)
                    if vx_val is not None and k1_val is not None:
                        key = (agent, entry['worker_id'])
                        vx_series[key]['t'].append(step)
                        vx_series[key]['vx'].append(vx_val)
                        vx_series[key]['k1'].append(k1_val)
                    interview_var = entry.get('interview_cost')
                    # Extract the specific action (interview cost) for this worker
                    action_for_worker = None
                    if agent in actions:
                        action_vec = np.asarray(actions[agent]).reshape(-1)
                        worker_id = entry['worker_id']
                        if 0 <= worker_id < len(action_vec):
                            action_for_worker = float(action_vec[worker_id])
                    csv_rows.append({
                        'timestep': step,
                        'agent': agent,
                        'worker_id': entry['worker_id'],
                        'sigma_true': entry['sigma_true'],
                        'sigma_hat': entry['sigma_hat'],
                        'sigma_tilde': entry['sigma_tilde'],
                        'wage': entry['wage'],
                        'profit': profit_val,
                        'profit_signal': profit_signal,
                        'action_value': action_for_worker,
                        'interview_var': interview_var,
                        'vx': vx_val,
                        'k1': k1_val,
                    })
                    if interview_var is not None and interview_var < base_interview_var - 1e-8:
                        key = (agent, entry['worker_id'])
                        interview_events[key]['t'].append(step)
                        interview_events[key]['value'].append(interview_var)

        # Detect hires/fires based on employed_by changes
        new_employed_by = env.employed_by.copy()
        for worker_id, firm_id in enumerate(new_employed_by):
            prev_firm = prev_employed_by[worker_id]
            if prev_firm < 0 and firm_id >= 0:
                agent = f"company_{firm_id}"
                hiring_events[(agent, worker_id)].append(step)
                agent_hires[agent] += 1
                step_total_hires += 1
            elif prev_firm >= 0 and firm_id < 0:
                agent = f"company_{prev_firm}"
                firing_events[(agent, worker_id)].append(step)
                agent_fires[agent] += 1
                step_total_fires += 1
        prev_employed_by = new_employed_by

        for agent in env.possible_agents:
            hire_counts[agent].append(agent_hires.get(agent, 0))
            fire_counts[agent].append(agent_fires.get(agent, 0))

        total_hires_per_step.append(step_total_hires)
        total_fires_per_step.append(step_total_fires)
        step_indices.append(step)

        # Workforce size after this step
        for agent in env.possible_agents:
            company_idx = int(agent.split('_')[1])
            workforce_sizes[agent]['t'].append(step)
            workforce_sizes[agent]['size'].append(int(np.sum(env.employed_by == company_idx)))
        # Only break early if the env actually reports termination/truncation flags
        if (terminations and all(terminations.values())) or (truncations and all(truncations.values())):
            print("Episode ended early.")
            break

    save_path = Path(__file__).with_name("manual_simulation_sigma.csv")
    with save_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'timestep', 'agent', 'worker_id',
            'sigma_true', 'sigma_hat', 'sigma_tilde', 'wage', 'profit', 'profit_signal', 'action_value', 'interview_var',
            'vx', 'k1'
        ])
        writer.writeheader()
        for row in csv_rows:
            # Backward compatibility: allow older key name
            if 'interview_var' not in row and 'interview_cost' in row:
                row['interview_var'] = row.pop('interview_cost')
            writer.writerow(row)
    print(f"\nSaved sigma/action log to {save_path}")

    sigma_series = defaultdict(lambda: {'t': [], 'sigma_true': [], 'sigma_hat': [], 'sigma_tilde': []})
    for row in csv_rows:
        key = (row['agent'], row['worker_id'])
        sigma_series[key]['t'].append(row['timestep'])
        sigma_series[key]['sigma_true'].append(row['sigma_true'])
        sigma_series[key]['sigma_hat'].append(row['sigma_hat'])
        sigma_series[key]['sigma_tilde'].append(row['sigma_tilde'])

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
        hire_t = hiring_events.get((agent, worker_id), [])
        if hire_t:
            for t in hire_t:
                plt.axvline(t, color='blue', linestyle='-.', alpha=0.4, label='hire')
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

    # Plot vx and k1 per worker if available
    for (agent, worker_id), data in vx_series.items():
        if not data['t']:
            continue
        plt.figure()
        plt.plot(data['t'], data['vx'], label='vx')
        plt.plot(data['t'], data['k1'], label='k1')
        plt.xlabel('timestep')
        plt.ylabel('value')
        plt.title(f'{agent} worker {worker_id} vx/k1')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f'{agent}_worker{worker_id}_vx_k1.png')
        plt.close()

    # Profit signal s(p) per worker
    for (agent, worker_id), data in profit_signal_series.items():
        if not data['t']:
            continue
        plt.figure()
        plt.plot(data['t'], data['profit_signal'], label='s(p)')
        plt.axhline(0.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.6)
        plt.xlabel('timestep')
        plt.ylabel('s(p)')
        plt.title(f'{agent} worker {worker_id} s(p)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f'{agent}_worker{worker_id}_profit_signal.png')
        plt.close()

    # Finance series per firm
    for agent, data in finance_series.items():
        if not data['t']:
            continue
        plt.figure()
        plt.plot(data['t'], data['profit'], label='profit')
        plt.plot(data['t'], data['wage'], label='wage')
        plt.plot(data['t'], data['firing_cost'], label='firing_cost')
        plt.plot(data['t'], data['reward'], label='reward (net)')
        plt.xlabel('timestep')
        plt.ylabel('value')
        plt.title(f'{agent} finance per timestep')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f'{agent}_finance.png')
        plt.close()

    # Wage + firing cost per timestep per firm
    for agent, data in finance_series.items():
        if not data['t']:
            continue
        plt.figure()
        plt.plot(data['t'], data['wage'], label='wage')
        plt.plot(data['t'], data['firing_cost'], label='firing_cost')
        plt.xlabel('timestep')
        plt.ylabel('value')
        plt.title(f'{agent} wage & firing cost per timestep')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f'{agent}_wage_and_firing_cost.png')
        plt.close()

    # Workforce size per timestep per firm
    for agent, data in workforce_sizes.items():
        if not data['t']:
            continue
        plt.figure()
        plt.step(data['t'], data['size'], where='mid', label='workforce size')
        plt.xlabel('timestep')
        plt.ylabel('headcount')
        plt.title(f'{agent} workforce size per timestep')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f'{agent}_workforce_size.png')
        plt.close()

    if step_indices:
        union_agents = sorted(set(hire_counts.keys()) | set(fire_counts.keys()))
        for agent in union_agents:
            hires = hire_counts.get(agent, [])
            fires = fire_counts.get(agent, [])
            if not hires and not fires:
                continue
            if not any(hires) and not any(fires):
                continue
            series_len = max(len(hires), len(fires))
            if series_len == 0:
                continue
            steps = step_indices[:series_len]
            plt.figure()
            plt.step(steps, hires, where='mid', label='hires per step')
            plt.step(steps, fires, where='mid', label='firings per step')
            plt.xlabel('timestep')
            plt.ylabel('count')
            plt.title(f'{agent} hires/firings per timestep')
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_dir / f'{agent}_hire_fire_counts.png')
            plt.close()

        plt.figure()
        plt.step(step_indices, total_hires_per_step, where='mid', label='total hires')
        plt.step(step_indices, total_fires_per_step, where='mid', label='total firings')
        plt.xlabel('timestep')
        plt.ylabel('count')
        plt.title('Aggregate hires/firings per timestep')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / 'aggregate_hire_fire_counts.png')
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
