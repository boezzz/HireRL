# HireRL: Multi-Agent Reinforcement Learning for Job Market Search and Matching

A comprehensive MARL environment implementing job market dynamics with information asymmetry, costly screening, and employer learning.

## Overview

HireRL implements the theoretical framework from "Multi-agent Reinforcement Learning benchmark for job market search and matching" as a PettingZoo Parallel environment. The environment models:

- **Strategic Agents**: Companies learning hiring policies through RL
- **Environment**: Worker pool with private abilities and public signals
- **Information Asymmetry**: Workers know true ability σ_j, firms observe noisy signals σ̂_j,0
- **Costly Screening**: Firms invest to better estimate worker abilities
- **Employer Learning**: Firms update beliefs based on performance observations

### Key Research Questions

1. **Optimal Screening**: How should firms choose their level of investment in screening to infer worker ability?
2. **Greedy vs Stable Matching**: How does myopic profit-maximizing affect time-to-match compared to stable matching?

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python >= 3.8
- numpy >= 1.20.0
- gymnasium >= 0.28.0
- pettingzoo >= 1.23.0
- torch >= 2.0.0 (for PPO training)
- tensorboard >= 2.10.0 (for experiment tracking and visualization)

## Quick Start

```python
from pettingzoo.hirerl import JobMarketEnv
from pettingzoo.policies import GreedyPolicy

# Create environment
env = JobMarketEnv(
    num_companies=3,
    num_workers=10,
    max_workers_per_company=5
)

# Run episode with greedy policy
observations, infos = env.reset()

for _ in range(100):
    # Simple greedy policy: hire best workers
    actions = {agent: 0 for agent in env.agents}  # no-op
    observations, rewards, terminations, truncations, infos = env.step(actions)

    if all(terminations.values()):
        break
```

## Training with PPO

Train companies to learn optimal hiring policies using Independent PPO (IPPO):

```bash
# Train with default settings (1M steps, 3 companies, 10 workers)
python train_ppo.py

# View training progress in TensorBoard
tensorboard --logdir=runs
```

### PPO Features

The PPO implementation includes CleanRL best practices:
- **Action Masking**: Neural network masks invalid actions before sampling
- **Orthogonal Initialization**: Better weight initialization for stability
- **TensorBoard Logging**: Track episodic returns, losses, entropy, KL divergence
- **Learning Rate Annealing**: Linear decay over training
- **Explained Variance**: Monitor value function prediction quality
- **Clipped Value Loss**: Prevent value function over-updating
- **Unique Run IDs**: Each training run gets timestamped directory

### Key Metrics to Monitor

- **Episodic Returns**: Total reward per episode for each company
- **Explained Variance**: >0.5 indicates good value function (higher is better)
- **Entropy**: Should gradually decrease (exploration → exploitation)
- **Approx KL**: Should stay <0.1 (policy update stability)
- **Clip Fraction**: 0.1-0.3 indicates healthy PPO clipping

## Running Tests

```bash
# Run PettingZoo compliance tests
python tests/test_pettingzoo_compliance.py

# Run basic verification tests
python tests/test_simple.py

# Compare baseline policies
python tests/test_baseline_policies.py
```

### PettingZoo Compliance

The environment passes all official PettingZoo tests:
- ✅ Parallel API compliance
- ✅ Seed determinism
- ✅ Action masking validation
- ✅ Observation space consistency
- ✅ Render functionality

## Environment Details

### Action Space

Companies can take four types of actions:

- **NO_OP** (0): Do nothing
- **FIRE** (1 to N): Fire worker j
- **OFFER** (N+1 to 2N): Make wage offer to unemployed worker j
- **INTERVIEW** (2N+1 to 3N): Screen worker j before hiring

Action encoding: `Discrete(1 + 3*N)` where N = number of workers

**Action Masking**: Invalid actions are automatically masked:
- FIRE: Only valid if worker is employed by this company
- OFFER: Only valid if worker is unemployed AND company has capacity
- INTERVIEW: Only valid if worker is unemployed
- NO_OP: Always valid

### Observation Space

Each company observes a dictionary containing:

**Observation** (`Box`):
- **Public Information**:
  - σ̂_j,t: Public ability signals for all workers
  - exp_j,t: Experience levels
  - τ_j,t: Tenure (time employed)
  - Employment status and current wages
- **Private Information**:
  - Belief about each worker's ability (mean and variance)
  - Own workforce and profit

**Action Mask** (`MultiBinary`):
- Binary mask indicating valid actions (1) vs invalid actions (0)
- Enables PPO agents to avoid invalid actions during training

### Reward Structure

```
r_i,t = Σ_{j ∈ E_i,t} (p_ij,t - w_ij,t) - c_fire - c_hire - c_screen

where:
- p_ij,t = σ_j + β*log(1 + exp_j,t): Match-specific profit
- w_ij,t: Wage paid to worker j
- c_fire, c_hire, c_screen: Action costs
```

## Key Features

### 1. Worker Dynamics

Workers have:
- **True ability** σ_j ~ N(0, 1): Private, static
- **Public signal** σ̂_j,0 = σ_j + ε: Noisy resume/CV
- **Experience** exp_j,t: Grows while employed at rate g(σ_j) = g0 + g1*σ_j
- **Tenure** τ_j,t: Total time employed
- **Public signal update**: σ̂_j,t = σ̂_j,0 + γ*τ_j,t

### 2. Screening Mechanism

Firms can invest cost c to get better ability estimates:

```
σ_estimate = σ̂_j,0 + precision(c) * (σ_j - σ̂_j,0) + ε

where precision(c) ∈ [0, 1] increases with cost
```

Screening technologies available: SQRT (default), LINEAR, LOGARITHMIC, SIGMOID

### 3. Employer Learning

Firms perform Bayesian updating:
1. **Prior**: Initialize beliefs from public signals
2. **Screening Update**: Incorporate interview results
3. **Performance Update**: Update from observed profits

### 4. Deterministic Quit Rule

Workers quit if they observe comparable workers earning higher wages. This creates wage competition.

### 5. Stable Matching Alternative

Implements Gale-Shapley deferred acceptance for comparison with greedy matching.

## Project Structure

```
HireRL/
├── pettingzoo/
│   ├── hirerl.py           # Main environment (with action masking)
│   ├── workers.py          # Worker pool management
│   ├── screening.py        # Screening mechanism & Bayesian beliefs
│   ├── matching.py         # Stable matching algorithms
│   ├── policies.py         # Baseline policies (action masking compatible)
│   └── utils.py            # Logging & visualization
├── tests/
│   ├── test_pettingzoo_compliance.py  # PettingZoo API tests
│   ├── test_simple.py                 # Basic verification
│   └── test_baseline_policies.py      # Policy comparison
├── train_ppo.py            # IPPO training with CleanRL best practices
├── requirements.txt        # Dependencies (includes TensorBoard)
├── README.md
└── runs/                   # TensorBoard logs (generated)
```

## Baseline Policies

Six baseline policies for testing (all support action masking):

1. **RandomPolicy**: Random action selection from valid actions only
2. **GreedyPolicy**: Hire best available, fire worst performers
3. **NoScreeningPolicy**: Greedy strategy without interviews
4. **HighScreeningPolicy**: Always screen workers before hiring
5. **NeverFirePolicy**: Only hire, never fire workers
6. **HeuristicPolicy**: Rule-based strategy with screening threshold

All policies automatically respect action masks and only select valid actions.

## Economic Concepts Implemented

- **Information Asymmetry**: Signaling (Spence 1978) and Screening (Stiglitz 1975)
- **Employer Learning**: Altonji & Pierret (2001)
- **Stable Matching**: Gale-Shapley deferred acceptance
- **Search Frictions**: Diamond-Mortensen-Pissarides framework
- **Wage Determination**: Mincer equation with ability and experience

## License

MIT License

## Citation

If you use this environment in your research, please cite:

```bibtex
@article{hirerl2024,
  title={Multi-agent Reinforcement Learning benchmark for job market search and matching},
  author={Zong, Haijing and Zhou, Boyang},
  year={2024}
}
```