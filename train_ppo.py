"""
IPPO Training for Job Market Environment

- Action masking support
- TensorBoard logging
- Orthogonal initialization
- Explained variance tracking
- Learning rate annealing
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'realzoo'))

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional
import time
import random
from collections import defaultdict

from realzoo.hirerl import JobMarketEnv
from gymnasium.spaces import Box as GymBox, Discrete as GymDiscrete

# Empirical init / calibration helpers
try:
    from real_data_init import sde as sde_data
except Exception:
    sde_data = None  # fall back to synthetic defaults if unavailable


def set_global_seed(seed: int) -> None:
    """
    Seed all major RNG sources to keep experiments reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Orthogonal initialization for better training stability.

    Based on CleanRL's implementation.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """
    Actor-Critic network supporting both continuous and discrete actions.

    Network outputs interview costs for ALL workers, then training code
    extracts the cost for the assigned worker.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str,
        num_workers: int,
        action_low: Optional[np.ndarray] = None,
        action_high: Optional[np.ndarray] = None,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.action_type = action_type
        self.num_workers = num_workers
        self.action_dim = action_dim

        if self.action_type == "continuous":
            if action_low is None or action_high is None:
                raise ValueError("Continuous actions require action bounds.")
            self.register_buffer("_action_low", torch.tensor(action_low, dtype=torch.float32))
            self.register_buffer("_action_high", torch.tensor(action_high, dtype=torch.float32))

        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU(),
        )

        # Actor outputs interview costs for ALL workers
        if self.action_type == "continuous":
            # Output mean cost for each worker
            self.actor_mean = layer_init(nn.Linear(hidden_dim, num_workers), std=0.01)
            self.log_std = nn.Parameter(torch.zeros(num_workers))
        else:
            # For discrete mode: output continuous costs, then discretize
            # This is simpler than separate discrete distributions per worker
            self.actor_mean = layer_init(nn.Linear(hidden_dim, num_workers), std=0.01)
            self.log_std = nn.Parameter(torch.zeros(num_workers))

        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def _squash(self, raw_action: torch.Tensor) -> torch.Tensor:
        return torch.tanh(raw_action)

    def _scale(self, squashed_action: torch.Tensor) -> torch.Tensor:
        action_range = self._action_high - self._action_low
        return self._action_low + 0.5 * (squashed_action + 1.0) * action_range

    def act(
        self,
        obs_tensor: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            action: Interview costs for ALL workers, shape (batch, num_workers)
            raw_action: Unsquashed costs for ALL workers
            log_prob: Joint log probability over all workers
            entropy: Total entropy over all workers
            value: State value
        """
        features = self.shared(obs_tensor)
        value = self.critic(features).squeeze(-1)

        # Both continuous and discrete use same approach: output costs for all workers
        mean = self.actor_mean(features)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        raw_action = mean if deterministic else dist.rsample()
        squashed = self._squash(raw_action)
        action = self._scale(squashed)

        # Joint log prob: sum over all workers
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        correction = torch.log(1 - squashed.pow(2) + 1e-6).sum(dim=-1)
        log_prob = log_prob - correction
        entropy = dist.entropy().sum(dim=-1)

        return action, raw_action, log_prob, entropy, value

    def get_action(self, obs_dict, deterministic: bool = False):
        """
        Utility used by evaluation helpers.
        Returns interview costs for ALL workers.
        """
        if isinstance(obs_dict, dict):
            obs = obs_dict['observation']
        else:
            obs = obs_dict
        device = next(self.parameters()).device
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            action, _, _, _, _ = self.act(obs_tensor, deterministic=deterministic)
        # Return full cost vector for all workers
        return action.squeeze(0).cpu().numpy()

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions during training.

        Args:
            obs: Observations (batch, obs_dim)
            actions: Raw actions for ALL workers (batch, num_workers)

        Returns:
            log_probs: Joint log probability over all workers
            value: State values
            entropy: Total entropy over all workers
        """
        features = self.shared(obs)
        value = self.critic(features).squeeze(-1)

        # Treat as continuous actions for all workers
        mean = self.actor_mean(features)
        log_std = self.log_std.expand_as(mean)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        # Joint log prob over all workers
        log_probs = dist.log_prob(actions).sum(dim=-1)
        squashed = torch.tanh(actions)
        log_probs = log_probs - torch.log(1 - squashed.pow(2) + 1e-6).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return log_probs, value, entropy


# Note: compute_assigned_worker function removed - no longer needed
# The environment now accepts full action vectors (one cost per worker)


class RolloutBuffer:
    """Buffer for storing experience during rollout."""

    def __init__(self):
        self.observations = []
        self.actions = []
        self.raw_actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, obs, action, raw_action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.raw_actions.append(raw_action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.raw_actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def get(self):
        return {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'raw_actions': np.array(self.raw_actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'dones': np.array(self.dones)
        }


class PPOAgent:
    """
    Proximal Policy Optimization agent supporting both action types.

    Network outputs interview costs for all workers, then extracts the cost
    for the assigned worker as the environment action.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str,
        num_workers: int,
        action_low: Optional[np.ndarray],
        action_high: Optional[np.ndarray],
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu'
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.action_type = action_type
        self.num_workers = num_workers

        self.network = ActorCritic(
            obs_dim=obs_dim,
            action_dim=action_dim,
            action_type=action_type,
            num_workers=num_workers,
            action_low=action_low,
            action_high=action_high,
        ).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        self.buffer = RolloutBuffer()

    def get_action(
        self,
        obs_dict,
        deterministic: bool = False
    ):
        """
        Get action for environment.

        Network outputs interview costs for ALL workers, and we pass the
        full vector to the environment.

        Args:
            obs_dict: Observation dictionary
            deterministic: Whether to use deterministic policy

        Returns:
            env_action: Full action vector (num_workers,) of interview costs
            stored_raw: Raw action vector (num_workers,) for training
            value: State value
            log_prob: Joint log probability over all workers
        """
        if isinstance(obs_dict, dict):
            obs = obs_dict['observation']
        else:
            obs = obs_dict

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # action: (1, num_workers) costs for all workers
            action, raw_action, log_prob, _, value = self.network.act(
                obs_tensor,
                deterministic=deterministic,
            )

        # Extract full cost vectors
        all_costs = action.squeeze(0).cpu().numpy()  # (num_workers,)
        stored_raw = raw_action.squeeze(0).cpu().numpy()  # (num_workers,)

        # Return full action vector to environment
        return all_costs, stored_raw, value.item(), log_prob.item()

    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]

            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, next_obs_dict, n_epochs=4, batch_size=64):
        """
        Update policy using PPO.

        Returns dictionary with training metrics including explained variance.
        """
        # Ensure network is in training mode
        self.network.train()

        # Get rollout data
        data = self.buffer.get()

        if isinstance(next_obs_dict, dict):
            next_obs = next_obs_dict['observation']
        else:
            next_obs = next_obs_dict

        # Compute next value
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            _, _, _, _, next_value = self.network.act(next_obs_tensor, deterministic=True)
            next_value = next_value.item()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            data['rewards'], data['values'], data['dones'], next_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(data['observations']).to(self.device)
        if self.action_type == "continuous":
            action_tensor = torch.FloatTensor(data['raw_actions']).to(self.device)
        else:
            action_tensor = torch.LongTensor(data['actions']).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(data['log_probs']).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        old_values_tensor = torch.FloatTensor(data['values']).to(self.device)

        # PPO update
        n_samples = len(data['observations'])

        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'clip_fraction': [],
            'approx_kl': []
        }

        for epoch in range(n_epochs):
            # Generate random indices for mini-batches
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]

                # Get batch
                obs_batch = obs_tensor[batch_indices]
                actions_batch = action_tensor[batch_indices]
                old_log_probs_batch = old_log_probs_tensor[batch_indices]
                advantages_batch = advantages_tensor[batch_indices]
                returns_batch = returns_tensor[batch_indices]
                old_values_batch = old_values_tensor[batch_indices]

                log_probs, values, entropy = self.network.evaluate_actions(
                    obs_batch, actions_batch
                )

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped as per CleanRL)
                value_pred_clipped = old_values_batch + torch.clamp(
                    values - old_values_batch,
                    -self.clip_epsilon,
                    self.clip_epsilon
                )
                value_losses = (values - returns_batch) ** 2
                value_losses_clipped = (value_pred_clipped - returns_batch) ** 2
                value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track stats
                with torch.no_grad():
                    clip_fraction = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean()
                    # Approximate KL divergence
                    log_ratio = log_probs - old_log_probs_batch
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(-entropy_loss.item())
                stats['total_loss'].append(loss.item())
                stats['clip_fraction'].append(clip_fraction.item())
                stats['approx_kl'].append(approx_kl.item())

        # Compute explained variance
        y_pred = data['values']
        y_true = returns
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Clear buffer
        self.buffer.clear()

        # Return averaged stats
        result = {k: np.mean(v) for k, v in stats.items()}
        result['explained_variance'] = explained_var
        return result


class IPPOTrainer:
    """
    Independent PPO Trainer for multi-agent environment.

    - TensorBoard logging
    - Learning rate annealing
    - Explained variance tracking
    """

    def __init__(
        self,
        env: JobMarketEnv,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        device: str = 'cpu',
        seed: int = 42,
        anneal_lr: bool = False,
        run_name: Optional[str] = None,
        log_step_data: bool = True,
        step_log_path: Optional[str] = None
    ):
        self.env = env
        self.device = device
        self.initial_lr = lr
        self.anneal_lr = anneal_lr
        self.seed = seed

        # Set seeds for reproducibility
        set_global_seed(seed)

        # Create PPO agent for each company
        self.agents: Dict[str, PPOAgent] = {}
        self._default_observations: Dict[str, np.ndarray] = {}

        for agent_name in env.possible_agents:
            # Get observation dimension from the Box space inside Dict
            obs_space = env.observation_space(agent_name)
            obs_dim = obs_space.spaces['observation'].shape[0]
            action_space = env.action_space(agent_name)
            if isinstance(action_space, GymBox):
                action_type = "continuous"
                action_dim = int(np.prod(action_space.shape))
                action_low = action_space.low
                action_high = action_space.high
            elif isinstance(action_space, GymDiscrete):
                action_type = "discrete"
                action_dim = action_space.n
                action_low = None
                action_high = None
            else:
                raise NotImplementedError("Unsupported action space type")

            self.agents[agent_name] = PPOAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                action_type=action_type,
                num_workers=env.num_workers,
                action_low=action_low,
                action_high=action_high,
                lr=lr,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_epsilon=clip_epsilon,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                device=device
            )
            self._default_observations[agent_name] = np.zeros(obs_dim, dtype=np.float32)

        # Setup run directory
        if run_name is None:
            run_name = f"ippo_{int(time.time())}"
        self.run_name = run_name
        self.run_dir = f"runs/{run_name}"
        os.makedirs(self.run_dir, exist_ok=True)

        # TensorBoard writer (logs go directly in run_dir)
        self.writer = SummaryWriter(self.run_dir)

        # Save training config
        import json
        self.config = {
            'env': {
                'num_companies': env.num_companies,
                'num_workers': env.num_workers,
                'max_workers_per_company': env.max_workers_per_company,
                'ability_dim': env.ability_dim,
                'g0': env.g0,
                'g1': env.g1,
                'firing_cost_multiplier': env.firing_cost_multiplier,
                'max_interview_cost': env.max_interview_cost,
                'action_mode': env.action_mode,
                'max_timesteps': env.max_timesteps,
            },
            'training': {
                'lr': lr,
                'gamma': gamma,
                'gae_lambda': gae_lambda,
                'clip_epsilon': clip_epsilon,
                'value_coef': value_coef,
                'entropy_coef': entropy_coef,
                'max_grad_norm': max_grad_norm,
                'device': device,
                'seed': seed,
                'anneal_lr': anneal_lr
            }
        }
        with open(f"{self.run_dir}/config.json", 'w') as f:
            json.dump(self.config, f, indent=2)

        # Per-step logging
        self.log_step_data = log_step_data
        self.step_log_path = step_log_path or os.path.join("runs", run_name, "step_log.csv")
        self._step_log_buffer: List[Dict[str, float]] = []
        self._step_log_fieldnames = ['global_step', 'agent', 'action', 'reward']
        self._step_log_initialized = False
        if self.log_step_data:
            os.makedirs(os.path.dirname(self.step_log_path), exist_ok=True)
            self._init_step_log_file()

        # Log hyperparameters
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n" + "\n".join([
                f"|lr|{lr}|",
                f"|gamma|{gamma}|",
                f"|gae_lambda|{gae_lambda}|",
                f"|clip_epsilon|{clip_epsilon}|",
                f"|value_coef|{value_coef}|",
                f"|entropy_coef|{entropy_coef}|",
                f"|max_grad_norm|{max_grad_norm}|",
                f"|device|{device}|",
                f"|seed|{seed}|",
                f"|anneal_lr|{anneal_lr}|"
            ])
        )

        # Tracking
        self.episode_rewards = {agent: [] for agent in env.possible_agents}
        self.episode_lengths = []
        self.global_step = 0

    def _init_step_log_file(self):
        if not self.log_step_data or self._step_log_initialized:
            return
        file_exists = os.path.exists(self.step_log_path)
        with open(self.step_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._step_log_fieldnames)
            if not file_exists:
                writer.writeheader()
        self._step_log_initialized = True

    def _flush_step_log_buffer(self):
        if not self.log_step_data or not self._step_log_buffer:
            return
        with open(self.step_log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._step_log_fieldnames)
            writer.writerows(self._step_log_buffer)
        self._step_log_buffer.clear()

    def collect_rollout(self, n_steps: int):
        """Collect n_steps of experience for each agent."""
        observations, _ = self.env.reset()

        # Track episode returns
        current_episode_rewards = {agent: 0.0 for agent in self.env.possible_agents}
        episode_length = 0

        # Track interview cost statistics
        interview_costs = []
        all_worker_costs = {agent: [] for agent in self.env.possible_agents}  # Track costs for ALL workers

        # Track environment actions (hires, fires)
        episode_hires = {agent: 0 for agent in self.env.possible_agents}
        episode_fires = {agent: 0 for agent in self.env.possible_agents}
        total_hires = 0
        total_fires = 0

        for step in range(n_steps):
            # Get actions from all agents
            actions = {}
            values = {}
            log_probs = {}
            raw_actions = {}

            for agent_name in self.env.agents:
                agent = self.agents[agent_name]

                # Get action: full vector of interview costs for all workers
                env_action, raw_action, value, log_prob = agent.get_action(
                    obs_dict=observations[agent_name],
                    deterministic=False
                )

                # env_action is (num_workers,) vector of costs
                # raw_action is (num_workers,) vector of raw (unsquashed) costs
                actions[agent_name] = env_action
                values[agent_name] = value
                log_probs[agent_name] = log_prob
                raw_actions[agent_name] = raw_action  # Store full vector

                # Track interview costs (costs > 0 indicate interviews)
                interviewed_workers = env_action > 0.0
                num_interviewed = int(np.sum(interviewed_workers))
                if num_interviewed > 0:
                    avg_cost = float(np.mean(env_action[interviewed_workers]))
                    interview_costs.append(avg_cost)

                # Track full action distribution
                all_worker_costs[agent_name].append(env_action)  # (num_workers,) per step

            # Step environment
            next_observations, rewards, terminations, truncations, infos = self.env.step(actions)

            # Track hires and fires from this step
            for agent_name in self.env.agents:
                if agent_name in infos:
                    num_hires = len(infos[agent_name].get('hirings', []))
                    num_fires = len(infos[agent_name].get('firings', []))
                    episode_hires[agent_name] += num_hires
                    episode_fires[agent_name] += num_fires
                    total_hires += num_hires
                    total_fires += num_fires

            # Accumulate episode rewards
            for agent_name in self.env.possible_agents:
                if agent_name in rewards:
                    current_episode_rewards[agent_name] += rewards[agent_name]
            episode_length += 1

            # Store experience in buffers
            for agent_name in self.env.agents:
                done = terminations[agent_name] or truncations[agent_name]

                # Extract observation and mask
                obs = observations[agent_name]['observation']

                self.agents[agent_name].buffer.add(
                    obs=obs,
                    action=actions[agent_name],
                    raw_action=raw_actions[agent_name],
                    reward=rewards[agent_name],
                    value=values[agent_name],
                    log_prob=log_probs[agent_name],
                    done=done
                )
                self.writer.add_scalar(f"{agent_name}/reward", rewards[agent_name], self.global_step)

            if self.log_step_data:
                for agent_name in self.env.agents:
                    reward = rewards.get(agent_name, 0.0)
                    self._step_log_buffer.append({
                        'global_step': self.global_step,
                        'agent': agent_name,
                        'action': actions[agent_name],
                        'reward': reward
                    })

            observations = next_observations
            self.global_step += 1

            # Check if episode ended
            if all(terminations.values()) or all(truncations.values()):
                # Log episodic returns per agent
                total_episodic_return = 0.0
                for agent_name in self.env.possible_agents:
                    agent_return = current_episode_rewards[agent_name]
                    self.writer.add_scalar(
                        f"charts/{agent_name}_episodic_return",
                        agent_return,
                        self.global_step
                    )
                    total_episodic_return += agent_return

                # Log aggregate episodic return
                self.writer.add_scalar("charts/total_episodic_return", total_episodic_return, self.global_step)
                self.writer.add_scalar("charts/avg_episodic_return", total_episodic_return / len(self.env.possible_agents), self.global_step)
                self.writer.add_scalar("charts/episode_length", episode_length, self.global_step)

                # Log hiring/firing actions
                self.writer.add_scalar("actions/total_hires", total_hires, self.global_step)
                self.writer.add_scalar("actions/total_fires", total_fires, self.global_step)
                self.writer.add_scalar("actions/net_employment_change", total_hires - total_fires, self.global_step)

                # Per-agent hiring/firing
                for agent_name in self.env.possible_agents:
                    self.writer.add_scalar(
                        f"actions/{agent_name}/hires",
                        episode_hires[agent_name],
                        self.global_step
                    )
                    self.writer.add_scalar(
                        f"actions/{agent_name}/fires",
                        episode_fires[agent_name],
                        self.global_step
                    )
                    # Hiring rate (hires per episode step)
                    if episode_length > 0:
                        self.writer.add_scalar(
                            f"actions/{agent_name}/hiring_rate",
                            episode_hires[agent_name] / episode_length,
                            self.global_step
                        )
                        self.writer.add_scalar(
                            f"actions/{agent_name}/firing_rate",
                            episode_fires[agent_name] / episode_length,
                            self.global_step
                        )

                # Log interview cost statistics
                if interview_costs:
                    self.writer.add_scalar("interview/avg_cost", np.mean(interview_costs), self.global_step)
                    self.writer.add_scalar("interview/max_cost", np.max(interview_costs), self.global_step)
                    self.writer.add_scalar("interview/min_cost", np.min(interview_costs), self.global_step)
                    self.writer.add_scalar("interview/std_cost", np.std(interview_costs), self.global_step)

                    # Histogram of interview costs
                    self.writer.add_histogram("interview/cost_distribution", np.array(interview_costs), self.global_step)

                # Log detailed action distributions per agent
                for agent_name in self.env.possible_agents:
                    if all_worker_costs[agent_name]:
                        # Stack all cost vectors: (num_steps, num_workers)
                        costs_array = np.array(all_worker_costs[agent_name])

                        # Overall statistics across all workers
                        self.writer.add_scalar(
                            f"actions/{agent_name}/mean_cost_all_workers",
                            costs_array.mean(),
                            self.global_step
                        )
                        self.writer.add_scalar(
                            f"actions/{agent_name}/std_cost_all_workers",
                            costs_array.std(),
                            self.global_step
                        )

                        # Histogram of ALL predicted costs (for all workers)
                        self.writer.add_histogram(
                            f"actions/{agent_name}/all_worker_costs",
                            costs_array.flatten(),
                            self.global_step
                        )

                        # Per-worker cost statistics (average across episode)
                        per_worker_mean = costs_array.mean(axis=0)  # (num_workers,)
                        for worker_id in range(self.env.num_workers):
                            self.writer.add_scalar(
                                f"actions/{agent_name}/worker_{worker_id}_mean_cost",
                                per_worker_mean[worker_id],
                                self.global_step
                            )

                # Reset environment and episode tracking
                observations, _ = self.env.reset()
                current_episode_rewards = {agent: 0.0 for agent in self.env.possible_agents}
                episode_length = 0
                interview_costs = []
                all_worker_costs = {agent: [] for agent in self.env.possible_agents}
                episode_hires = {agent: 0 for agent in self.env.possible_agents}
                episode_fires = {agent: 0 for agent in self.env.possible_agents}
                total_hires = 0
                total_fires = 0

        self._flush_step_log_buffer()
        return observations

    def train(
        self,
        total_timesteps: int,
        n_steps: int = 2048,
        n_epochs: int = 4,
        batch_size: int = 64,
        log_interval: int = 10,
        save_interval: int = 100,
        eval_interval: int = 0,
        save_path: Optional[str] = None,
    ):
        """
        Train all agents using Independent PPO.

        Features:
        - TensorBoard logging
        - Learning rate annealing
        - Explained variance tracking
        - Action masking
        - Periodic evaluation with detailed metrics

        Args:
            eval_interval: Run evaluation every N updates (0 = no periodic eval)
        """
        # Create checkpoint directory
        checkpoint_dir = save_path or f"{self.run_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)

        n_updates = total_timesteps // n_steps

        print(f"\nTraining IPPO: {self.run_name}")
        print(f"Timesteps: {total_timesteps:,} | Updates: {n_updates} | Agents: {len(self.env.possible_agents)}")
        if eval_interval > 0:
            print(f"Evaluation interval: every {eval_interval} updates")

        start_time = time.time()

        for update in range(1, n_updates + 1):
            # Learning rate annealing
            if self.anneal_lr:
                frac = 1.0 - (update - 1.0) / n_updates
                lr_now = frac * self.initial_lr
                for agent in self.agents.values():
                    agent.optimizer.param_groups[0]['lr'] = lr_now

            # Collect rollout
            next_observations = self.collect_rollout(n_steps)

            # Update all agents
            update_stats = {}
            for agent_name in self.env.possible_agents:
                fallback_obs = {'observation': self._default_observations[agent_name]}
                stats = self.agents[agent_name].update(
                    next_obs_dict=next_observations.get(agent_name, fallback_obs),
                    n_epochs=n_epochs,
                    batch_size=batch_size
                )
                update_stats[agent_name] = stats

            current_timesteps = update * n_steps

            # Log to TensorBoard
            for agent_name, stats in update_stats.items():
                self.writer.add_scalar(f"{agent_name}/policy_loss", stats['policy_loss'], current_timesteps)
                self.writer.add_scalar(f"{agent_name}/value_loss", stats['value_loss'], current_timesteps)
                self.writer.add_scalar(f"{agent_name}/entropy", stats['entropy'], current_timesteps)
                self.writer.add_scalar(f"{agent_name}/clip_fraction", stats['clip_fraction'], current_timesteps)
                self.writer.add_scalar(f"{agent_name}/approx_kl", stats['approx_kl'], current_timesteps)
                self.writer.add_scalar(f"{agent_name}/explained_variance", stats['explained_variance'], current_timesteps)

            # Log aggregate statistics across all agents
            avg_entropy = np.mean([s['entropy'] for s in update_stats.values()])
            avg_explained_var = np.mean([s['explained_variance'] for s in update_stats.values()])
            self.writer.add_scalar("charts/avg_entropy", avg_entropy, current_timesteps)
            self.writer.add_scalar("charts/avg_explained_variance", avg_explained_var, current_timesteps)

            # Log learning rate
            if self.anneal_lr:
                self.writer.add_scalar("charts/learning_rate", lr_now, current_timesteps)

            # Log SPS (steps per second)
            elapsed = time.time() - start_time
            sps = current_timesteps / elapsed
            self.writer.add_scalar("charts/SPS", sps, current_timesteps)

            # Console logging
            if update % log_interval == 0:
                avg_policy_loss = np.mean([s['policy_loss'] for s in update_stats.values()])
                avg_value_loss = np.mean([s['value_loss'] for s in update_stats.values()])
                avg_entropy = np.mean([s['entropy'] for s in update_stats.values()])
                lr_str = f" | LR: {lr_now:.6f}" if self.anneal_lr else ""
                print(f"[{update}/{n_updates}] Step {current_timesteps:,} | SPS: {sps:.0f}{lr_str} | "
                      f"PL: {avg_policy_loss:.3f} VL: {avg_value_loss:.3f} Ent: {avg_entropy:.3f}")

            # Save checkpoints
            if update % save_interval == 0:
                for agent_name in self.env.possible_agents:
                    model_path = os.path.join(checkpoint_dir, f"{agent_name}_update_{update}.pt")
                    torch.save(self.agents[agent_name].network.state_dict(), model_path)
                print(f"  ✓ Checkpoint saved at update {update}")

            # Periodic evaluation
            if eval_interval > 0 and update % eval_interval == 0:
                print(f"\n--- Running evaluation at update {update} ---")
                self.evaluate(n_episodes=5, deterministic=True)
                print(f"--- Evaluation complete, resuming training ---\n")

        # Save final models
        for agent_name in self.env.possible_agents:
            model_path = os.path.join(checkpoint_dir, f"{agent_name}_final.pt")
            torch.save(self.agents[agent_name].network.state_dict(), model_path)

        elapsed_time = time.time() - start_time
        print(f"\n✓ Training complete! Time: {elapsed_time:.1f}s")
        print(f"  Models: {self.run_dir}/checkpoints/")
        print(f"  Logs: {self.run_dir}/")

        self.writer.close()

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True):
        """Evaluate learned policies with time-series TensorBoard logging."""
        # Set all networks to eval mode
        for agent in self.agents.values():
            agent.network.eval()

        print(f"\nEvaluating {n_episodes} episodes...")

        episode_rewards = {agent: [] for agent in self.env.possible_agents}
        episode_lengths = []

        # Track time-series metrics per episode (to average across episodes later)
        # Structure: {agent: {metric: [episode1_timeseries, episode2_timeseries, ...]}}
        time_series_metrics = {
            agent: {
                'interview_costs': [],      # List of lists: each inner = per-step costs in one episode
                'hires_per_step': [],
                'fires_per_step': [],
                'profits_per_step': [],
                'wages_per_step': [],
                'firing_costs_per_step': [],
                'cumulative_profit': [],
                'workforce_size': [],
            }
            for agent in self.env.possible_agents
        }

        for episode in range(n_episodes):
            observations, _ = self.env.reset()
            ep_rewards = {agent: 0.0 for agent in self.env.possible_agents}
            ep_length = 0
            timestep = 0

            # Track metrics for THIS episode as time series
            ep_metrics = {
                agent: {
                    'interview_costs': [],
                    'hires_per_step': [],
                    'fires_per_step': [],
                    'profits_per_step': [],
                    'wages_per_step': [],
                    'firing_costs_per_step': [],
                    'cumulative_profit': [],
                    'workforce_size': [],
                }
                for agent in self.env.possible_agents
            }
            cumulative_profit = {agent: 0.0 for agent in self.env.possible_agents}

            # Track employed_by to detect hires/fires
            prev_employed_by = self.env.employed_by.copy()

            done = False
            while not done:
                # Get actions (deterministic for evaluation)
                actions = {}
                step_interview_costs = {}

                for agent_name in self.env.agents:
                    # Get full cost vector from network (returns all worker costs)
                    all_costs = self.agents[agent_name].network.get_action(
                        observations[agent_name],
                        deterministic=deterministic
                    )
                    # Pass full action vector to environment
                    actions[agent_name] = all_costs

                    # Calculate interview spending (sum of costs > 0)
                    interviewed_mask = all_costs > 0.0
                    total_interview_cost = float(np.sum(all_costs[interviewed_mask]))
                    step_interview_costs[agent_name] = total_interview_cost

                # Step
                observations, rewards, terminations, truncations, infos = self.env.step(actions)

                # Detect hires and fires by comparing employed_by before/after step
                new_employed_by = self.env.employed_by.copy()
                step_hires = {agent: 0 for agent in self.env.possible_agents}
                step_fires = {agent: 0 for agent in self.env.possible_agents}

                for worker_id in range(self.env.num_workers):
                    prev_firm = prev_employed_by[worker_id]
                    new_firm = new_employed_by[worker_id]

                    if prev_firm < 0 and new_firm >= 0:
                        # Worker was hired
                        agent = f"company_{new_firm}"
                        step_hires[agent] += 1
                    elif prev_firm >= 0 and new_firm < 0:
                        # Worker was fired
                        agent = f"company_{prev_firm}"
                        step_fires[agent] += 1

                prev_employed_by = new_employed_by

                # Track detailed metrics from this step
                for agent_name in self.env.agents:
                    ep_rewards[agent_name] += rewards[agent_name]

                    # Get info for this agent
                    info = infos.get(agent_name, {})

                    # Interview costs
                    ep_metrics[agent_name]['interview_costs'].append(step_interview_costs.get(agent_name, 0.0))

                    # Hires and fires (from our tracking)
                    ep_metrics[agent_name]['hires_per_step'].append(step_hires[agent_name])
                    ep_metrics[agent_name]['fires_per_step'].append(step_fires[agent_name])

                    # Financial metrics
                    step_profit = info.get('last_step_profit', 0.0)
                    step_wage = info.get('last_step_wage', 0.0)
                    step_firing_cost = info.get('last_step_firing_cost', 0.0)

                    ep_metrics[agent_name]['profits_per_step'].append(step_profit)
                    ep_metrics[agent_name]['wages_per_step'].append(step_wage)
                    ep_metrics[agent_name]['firing_costs_per_step'].append(step_firing_cost)

                    # Cumulative profit
                    cumulative_profit[agent_name] += step_profit
                    ep_metrics[agent_name]['cumulative_profit'].append(cumulative_profit[agent_name])

                    # Workforce size
                    company_idx = int(agent_name.split('_')[1])
                    workforce_size = int(np.sum(self.env.employed_by == company_idx))
                    ep_metrics[agent_name]['workforce_size'].append(workforce_size)

                ep_length += 1
                timestep += 1
                done = all(terminations.values()) or all(truncations.values())

            # Store this episode's time series
            for agent_name in self.env.possible_agents:
                for metric_name, values in ep_metrics[agent_name].items():
                    time_series_metrics[agent_name][metric_name].append(values)

            # Store episode stats
            for agent_name in self.env.possible_agents:
                episode_rewards[agent_name].append(ep_rewards.get(agent_name, 0.0))
            episode_lengths.append(ep_length)

        # Compute mean time series across episodes and log to TensorBoard
        eval_tag = f"eval"

        for agent_name in self.env.possible_agents:
            agent_metrics = time_series_metrics[agent_name]

            # For each metric, compute mean across episodes at each timestep
            for metric_name, episode_list in agent_metrics.items():
                if not episode_list:
                    continue

                # episode_list is a list of lists: [episode1_timeseries, episode2_timeseries, ...]
                # Find the minimum length (in case episodes ended at different timesteps)
                min_length = min(len(ep_data) for ep_data in episode_list)

                if min_length == 0:
                    continue

                # Compute mean across episodes for each timestep
                mean_timeseries = []
                for t in range(min_length):
                    values_at_t = [ep_data[t] for ep_data in episode_list]
                    mean_timeseries.append(np.mean(values_at_t))

                # Log each timestep to TensorBoard
                # X-axis = episode timestep (0-99), tag includes training step for differentiation
                for t, value in enumerate(mean_timeseries):
                    # Tag format: eval_at_step_X/agent_name/metric_name
                    # X-axis = episode timestep
                    self.writer.add_scalar(
                        f"{eval_tag}_at_step_{self.global_step}/{agent_name}/{metric_name}",
                        value,
                        t  # Episode timestep
                    )

        # Export time-series data to CSV for offline analysis
        csv_path = f"{self.run_dir}/eval_step_{self.global_step}_timeseries.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'episode', 'timestep', 'agent',
                'interview_cost', 'hires', 'fires',
                'profit', 'wage', 'firing_cost',
                'cumulative_profit', 'workforce_size'
            ])
            writer.writeheader()

            # Write data from each episode
            for episode_idx in range(n_episodes):
                for agent_name in self.env.possible_agents:
                    metrics = time_series_metrics[agent_name]

                    # Get this episode's data
                    if episode_idx < len(metrics['interview_costs']):
                        ep_len = len(metrics['interview_costs'][episode_idx])

                        for t in range(ep_len):
                            row = {
                                'episode': episode_idx,
                                'timestep': t,
                                'agent': agent_name,
                                'interview_cost': metrics['interview_costs'][episode_idx][t],
                                'hires': metrics['hires_per_step'][episode_idx][t],
                                'fires': metrics['fires_per_step'][episode_idx][t],
                                'profit': metrics['profits_per_step'][episode_idx][t],
                                'wage': metrics['wages_per_step'][episode_idx][t],
                                'firing_cost': metrics['firing_costs_per_step'][episode_idx][t],
                                'cumulative_profit': metrics['cumulative_profit'][episode_idx][t],
                                'workforce_size': metrics['workforce_size'][episode_idx][t],
                            }
                            writer.writerow(row)

        print(f"  Saved time-series data to: {csv_path}")

        # Print summary
        print(f"\nEvaluation Results:")
        for agent_name in self.env.possible_agents:
            mean_reward = np.mean(episode_rewards[agent_name])
            std_reward = np.std(episode_rewards[agent_name])
            print(f"  {agent_name}: {mean_reward:.2f} ± {std_reward:.2f}")

            # Print additional metrics (flatten time series for summary stats)
            metrics = time_series_metrics[agent_name]
            if metrics['interview_costs']:
                all_costs = [val for ep in metrics['interview_costs'] for val in ep]
                print(f"    Avg interview cost/step: {np.mean(all_costs):.2f}")
            if metrics['hires_per_step']:
                all_hires = [val for ep in metrics['hires_per_step'] for val in ep]
                print(f"    Avg hires/step: {np.mean(all_hires):.2f}")
            if metrics['fires_per_step']:
                all_fires = [val for ep in metrics['fires_per_step'] for val in ep]
                print(f"    Avg fires/step: {np.mean(all_fires):.2f}")
            if metrics['workforce_size']:
                all_workforce = [val for ep in metrics['workforce_size'] for val in ep]
                print(f"    Avg workforce: {np.mean(all_workforce):.1f}")
        print(f"  Episode Length: {np.mean(episode_lengths):.1f}")

        # Restore networks to train mode
        for agent in self.agents.values():
            agent.network.train()


def main():
    """Main training script."""
    seed = 42

    # Ensure deterministic behavior before anything else
    set_global_seed(seed)

    # --- Empirical initialization toggles ---
    use_empirical_init = True   # set False to revert to old synthetic defaults
    auto_calibrate_noise = True # set False to keep default noise params
    employees_per_agent = 20_000  # scaling: real employees -> env capacity

    # Defaults (synthetic)
    num_companies = 5
    num_workers = 50
    firm_capacities = None
    firm_types = None
    firm_type_premia = None
    profit_noise_var = 0.1
    delta_interview0_sq = 0.4
    wage_scale = 1.0

    if use_empirical_init and sde_data is not None:
        try:
            firms_df = sde_data.initialize_firms(
                num_firms=num_companies,
                type_config=sde_data.firms_type_config,
                random_state=seed,
                sample_strategy="empirical",
            )
            firm_capacities = sde_data.to_env_capacities(
                firms_df, employees_per_agent=employees_per_agent
            )
            firm_types = firms_df["firm_type"].tolist()
            firm_type_premia = sde_data.estimate_size_wage_premia()
            num_workers = int(sum(firm_capacities))
            num_companies = len(firm_capacities)
            max_workers_per_company = max(firm_capacities)
        except Exception as e:
            print(f"[warn] empirical init failed, using synthetic defaults: {e}")
            firm_capacities = None
            firm_types = None
            firm_type_premia = None
            max_workers_per_company = 10
    else:
        max_workers_per_company = 10

    if auto_calibrate_noise and sde_data is not None:
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
                print(f"[info] calibrated noise: delta_interview0_sq={delta_interview0_sq}, delta_profit_sq={profit_noise_var}")
            # Set a simple wage scale to bring model wages closer to empirical magnitude
            wage_mean = float(wage_df["salary"].mean())
            wage_scale = wage_mean if wage_mean > 0 else wage_scale
        except Exception as e:
            print(f"[warn] noise calibration failed, using defaults: {e}")
            delta_interview0_sq = 0.4
            profit_noise_var = 0.05
    else:
        max_workers_per_company = max_workers_per_company if firm_capacities else 10

    # Create multi-agent environment with multiple companies competing
    env = JobMarketEnv(
        num_companies=num_companies,
        num_workers=num_workers,
        max_workers_per_company=max_workers_per_company,
        firm_capacities=firm_capacities,
        firm_types=firm_types,
        firm_type_premia=firm_type_premia,
        max_timesteps=100,  # Shorter episodes for faster learning
        profit_noise_var=profit_noise_var,
        wage_scale=locals().get("wage_scale", 1.0),
        seed=seed,
    )
    # Align interview noise baseline with calibrated value
    env.screening.delta0_sq = float(delta_interview0_sq)

    # Create trainer with unique run name
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"hirerl_ippo_{timestamp}"

    trainer = IPPOTrainer(
        env=env,
        lr=3e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device='cpu',
        seed=seed,
        anneal_lr=False,
        run_name=run_name
    )

    # Train
    trainer.train(
        total_timesteps=100_000,  # More timesteps for meaningful learning
        n_steps=1024,  # Standard PPO rollout length
        n_epochs=10,  # PPO update epochs
        batch_size=64,  # Batch size for updates
        log_interval=10,
        save_interval=50000,
        eval_interval=10  # Evaluate every 10 updates to track learning progress
    )

    # Final evaluation - single episode to see clean temporal behavior
    trainer.evaluate(n_episodes=10, deterministic=True)


if __name__ == '__main__':
    main()
