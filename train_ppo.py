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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pettingzoo'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Tuple, Optional
import time
import random
from collections import defaultdict

from hirerl import JobMarketEnv
from utils import EpisodeLogger, PerformanceMetrics


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
    Actor-Critic network for PPO with action masking support.

    Architecture:
    - Shared feature extractor
    - Actor head (policy): outputs action logits
    - Critic head (value): outputs state value estimate

    Features:
    - Orthogonal initialization
    - Action masking in forward pass
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature extractor with orthogonal init
        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.ReLU()
        )

        # Actor head (policy) - small std for better initial exploration
        self.actor = layer_init(nn.Linear(hidden_dim, action_dim), std=0.01)

        # Critic head (value function) - std=1 as per CleanRL
        self.critic = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, x, action_mask=None):
        """
        Forward pass with optional action masking.

        Args:
            x: Observation tensor
            action_mask: Optional binary mask (1=valid, 0=invalid)

        Returns:
            action_logits: Logits for action distribution (masked if mask provided)
            value: State value estimate
        """
        features = self.shared(x)
        action_logits = self.actor(features)

        # Apply action masking if provided
        if action_mask is not None:
            # Set logits of invalid actions to -inf
            action_logits = torch.where(
                action_mask.bool(),
                action_logits,
                torch.tensor(-1e8, dtype=action_logits.dtype, device=action_logits.device)
            )

        value = self.critic(features)
        return action_logits, value

    def get_action(self, obs_dict, deterministic=False):
        """
        Sample action from policy with action masking.

        Args:
            obs_dict: Dictionary with 'observation' and 'action_mask' keys
            deterministic: If True, select argmax instead of sampling

        Returns:
            action: Selected action
        """
        with torch.no_grad():
            # Extract observation and mask
            if isinstance(obs_dict, dict):
                obs = obs_dict['observation']
                action_mask = obs_dict['action_mask']
            else:
                # Fallback for non-dict observations (backwards compatibility)
                obs = obs_dict
                action_mask = None

            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            if action_mask is not None:
                mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0)
            else:
                mask_tensor = None

            action_logits, value = self.forward(obs_tensor, mask_tensor)

            if deterministic:
                action = action_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(action_logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()

            return action.item()

    def evaluate_actions(self, obs, actions, action_masks=None):
        """
        Evaluate log probabilities and values for given observations and actions.

        Args:
            obs: Observation tensor
            actions: Action tensor
            action_masks: Optional action mask tensor

        Returns:
            log_probs: Log probabilities of actions
            values: Value estimates
            entropy: Policy entropy
        """
        action_logits, values = self.forward(obs, action_masks)
        probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """
    Buffer for storing experience during rollout.

    Now includes action masks for proper action masking during training.
    """

    def __init__(self):
        self.observations = []
        self.action_masks = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, obs, action_mask, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.action_masks.append(action_mask)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.observations.clear()
        self.action_masks.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def get(self):
        return {
            'observations': np.array(self.observations),
            'action_masks': np.array(self.action_masks),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'dones': np.array(self.dones)
        }


class PPOAgent:
    """
    Proximal Policy Optimization agent with action masking.

    - Action masking support
    - Explained variance metric
    - Better initialization
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
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

        # Create network with orthogonal initialization
        self.network = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)

        # Buffer
        self.buffer = RolloutBuffer()

    def get_action(self, obs_dict, deterministic=False):
        """
        Get action from policy with action masking.

        Args:
            obs_dict: Dictionary with 'observation' and 'action_mask'
            deterministic: If True, use argmax instead of sampling

        Returns:
            action, value, log_prob
        """
        # Extract observation and mask
        if isinstance(obs_dict, dict):
            obs = obs_dict['observation']
            action_mask = obs_dict['action_mask']
        else:
            obs = obs_dict
            action_mask = None

        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        if action_mask is not None:
            mask_tensor = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
        else:
            mask_tensor = None

        with torch.no_grad():
            action_logits, value = self.network(obs_tensor, mask_tensor)
            probs = torch.softmax(action_logits, dim=-1)
            dist = Categorical(probs)

            if deterministic:
                action = action_logits.argmax(dim=-1)
            else:
                action = dist.sample()

            log_prob = dist.log_prob(action)

        return action.item(), value.item(), log_prob.item()

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
        Update policy using PPO with action masking.

        Returns dictionary with training metrics including explained variance.
        """
        # Get rollout data
        data = self.buffer.get()

        # Extract next observation and mask
        if isinstance(next_obs_dict, dict):
            next_obs = next_obs_dict['observation']
            next_mask = next_obs_dict['action_mask']
        else:
            next_obs = next_obs_dict
            next_mask = None

        # Compute next value
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            if next_mask is not None:
                next_mask_tensor = torch.FloatTensor(next_mask).unsqueeze(0).to(self.device)
            else:
                next_mask_tensor = None
            _, next_value = self.network(next_obs_tensor, next_mask_tensor)
            next_value = next_value.item()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            data['rewards'], data['values'], data['dones'], next_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(data['observations']).to(self.device)
        mask_tensor = torch.FloatTensor(data['action_masks']).to(self.device)
        actions_tensor = torch.LongTensor(data['actions']).to(self.device)
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
                mask_batch = mask_tensor[batch_indices]
                actions_batch = actions_tensor[batch_indices]
                old_log_probs_batch = old_log_probs_tensor[batch_indices]
                advantages_batch = advantages_tensor[batch_indices]
                returns_batch = returns_tensor[batch_indices]
                old_values_batch = old_values_tensor[batch_indices]

                # Evaluate actions with masks
                log_probs, values, entropy = self.network.evaluate_actions(
                    obs_batch, actions_batch, mask_batch
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
    - Action masking support
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
        anneal_lr: bool = True,
        run_name: Optional[str] = None
    ):
        self.env = env
        self.device = device
        self.initial_lr = lr
        self.anneal_lr = anneal_lr

        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

        # Create PPO agent for each company
        self.agents: Dict[str, PPOAgent] = {}

        for agent_name in env.possible_agents:
            # Get observation dimension from the Box space inside Dict
            obs_space = env.observation_space(agent_name)
            obs_dim = obs_space.spaces['observation'].shape[0]
            action_dim = env.action_space(agent_name).n

            self.agents[agent_name] = PPOAgent(
                obs_dim=obs_dim,
                action_dim=action_dim,
                lr=lr,
                gamma=gamma,
                gae_lambda=gae_lambda,
                clip_epsilon=clip_epsilon,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                max_grad_norm=max_grad_norm,
                device=device
            )

        # TensorBoard writer
        if run_name is None:
            run_name = f"ippo_{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.run_name = run_name

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

    def collect_rollout(self, n_steps: int):
        """Collect n_steps of experience for each agent with action masking."""
        observations, _ = self.env.reset()

        # Track episode returns
        current_episode_rewards = {agent: 0.0 for agent in self.env.possible_agents}
        episode_length = 0

        for step in range(n_steps):
            # Get actions from all agents
            actions = {}
            values = {}
            log_probs = {}

            for agent_name in self.env.agents:
                action, value, log_prob = self.agents[agent_name].get_action(
                    observations[agent_name]
                )
                actions[agent_name] = action
                values[agent_name] = value
                log_probs[agent_name] = log_prob

            # Step environment
            next_observations, rewards, terminations, truncations, infos = self.env.step(actions)

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
                mask = observations[agent_name]['action_mask']

                self.agents[agent_name].buffer.add(
                    obs=obs,
                    action_mask=mask,
                    action=actions[agent_name],
                    reward=rewards[agent_name],
                    value=values[agent_name],
                    log_prob=log_probs[agent_name],
                    done=done
                )

            observations = next_observations
            self.global_step += 1

            # Check if episode ended
            if all(terminations.values()) or all(truncations.values()):
                # Log episodic returns
                for agent_name in self.env.possible_agents:
                    self.writer.add_scalar(
                        f"charts/{agent_name}_episodic_return",
                        current_episode_rewards[agent_name],
                        self.global_step
                    )
                self.writer.add_scalar("charts/episode_length", episode_length, self.global_step)

                # Reset environment and episode tracking
                observations, _ = self.env.reset()
                current_episode_rewards = {agent: 0.0 for agent in self.env.possible_agents}
                episode_length = 0

        return observations

    def train(
        self,
        total_timesteps: int,
        n_steps: int = 2048,
        n_epochs: int = 4,
        batch_size: int = 64,
        log_interval: int = 10,
        save_interval: int = 100,
        save_path: str = 'checkpoints'
    ):
        """
        Train all agents using Independent PPO.

        Features:
        - TensorBoard logging
        - Learning rate annealing
        - Explained variance tracking
        - Action masking
        """
        os.makedirs(save_path, exist_ok=True)

        n_updates = total_timesteps // n_steps

        print("="*70)
        print("INDEPENDENT PPO TRAINING")
        print("="*70)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Steps per rollout: {n_steps}")
        print(f"Number of updates: {n_updates}")
        print(f"Agents: {self.env.possible_agents}")
        print(f"Run name: {self.run_name}")
        print(f"TensorBoard: runs/{self.run_name}")
        print("="*70)

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
                stats = self.agents[agent_name].update(
                    next_obs_dict=next_observations.get(agent_name, {
                        'observation': np.zeros(self.agents[agent_name].network.shared[0].in_features),
                        'action_mask': np.ones(self.agents[agent_name].network.actor.out_features)
                    }),
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

            # Log learning rate
            if self.anneal_lr:
                self.writer.add_scalar("charts/learning_rate", lr_now, current_timesteps)

            # Log SPS (steps per second)
            elapsed = time.time() - start_time
            sps = current_timesteps / elapsed
            self.writer.add_scalar("charts/SPS", sps, current_timesteps)

            # Console logging
            if update % log_interval == 0:
                print(f"\n{'='*70}")
                print(f"Update {update}/{n_updates} | Timesteps: {current_timesteps:,} | SPS: {sps:.0f}")
                if self.anneal_lr:
                    print(f"Learning Rate: {lr_now:.6f}")
                print(f"{'='*70}")

                for agent_name, stats in update_stats.items():
                    print(f"\n{agent_name}:")
                    print(f"  Policy Loss: {stats['policy_loss']:.4f}")
                    print(f"  Value Loss: {stats['value_loss']:.4f}")
                    print(f"  Entropy: {stats['entropy']:.4f}")
                    print(f"  Clip Fraction: {stats['clip_fraction']:.4f}")
                    print(f"  Approx KL: {stats['approx_kl']:.4f}")
                    print(f"  Explained Var: {stats['explained_variance']:.4f}")

            # Save checkpoints
            if update % save_interval == 0:
                for agent_name in self.env.possible_agents:
                    model_path = os.path.join(save_path, f"{agent_name}_update_{update}.pt")
                    torch.save(self.agents[agent_name].network.state_dict(), model_path)
                print(f"\n✓ Saved checkpoints at update {update}")

        # Save final models
        for agent_name in self.env.possible_agents:
            model_path = os.path.join(save_path, f"{agent_name}_final.pt")
            torch.save(self.agents[agent_name].network.state_dict(), model_path)

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"Total time: {time.time() - start_time:.2f}s")
        print(f"Final models saved to: {save_path}")
        print(f"TensorBoard logs: runs/{self.run_name}")
        print(f"{'='*70}\n")

        self.writer.close()

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True):
        """Evaluate learned policies with action masking."""
        print(f"\n{'='*70}")
        print(f"EVALUATION ({n_episodes} episodes)")
        print(f"{'='*70}\n")

        episode_rewards = {agent: [] for agent in self.env.possible_agents}
        episode_lengths = []

        for episode in range(n_episodes):
            observations, _ = self.env.reset()
            ep_rewards = {agent: 0.0 for agent in self.env.possible_agents}
            ep_length = 0

            done = False
            while not done:
                # Get actions (deterministic for evaluation)
                actions = {}
                for agent_name in self.env.agents:
                    action = self.agents[agent_name].network.get_action(
                        observations[agent_name],
                        deterministic=deterministic
                    )
                    actions[agent_name] = action

                # Step
                observations, rewards, terminations, truncations, infos = self.env.step(actions)

                # Track rewards
                for agent_name in self.env.agents:
                    ep_rewards[agent_name] += rewards[agent_name]

                ep_length += 1
                done = all(terminations.values()) or all(truncations.values())

            # Store episode stats
            for agent_name in self.env.possible_agents:
                episode_rewards[agent_name].append(ep_rewards.get(agent_name, 0.0))
            episode_lengths.append(ep_length)

            print(f"Episode {episode + 1}: Length={ep_length}, Rewards={ep_rewards}")

        # Print summary
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        for agent_name in self.env.possible_agents:
            mean_reward = np.mean(episode_rewards[agent_name])
            std_reward = np.std(episode_rewards[agent_name])
            print(f"{agent_name}: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"Mean Episode Length: {np.mean(episode_lengths):.1f}")
        print(f"{'='*70}\n")


def main():
    """Main training script."""
    # Create environment
    env = JobMarketEnv(
        num_companies=3,
        num_workers=10,
        max_workers_per_company=5,
        max_timesteps=100,
        seed=42
    )

    # Create trainer with unique run name
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"hirerl_ippo_{timestamp}"

    trainer = IPPOTrainer(
        env=env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        device='cpu',
        seed=42,
        anneal_lr=True,
        run_name=run_name
    )

    # Train
    trainer.train(
        total_timesteps=1_000_000,  # 1M steps
        n_steps=2048,             # Collect 2048 steps before update
        n_epochs=4,               # 4 epochs per update
        batch_size=256,            # Batch size 64
        log_interval=5,           # Log every 5 updates
        save_interval=50000,         # Save every 50000 updates
        save_path='checkpoints'
    )

    # Evaluate
    trainer.evaluate(n_episodes=10, deterministic=True)


if __name__ == '__main__':
    main()
