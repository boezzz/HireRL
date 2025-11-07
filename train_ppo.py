"""
IPPO Training for Job Market Environment
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pettingzoo'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from typing import Dict, List, Tuple
import time
from collections import defaultdict

from hirerl import JobMarketEnv
from utils import EpisodeLogger, PerformanceMetrics


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    Architecture:
    - Shared feature extractor
    - Actor head (policy): outputs action probabilities
    - Critic head (value): outputs state value estimate
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic head (value function)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        features = self.shared(x)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value

    def get_action(self, obs, deterministic=False):
        """Sample action from policy."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
            action_logits, value = self.forward(obs_tensor)

            if deterministic:
                action = action_logits.argmax(dim=-1)
            else:
                probs = torch.softmax(action_logits, dim=-1)
                dist = Categorical(probs)
                action = dist.sample()

            return action.item()

    def evaluate_actions(self, obs, actions):
        """Evaluate log probabilities and values for given observations and actions."""
        action_logits, values = self.forward(obs)
        probs = torch.softmax(action_logits, dim=-1)
        dist = Categorical(probs)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return log_probs, values.squeeze(-1), entropy


class RolloutBuffer:
    """
    Buffer for storing experience during rollout.

    Stores: observations, actions, rewards, values, log_probs, dones
    """

    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def add(self, obs, action, reward, value, log_prob, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()

    def get(self):
        return {
            'observations': np.array(self.observations),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'values': np.array(self.values),
            'log_probs': np.array(self.log_probs),
            'dones': np.array(self.dones)
        }


class PPOAgent:
    """
    Proximal Policy Optimization agent.

    Implements PPO with clipped surrogate objective.
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

        # Create network
        self.network = ActorCritic(obs_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Buffer
        self.buffer = RolloutBuffer()

    def get_action(self, obs, deterministic=False):
        """Get action from policy (for rollout)."""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action_logits, value = self.network(obs_tensor)
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

    def update(self, next_obs, n_epochs=4, batch_size=64):
        """Update policy using PPO."""
        # Get rollout data
        data = self.buffer.get()

        # Compute next value
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            _, next_value = self.network(next_obs_tensor)
            next_value = next_value.item()

        # Compute advantages and returns
        advantages, returns = self.compute_gae(
            data['rewards'], data['values'], data['dones'], next_value
        )

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        obs_tensor = torch.FloatTensor(data['observations']).to(self.device)
        actions_tensor = torch.LongTensor(data['actions']).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(data['log_probs']).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)

        # PPO update
        n_samples = len(data['observations'])

        stats = {
            'policy_loss': [],
            'value_loss': [],
            'entropy': [],
            'total_loss': [],
            'clip_fraction': []
        }

        for epoch in range(n_epochs):
            # Generate random indices for mini-batches
            indices = np.random.permutation(n_samples)

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                batch_indices = indices[start:end]

                # Get batch
                obs_batch = obs_tensor[batch_indices]
                actions_batch = actions_tensor[batch_indices]
                old_log_probs_batch = old_log_probs_tensor[batch_indices]
                advantages_batch = advantages_tensor[batch_indices]
                returns_batch = returns_tensor[batch_indices]

                # Evaluate actions
                log_probs, values, entropy = self.network.evaluate_actions(obs_batch, actions_batch)

                # Policy loss (clipped surrogate objective)
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns_batch)

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

                stats['policy_loss'].append(policy_loss.item())
                stats['value_loss'].append(value_loss.item())
                stats['entropy'].append(-entropy_loss.item())
                stats['total_loss'].append(loss.item())
                stats['clip_fraction'].append(clip_fraction.item())

        # Clear buffer
        self.buffer.clear()

        # Return averaged stats
        return {k: np.mean(v) for k, v in stats.items()}


class IPPOTrainer:
    """
    Independent PPO Trainer for multi-agent environment.

    Each company has its own PPO agent that learns independently.
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
        seed: int = 42
    ):
        self.env = env
        self.device = device

        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Create PPO agent for each company
        self.agents: Dict[str, PPOAgent] = {}

        for agent_name in env.agents:
            obs_dim = env.observation_space(agent_name).shape[0]
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

        # Tracking
        self.episode_rewards = {agent: [] for agent in env.agents}
        self.episode_lengths = []

    def collect_rollout(self, n_steps: int):
        """Collect n_steps of experience for each agent."""
        observations, _ = self.env.reset()

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

            # Store experience in buffers
            for agent_name in self.env.agents:
                done = terminations[agent_name] or truncations[agent_name]

                self.agents[agent_name].buffer.add(
                    obs=observations[agent_name],
                    action=actions[agent_name],
                    reward=rewards[agent_name],
                    value=values[agent_name],
                    log_prob=log_probs[agent_name],
                    done=done
                )

            observations = next_observations

            # Check if episode ended
            if all(terminations.values()) or all(truncations.values()):
                observations, _ = self.env.reset()

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

        Args:
            total_timesteps: Total number of environment steps
            n_steps: Steps per rollout (before update)
            n_epochs: Number of epochs per update
            batch_size: Batch size for PPO updates
            log_interval: Log stats every N updates
            save_interval: Save models every N updates
            save_path: Directory to save checkpoints
        """
        os.makedirs(save_path, exist_ok=True)

        n_updates = total_timesteps // n_steps

        print("="*70)
        print("INDEPENDENT PPO TRAINING")
        print("="*70)
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Steps per rollout: {n_steps}")
        print(f"Number of updates: {n_updates}")
        print(f"Agents: {self.env.agents}")
        print("="*70)

        start_time = time.time()

        for update in range(1, n_updates + 1):
            # Collect rollout
            next_observations = self.collect_rollout(n_steps)

            # Update all agents
            update_stats = {}
            for agent_name in self.env.agents:
                stats = self.agents[agent_name].update(
                    next_obs=next_observations[agent_name],
                    n_epochs=n_epochs,
                    batch_size=batch_size
                )
                update_stats[agent_name] = stats

            # Log progress
            if update % log_interval == 0:
                elapsed = time.time() - start_time
                fps = (update * n_steps) / elapsed

                print(f"\n{'='*70}")
                print(f"Update {update}/{n_updates} | Timesteps: {update * n_steps:,} | FPS: {fps:.0f}")
                print(f"{'='*70}")

                for agent_name in self.env.agents:
                    stats = update_stats[agent_name]
                    print(f"\n{agent_name}:")
                    print(f"  Policy Loss: {stats['policy_loss']:.4f}")
                    print(f"  Value Loss: {stats['value_loss']:.4f}")
                    print(f"  Entropy: {stats['entropy']:.4f}")
                    print(f"  Clip Fraction: {stats['clip_fraction']:.4f}")

            # Save checkpoints
            if update % save_interval == 0:
                for agent_name in self.env.agents:
                    model_path = os.path.join(save_path, f"{agent_name}_update_{update}.pt")
                    torch.save(self.agents[agent_name].network.state_dict(), model_path)
                print(f"\n✓ Saved checkpoints at update {update}")

        # Save final models
        for agent_name in self.env.agents:
            model_path = os.path.join(save_path, f"{agent_name}_final.pt")
            torch.save(self.agents[agent_name].network.state_dict(), model_path)

        print(f"\n{'='*70}")
        print("TRAINING COMPLETE!")
        print(f"Total time: {time.time() - start_time:.2f}s")
        print(f"Final models saved to: {save_path}")
        print(f"{'='*70}\n")

    def evaluate(self, n_episodes: int = 10, deterministic: bool = True):
        """Evaluate learned policies."""
        print(f"\n{'='*70}")
        print(f"EVALUATION ({n_episodes} episodes)")
        print(f"{'='*70}\n")

        episode_rewards = {agent: [] for agent in self.env.agents}
        episode_lengths = []

        for episode in range(n_episodes):
            observations, _ = self.env.reset()
            ep_rewards = {agent: 0.0 for agent in self.env.agents}
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
            for agent_name in self.env.agents:
                episode_rewards[agent_name].append(ep_rewards[agent_name])
            episode_lengths.append(ep_length)

            print(f"Episode {episode + 1}: Length={ep_length}, Rewards={ep_rewards}")

        # Print summary
        print(f"\n{'='*70}")
        print("EVALUATION SUMMARY")
        print(f"{'='*70}")
        for agent_name in self.env.agents:
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

    # Create trainer
    trainer = IPPOTrainer(
        env=env,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        device='cpu',
        seed=42
    )

    # Train
    trainer.train(
        total_timesteps=100_000,  # 100k steps
        n_steps=2048,             # Collect 2048 steps before update
        n_epochs=4,               # 4 epochs per update
        batch_size=64,            # Batch size 64
        log_interval=10,          # Log every 10 updates
        save_interval=25,         # Save every 25 updates
        save_path='checkpoints'
    )

    # Evaluate
    trainer.evaluate(n_episodes=10, deterministic=True)


if __name__ == '__main__':
    main()
