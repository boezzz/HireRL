"""
Test IPPO Training
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pettingzoo'))

import numpy as np
import torch

from train_ppo import IPPOTrainer, PPOAgent
from hirerl import JobMarketEnv


def test_ppo_agent_creation():
    """Test that PPO agent can be created."""
    print("\n" + "="*60)
    print("TEST: PPO Agent Creation")
    print("="*60)

    agent = PPOAgent(obs_dim=41, action_dim=16, device='cpu')

    print(f"✓ Created PPO agent")
    print(f"  Observation dim: 41")
    print(f"  Action dim: 16")
    print(f"  Network parameters: {sum(p.numel() for p in agent.network.parameters()):,}")

    # Test forward pass
    obs = np.random.randn(41).astype(np.float32)
    action, value, log_prob = agent.get_action(obs)

    print(f"✓ Forward pass successful")
    print(f"  Sample action: {action}")
    print(f"  Value estimate: {value:.4f}")
    print(f"  Log prob: {log_prob:.4f}")

    print("\n✓ TEST PASSED\n")
    return True


def test_rollout_collection():
    """Test that rollouts can be collected."""
    print("\n" + "="*60)
    print("TEST: Rollout Collection")
    print("="*60)

    env = JobMarketEnv(
        num_companies=2,
        num_workers=5,
        max_workers_per_company=3,
        max_timesteps=20,
        seed=42
    )

    trainer = IPPOTrainer(env=env, device='cpu', seed=42)

    print(f"✓ Created IPPO trainer")
    print(f"  Number of agents: {len(trainer.agents)}")

    # Collect short rollout
    next_obs = trainer.collect_rollout(n_steps=10)

    print(f"✓ Collected 10-step rollout")

    # Check buffers
    for agent_name, agent in trainer.agents.items():
        buffer_size = len(agent.buffer.observations)
        print(f"  {agent_name}: buffer size = {buffer_size}")

    print("\n✓ TEST PASSED\n")
    return True


def test_ppo_update():
    """Test that PPO update works."""
    print("\n" + "="*60)
    print("TEST: PPO Update")
    print("="*60)

    env = JobMarketEnv(
        num_companies=2,
        num_workers=5,
        max_workers_per_company=3,
        max_timesteps=20,
        seed=42
    )

    trainer = IPPOTrainer(env=env, device='cpu', seed=42)

    # Collect rollout
    next_obs = trainer.collect_rollout(n_steps=50)

    print(f"✓ Collected 50-step rollout")

    # Perform update
    for agent_name, agent in trainer.agents.items():
        stats = agent.update(
            next_obs=next_obs[agent_name],
            n_epochs=2,
            batch_size=16
        )

        print(f"\n{agent_name} update stats:")
        print(f"  Policy loss: {stats['policy_loss']:.4f}")
        print(f"  Value loss: {stats['value_loss']:.4f}")
        print(f"  Entropy: {stats['entropy']:.4f}")
        print(f"  Clip fraction: {stats['clip_fraction']:.4f}")

    print("\n✓ TEST PASSED\n")
    return True


def test_short_training():
    """Test short training run."""
    print("\n" + "="*60)
    print("TEST: Short Training Run")
    print("="*60)

    env = JobMarketEnv(
        num_companies=2,
        num_workers=5,
        max_workers_per_company=3,
        max_timesteps=20,
        seed=42
    )

    trainer = IPPOTrainer(env=env, device='cpu', seed=42)

    print("Starting short training run (1000 steps)...\n")

    trainer.train(
        total_timesteps=1000,
        n_steps=200,
        n_epochs=2,
        batch_size=32,
        log_interval=5,
        save_interval=100,
        save_path='test_checkpoints'
    )

    print("\n✓ TEST PASSED\n")
    return True


def test_evaluation():
    """Test evaluation."""
    print("\n" + "="*60)
    print("TEST: Evaluation")
    print("="*60)

    env = JobMarketEnv(
        num_companies=2,
        num_workers=5,
        max_workers_per_company=3,
        max_timesteps=20,
        seed=42
    )

    trainer = IPPOTrainer(env=env, device='cpu', seed=42)

    # Train briefly
    trainer.train(
        total_timesteps=500,
        n_steps=100,
        n_epochs=2,
        batch_size=32,
        log_interval=100,
        save_interval=1000,
        save_path='test_checkpoints'
    )

    # Evaluate
    print("\nEvaluating...")
    trainer.evaluate(n_episodes=3, deterministic=True)

    print("\n✓ TEST PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*20 + "PPO TRAINING TEST SUITE")
    print("="*70)

    tests = [
        test_ppo_agent_creation,
        test_rollout_collection,
        test_ppo_update,
        test_short_training,
        test_evaluation
    ]

    results = []

    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"\n✗ TEST FAILED WITH ERROR: {e}\n")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {sum(results)}")
    print(f"Failed: {len(results) - sum(results)}")

    if all(results):
        print("\n✓ ALL TESTS PASSED!")
    else:
        print("\n✗ SOME TESTS FAILED")

    print("="*70 + "\n")

    # Cleanup
    import shutil
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')
        print("Cleaned up test checkpoints\n")


if __name__ == '__main__':
    main()
