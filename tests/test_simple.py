import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pettingzoo'))

import numpy as np

from hirerl import JobMarketEnv
from policies import RandomPolicy, GreedyPolicy
from utils import verify_environment, print_environment_info, EpisodeLogger


def test_basic_functionality():
    """Test basic environment functionality."""
    print("\n" + "="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)

    # Create simple environment
    env = JobMarketEnv(
        num_companies=2,
        num_workers=5,
        max_workers_per_company=3,
        max_timesteps=20,
        seed=42
    )

    # Print configuration
    print_environment_info(env)

    # Verify environment
    success = verify_environment(env, num_steps=10, verbose=True)

    if success:
        print("\n✓ TEST 1 PASSED\n")
    else:
        print("\n✗ TEST 1 FAILED\n")

    return success


def test_random_policy():
    """Test random policy."""
    print("\n" + "="*60)
    print("TEST 2: Random Policy")
    print("="*60)

    env = JobMarketEnv(
        num_companies=2,
        num_workers=5,
        max_workers_per_company=3,
        max_timesteps=20,
        seed=42
    )

    policy = RandomPolicy(num_workers=5, seed=42)
    logger = EpisodeLogger(env.agents)

    observations, infos = env.reset()
    print("Episode started...")

    for step in range(20):
        actions = {agent: policy.get_action(observations[agent], agent) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        logger.log_step(step, rewards, infos, actions)

        if step % 5 == 0:
            print(f"Step {step}: Rewards = {rewards}")

        if all(terminations.values()) or all(truncations.values()):
            break

    summary = logger.get_summary()
    print("\nEpisode Summary:")
    for agent, stats in summary['agents'].items():
        print(f"  {agent}: Total Reward = {stats['total_reward']:.2f}")
    print(f"  Market: Final Unemployment = {summary['market']['final_unemployment_rate']:.2%}")

    print("\n✓ TEST 2 PASSED\n")
    return True


def test_greedy_policy():
    """Test greedy policy."""
    print("\n" + "="*60)
    print("TEST 3: Greedy Policy")
    print("="*60)

    env = JobMarketEnv(
        num_companies=2,
        num_workers=5,
        max_workers_per_company=3,
        max_timesteps=20,
        seed=42
    )

    policy = GreedyPolicy(
        num_workers=5,
        num_companies=2,
        max_workers_per_company=3,
        ability_dim=1
    )

    logger = EpisodeLogger(env.agents)
    observations, infos = env.reset()

    print("Episode started...")

    for step in range(20):
        actions = {agent: policy.get_action(observations[agent], agent) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        logger.log_step(step, rewards, infos, actions)

        if step % 5 == 0:
            print(f"Step {step}: Rewards = {rewards}")

        if all(terminations.values()) or all(truncations.values()):
            break

    summary = logger.get_summary()
    print("\nEpisode Summary:")
    for agent, stats in summary['agents'].items():
        print(f"  {agent}: Total Reward = {stats['total_reward']:.2f}")
    print(f"  Market: Final Unemployment = {summary['market']['final_unemployment_rate']:.2%}")

    print("\n✓ TEST 3 PASSED\n")
    return True


def test_worker_pool():
    """Test worker pool mechanics."""
    print("\n" + "="*60)
    print("TEST 4: Worker Pool Mechanics")
    print("="*60)

    from workers import WorkerPool

    pool = WorkerPool(num_workers=5, ability_dim=1, seed=42)
    workers = pool.reset()

    print(f"Initialized {len(workers)} workers")

    # Test hiring
    pool.hire_worker(0, company_id=0, wage=1.5)
    pool.hire_worker(1, company_id=0, wage=1.2)
    pool.hire_worker(2, company_id=1, wage=1.8)

    print(f"Hired 3 workers")
    print(f"Unemployment rate: {pool.get_unemployment_rate():.2%}")
    print(f"Average wage: {pool.get_average_wage():.2f}")

    # Test experience accumulation
    pool.update_experience_and_tenure()
    print(f"Updated experience")

    # Test firing
    pool.fire_worker(1)
    print(f"Fired worker 1")
    print(f"Unemployment rate: {pool.get_unemployment_rate():.2%}")

    # Test quit rule
    quit_workers = pool.apply_deterministic_quit_rule()
    print(f"Workers who quit: {quit_workers}")

    print("\n✓ TEST 4 PASSED\n")
    return True


def test_screening():
    """Test screening mechanism."""
    print("\n" + "="*60)
    print("TEST 5: Screening Mechanism")
    print("="*60)

    from screening import ScreeningMechanism, ScreeningTechnology

    screening = ScreeningMechanism(
        technology=ScreeningTechnology.SQRT,
        c_max=1.0,
        seed=42
    )

    # Test precision function
    costs = [0.0, 0.1, 0.5, 1.0, 2.0]
    print("Cost-Precision Relationship:")
    for c in costs:
        precision = screening.get_precision(c)
        print(f"  Cost = {c:.2f} -> Precision = {precision:.3f}")

    # Test screening
    sigma_true = np.array([1.5], dtype=np.float32)
    sigma_hat_0 = np.array([1.0], dtype=np.float32)

    print(f"\nTrue ability: {sigma_true[0]:.2f}")
    print(f"Public signal: {sigma_hat_0[0]:.2f}")

    for c in [0.0, 0.5, 1.0]:
        estimate, precision = screening.screen_worker(sigma_true, sigma_hat_0, c)
        print(f"Screening with cost {c:.2f}: Estimate = {estimate[0]:.2f}, Precision = {precision:.3f}")

    print("\n✓ TEST 5 PASSED\n")
    return True


def main():
    """Run all tests."""
    print("\n" + "="*70)
    print(" "*20 + "HIRERL SIMPLE TEST SUITE")
    print("="*70)

    tests = [
        test_basic_functionality,
        test_random_policy,
        test_greedy_policy,
        test_worker_pool,
        test_screening
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


if __name__ == '__main__':
    main()
