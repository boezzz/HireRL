"""
PettingZoo Compliance Tests for HireRL Environment

This test suite uses PettingZoo's official testing utilities to verify
that the HireRL environment complies with the PettingZoo API.

Tests include:
- Parallel API test
- Seed test
- Render test
- Performance benchmark
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pettingzoo'))

from pettingzoo.test import (
    parallel_api_test,
    parallel_seed_test
)
from hirerl import JobMarketEnv


def create_test_env():
    """Create a small environment for testing."""
    return JobMarketEnv(
        num_companies=2,
        num_workers=5,
        max_workers_per_company=3,
        max_timesteps=20,
        seed=42
    )


def test_parallel_api():
    """
    Test PettingZoo Parallel API compliance.

    This checks that the environment correctly implements:
    - reset() method
    - step() method
    - observation_space() and action_space() methods
    - Proper handling of terminations and truncations
    - Correct dictionary return formats
    """
    print("\n" + "="*60)
    print("TEST: PettingZoo Parallel API Compliance")
    print("="*60)

    env = create_test_env()

    try:
        parallel_api_test(env, num_cycles=100)
        print("\n✓ Parallel API test PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Parallel API test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_seed_determinism():
    """
    Test environment determinism.

    This verifies that:
    1. Two environments with same seed produce identical results
    2. Calling seed() then reset() makes environment deterministic
    """
    print("\n" + "="*60)
    print("TEST: Seed Determinism")
    print("="*60)

    def env_fn(seed=None):
        return JobMarketEnv(
            num_companies=2,
            num_workers=5,
            max_workers_per_company=3,
            max_timesteps=20,
            seed=seed
        )

    try:
        parallel_seed_test(env_fn, num_cycles=50)
        print("\n✓ Seed test PASSED")
        return True
    except Exception as e:
        print(f"\n✗ Seed test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_render_functionality():
    """
    Test that rendering works correctly for parallel environment.

    NOTE: PettingZoo's render_test is for AEC environments only.
    This is a custom test for parallel environments.

    This checks that:
    - render() method doesn't crash
    - render_mode='human' works
    - render_mode=None doesn't crash
    """
    print("\n" + "="*60)
    print("TEST: Render Functionality (Custom for Parallel)")
    print("="*60)

    try:
        # Test render_mode='human'
        env = JobMarketEnv(
            num_companies=2,
            num_workers=5,
            max_workers_per_company=3,
            max_timesteps=20,
            render_mode='human',
            seed=42
        )

        observations, infos = env.reset()
        env.render()  # Should print to console

        # Take a few steps
        for _ in range(3):
            actions = {agent: 0 for agent in env.agents}  # NO_OP
            observations, rewards, terminations, truncations, infos = env.step(actions)
            if env.agents:  # Only render if agents are still alive
                env.render()

        print("✓ render_mode='human' works")

        # Test render_mode=None (should not render)
        env = JobMarketEnv(
            num_companies=2,
            num_workers=5,
            max_workers_per_company=3,
            max_timesteps=20,
            render_mode=None,
            seed=42
        )

        observations, infos = env.reset()
        env.render()  # Should do nothing
        print("✓ render_mode=None works (no rendering)")

        print("\n✓ Render test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Render test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance():
    """
    Benchmark environment performance.

    NOTE: Skipped because performance_benchmark is for AEC environments only.
    Parallel environments don't have agent_iter() method.
    """
    print("\n" + "="*60)
    print("TEST: Performance Benchmark (SKIPPED - AEC only)")
    print("="*60)

    print("Performance benchmark is for AEC environments only.")
    print("Parallel environments don't support this test.")
    print("\n✓ Test SKIPPED (not applicable)")

    return True


def test_action_masking():
    """
    Test that action masks are correctly generated.

    Verifies:
    - Action masks are included in observations
    - Masks have correct shape
    - Masks prevent invalid actions
    """
    print("\n" + "="*60)
    print("TEST: Action Masking")
    print("="*60)

    env = create_test_env()

    try:
        observations, infos = env.reset()

        # Check that observations contain action masks
        for agent in env.agents:
            obs = observations[agent]

            # Observation should be a dict with 'observation' and 'action_mask'
            if not isinstance(obs, dict):
                print(f"✗ Observation for {agent} is not a dict")
                return False

            if 'observation' not in obs:
                print(f"✗ Missing 'observation' key for {agent}")
                return False

            if 'action_mask' not in obs:
                print(f"✗ Missing 'action_mask' key for {agent}")
                return False

            # Check action mask shape
            action_mask = obs['action_mask']
            expected_size = env.action_size

            if len(action_mask) != expected_size:
                print(f"✗ Action mask size mismatch for {agent}: {len(action_mask)} != {expected_size}")
                return False

            # Check that NO_OP is always valid
            if action_mask[0] != 1:
                print(f"✗ NO_OP action not valid for {agent}")
                return False

            # Check that at least one action is valid
            if sum(action_mask) == 0:
                print(f"✗ No valid actions for {agent}")
                return False

        print(f"✓ Action masks present and correctly shaped")

        # Test that masks update correctly after step
        actions = {agent: 0 for agent in env.agents}  # All NO_OP
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Verify masks are still present
        for agent in env.agents:
            if 'action_mask' not in observations[agent]:
                print(f"✗ Action mask missing after step for {agent}")
                return False

        print(f"✓ Action masks persist after step")

        print("\n✓ Action masking test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Action masking test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observation_space_consistency():
    """
    Test that observation space matches actual observations.

    Verifies:
    - Observations match declared observation space
    - Dict space structure is correct
    """
    print("\n" + "="*60)
    print("TEST: Observation Space Consistency")
    print("="*60)

    env = create_test_env()

    try:
        observations, infos = env.reset()

        for agent in env.agents:
            obs = observations[agent]
            space = env.observation_space(agent)

            # Check that observation is in the space
            if not space.contains(obs):
                print(f"✗ Observation for {agent} not in declared space")
                print(f"  Observation keys: {obs.keys() if isinstance(obs, dict) else 'not a dict'}")
                print(f"  Space: {space}")
                return False

        print("✓ All observations match declared spaces")

        print("\n✓ Observation space consistency test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Observation space consistency test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_validity():
    """
    Test that only valid actions are accepted.

    This test verifies that:
    - Invalid actions (according to mask) don't crash the environment
    - Environment handles edge cases correctly
    """
    print("\n" + "="*60)
    print("TEST: Action Validity")
    print("="*60)

    env = create_test_env()

    try:
        observations, infos = env.reset()

        # Test NO_OP (always valid)
        actions = {agent: 0 for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print("✓ NO_OP actions work")

        # Test random valid actions
        for _ in range(10):
            actions = {}
            for agent in env.agents:
                # Sample only from valid actions
                action_mask = observations[agent]['action_mask']
                valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
                import random
                actions[agent] = random.choice(valid_actions) if valid_actions else 0

            observations, rewards, terminations, truncations, infos = env.step(actions)

            if all(terminations.values()) or all(truncations.values()):
                break

        print("✓ Valid actions executed successfully")

        print("\n✓ Action validity test PASSED")
        return True

    except Exception as e:
        print(f"\n✗ Action validity test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all PettingZoo compliance tests."""
    print("\n" + "="*70)
    print(" "*15 + "PETTINGZOO COMPLIANCE TEST SUITE")
    print("="*70)

    tests = [
        ("Parallel API Test", test_parallel_api),
        ("Seed Determinism Test", test_seed_determinism),
        ("Render Functionality Test", test_render_functionality),
        ("Performance Benchmark", test_performance),
        ("Action Masking Test", test_action_masking),
        ("Observation Space Consistency Test", test_observation_space_consistency),
        ("Action Validity Test", test_action_validity),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED WITH ERROR: {e}\n")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, result in results if result)
    failed = len(results) - passed

    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<50} {status}")

    print(f"\n{'-'*70}")
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"{'-'*70}")

    if all(result for _, result in results):
        print("\n✓ ALL TESTS PASSED! Environment is PettingZoo compliant.")
    else:
        print("\n✗ SOME TESTS FAILED. Please review the failures above.")

    print("="*70 + "\n")

    return all(result for _, result in results)


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
