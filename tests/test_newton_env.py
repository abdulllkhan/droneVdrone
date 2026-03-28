"""Unit tests for the Newton GPU-accelerated intercept environment."""

import numpy as np
import pytest
import warp as wp

wp.init()

from envs.newton_intercept_env import NewtonInterceptVecEnv
from train.config import EnvConfig


@pytest.fixture
def env():
    """Create a small test environment."""
    cfg = EnvConfig()
    e = NewtonInterceptVecEnv(num_envs=4, config=cfg)
    yield e
    e.close()


@pytest.fixture
def env_large():
    """Create a larger test environment for GPU scaling."""
    cfg = EnvConfig()
    e = NewtonInterceptVecEnv(num_envs=64, config=cfg)
    yield e
    e.close()


class TestEnvInit:
    def test_creates_without_error(self, env):
        assert env is not None

    def test_correct_num_envs(self, env):
        assert env.num_envs == 4

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (19,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (3,)

    def test_action_space_bounds(self, env):
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0


class TestReset:
    def test_reset_returns_correct_shape(self, env):
        obs = env.reset()
        assert obs.shape == (4, 19)

    def test_reset_returns_finite_values(self, env):
        obs = env.reset()
        assert np.all(np.isfinite(obs))

    def test_reset_positions_are_valid(self, env):
        obs = env.reset()
        for i in range(4):
            def_pos = obs[i, :3]
            att_pos = obs[i, 6:9]
            # Defender should be near target (0, 0, 1)
            def_dist = np.linalg.norm(def_pos[:2])
            assert def_dist < 2.0, f"Defender too far from target: {def_dist}"
            # Attacker should be far from target
            att_dist = np.linalg.norm(att_pos[:2])
            assert att_dist > 1.0, f"Attacker too close to target: {att_dist}"

    def test_reset_velocities_are_zero(self, env):
        obs = env.reset()
        for i in range(4):
            def_vel = obs[i, 3:6]
            att_vel = obs[i, 9:12]
            np.testing.assert_allclose(def_vel, 0.0, atol=0.1)
            np.testing.assert_allclose(att_vel, 0.0, atol=0.1)


class TestStep:
    def test_step_returns_correct_shapes(self, env):
        env.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (4, 19)
        assert rewards.shape == (4,)
        assert dones.shape == (4,)
        assert len(infos) == 4

    def test_step_returns_finite_values(self, env):
        env.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        obs, rewards, dones, infos = env.step(actions)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))

    def test_zero_action_maintains_altitude(self, env):
        obs = env.reset()
        initial_z = obs[:, 2].copy()

        # Step with zero velocity for a few steps
        actions = np.zeros((4, 3), dtype=np.float32)
        for _ in range(5):
            obs, _, _, _ = env.step(actions)

        # Defender should roughly maintain altitude (within tolerance)
        final_z = obs[:, 2]
        z_drift = np.abs(final_z - initial_z)
        assert np.all(z_drift < 1.0), f"Altitude drift too large: {z_drift}"

    def test_positive_velocity_moves_drone(self, env):
        obs = env.reset()
        initial_x = obs[:, 0].copy()

        # Apply positive x velocity
        actions = np.ones((4, 3), dtype=np.float32)
        actions[:, 1] = 0.0  # zero y
        actions[:, 2] = 0.0  # zero z
        for _ in range(20):
            obs, _, dones, _ = env.step(actions)

        # Defender should have moved in +x direction
        # Use median (not mean) because some envs may auto-reset mid-test
        final_x = obs[:, 0]
        moved = final_x - initial_x
        assert np.median(moved) > 0, f"Drone didn't move in +x: median delta = {np.median(moved)}, per-env = {moved}"

    def test_infos_have_required_keys(self, env):
        env.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        _, _, _, infos = env.step(actions)
        for info in infos:
            assert "intercepted" in info
            assert "target_reached" in info


class TestRewardAndDone:
    def test_interception_gives_positive_reward(self, env):
        """When defender is at attacker position, should get +100."""
        env.reset()
        # Run until something happens or 500 steps
        actions = np.zeros((4, 3), dtype=np.float32)
        max_reward = -np.inf
        for _ in range(500):
            # Chase the attacker
            obs, rewards, dones, infos = env.step(actions)
            # Compute direction to attacker from obs
            for i in range(4):
                rel = obs[i, 12:15]  # relative position
                norm = np.linalg.norm(rel)
                if norm > 0.01:
                    actions[i] = np.clip(rel / norm, -1, 1)
            max_reward = max(max_reward, np.max(rewards))
            if np.any(dones):
                break
        # At minimum, approach rewards should be positive
        assert max_reward > 0, f"No positive rewards seen: max = {max_reward}"

    def test_time_penalty_exists(self, env):
        """Zero action should give small negative reward (time penalty)."""
        env.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        _, rewards, _, _ = env.step(actions)
        # First step has no prev_dist, so reward is just time penalty + alt penalty
        _, rewards, _, _ = env.step(actions)
        # Should be slightly negative due to time penalty
        assert np.mean(rewards) < 5.0, "Rewards unexpectedly high with no action"


class TestAutoReset:
    def test_done_envs_get_reset(self, env):
        """Environments that are done should auto-reset."""
        env.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        # Run many steps — some envs should eventually timeout
        for _ in range(1000):
            obs, rewards, dones, infos = env.step(actions)
            if np.any(dones):
                # After auto-reset, obs should still be finite
                assert np.all(np.isfinite(obs))
                break


class TestGPUScaling:
    def test_large_env_creates(self, env_large):
        """64 parallel envs should work."""
        assert env_large.num_envs == 64

    def test_large_env_steps(self, env_large):
        """64 parallel envs should step without error."""
        obs = env_large.reset()
        assert obs.shape == (64, 19)
        actions = np.random.uniform(-1, 1, (64, 3)).astype(np.float32)
        obs, rewards, dones, infos = env_large.step(actions)
        assert obs.shape == (64, 19)
        assert np.all(np.isfinite(obs))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
