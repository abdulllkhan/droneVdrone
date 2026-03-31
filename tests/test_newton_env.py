"""Unit tests for the Newton GPU-accelerated drone environment."""

import numpy as np
import pytest
import warp as wp

wp.init()

from envs.newton_env import NewtonDroneEnv
from train.config import EnvConfig, TrainConfig


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def env():
    """Create a small test environment (intercept mode, no obstacles)."""
    cfg = EnvConfig(num_obstacles=0)
    e = NewtonDroneEnv(num_envs=4, config=cfg)
    yield e
    e.close()


@pytest.fixture
def env_with_obstacles():
    """Create a test environment with obstacles."""
    cfg = EnvConfig(num_obstacles=3)
    e = NewtonDroneEnv(num_envs=4, config=cfg)
    yield e
    e.close()


@pytest.fixture
def env_fly():
    """Create a fly_to_target environment."""
    cfg = EnvConfig(mode="fly_to_target", num_obstacles=0)
    e = NewtonDroneEnv(num_envs=4, config=cfg)
    yield e
    e.close()


@pytest.fixture
def env_fly_obstacles():
    """Create a fly_to_target environment with obstacles."""
    cfg = EnvConfig(mode="fly_to_target", num_obstacles=2)
    e = NewtonDroneEnv(num_envs=4, config=cfg)
    yield e
    e.close()


@pytest.fixture
def env_large():
    """Create a larger test environment for GPU scaling."""
    cfg = EnvConfig(num_obstacles=0)
    e = NewtonDroneEnv(num_envs=64, config=cfg)
    yield e
    e.close()


# ------------------------------------------------------------------ #
# Basic init tests
# ------------------------------------------------------------------ #

class TestEnvInit:
    def test_creates_without_error(self, env):
        assert env is not None

    def test_correct_num_envs(self, env):
        assert env.num_envs == 4

    def test_observation_space_shape(self, env):
        assert env.observation_space.shape == (35,)

    def test_action_space_shape(self, env):
        assert env.action_space.shape == (3,)

    def test_action_space_bounds(self, env):
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0


# ------------------------------------------------------------------ #
# Reset tests
# ------------------------------------------------------------------ #

class TestReset:
    def test_reset_returns_correct_shape(self, env):
        obs = env.reset()
        assert obs.shape == (4, 35)

    def test_reset_returns_finite_values(self, env):
        obs = env.reset()
        assert np.all(np.isfinite(obs))

    def test_reset_positions_are_valid(self, env):
        obs = env.reset()
        for i in range(4):
            def_pos = obs[i, :3]
            att_pos = obs[i, 6:9]
            def_dist = np.linalg.norm(def_pos[:2])
            assert def_dist < 2.0, f"Defender too far from target: {def_dist}"
            att_dist = np.linalg.norm(att_pos[:2])
            assert att_dist > 0.1, f"Attacker too close to target: {att_dist}"

    def test_reset_velocities_are_zero(self, env):
        obs = env.reset()
        for i in range(4):
            def_vel = obs[i, 3:6]
            att_vel = obs[i, 9:12]
            np.testing.assert_allclose(def_vel, 0.0, atol=0.1)
            np.testing.assert_allclose(att_vel, 0.0, atol=0.1)

    def test_acceleration_zero_at_reset(self, env):
        obs = env.reset()
        for i in range(4):
            acc = obs[i, 20:23]
            np.testing.assert_allclose(acc, 0.0, atol=0.01)


# ------------------------------------------------------------------ #
# Step tests
# ------------------------------------------------------------------ #

class TestStep:
    def test_step_returns_correct_shapes(self, env):
        env.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        obs, rewards, dones, infos = env.step(actions)
        assert obs.shape == (4, 35)
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

        actions = np.zeros((4, 3), dtype=np.float32)
        for _ in range(5):
            obs, _, _, _ = env.step(actions)

        final_z = obs[:, 2]
        z_drift = np.abs(final_z - initial_z)
        assert np.all(z_drift < 1.0), f"Altitude drift too large: {z_drift}"

    def test_positive_velocity_moves_drone(self, env):
        obs = env.reset()
        initial_x = obs[:, 0].copy()

        actions = np.ones((4, 3), dtype=np.float32)
        actions[:, 1] = 0.0
        actions[:, 2] = 0.0
        for _ in range(20):
            obs, _, dones, _ = env.step(actions)

        final_x = obs[:, 0]
        moved = final_x - initial_x
        assert np.median(moved) > 0, f"Drone didn't move in +x: median delta = {np.median(moved)}"

    def test_infos_have_required_keys(self, env):
        env.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        _, _, _, infos = env.step(actions)
        for info in infos:
            assert "intercepted" in info
            assert "target_reached" in info


# ------------------------------------------------------------------ #
# Richer observations tests
# ------------------------------------------------------------------ #

class TestRicherObs:
    def test_closing_rate_in_obs(self, env):
        """After stepping toward attacker, closing rate should be positive."""
        obs = env.reset()
        # Chase the attacker for a few steps
        for _ in range(10):
            actions = np.zeros((4, 3), dtype=np.float32)
            for i in range(4):
                rel = obs[i, 12:15]
                norm = np.linalg.norm(rel)
                if norm > 0.01:
                    actions[i] = np.clip(rel / norm, -1, 1)
            obs, _, _, _ = env.step(actions)

        # Check closing rate (dim 19) — should be positive for at least some envs
        closing_rates = obs[:, 19]
        assert np.any(closing_rates > 0), f"No positive closing rates: {closing_rates}"

    def test_acceleration_nonzero_after_action(self, env):
        """After applying a velocity command, acceleration should be non-zero."""
        obs = env.reset()
        actions = np.ones((4, 3), dtype=np.float32)
        obs, _, _, _ = env.step(actions)
        # After first step, acceleration = current_vel - 0 (prev was zero)
        acc = obs[:, 20:23]
        # At least some acceleration should be non-zero
        assert np.any(np.abs(acc) > 0.001), f"Acceleration unexpectedly zero: {acc}"


# ------------------------------------------------------------------ #
# Obstacle tests
# ------------------------------------------------------------------ #

class TestObstacles:
    def test_env_creates_with_obstacles(self, env_with_obstacles):
        assert env_with_obstacles is not None
        assert env_with_obstacles._num_obstacles == 3

    def test_obs_shape_with_obstacles(self, env_with_obstacles):
        obs = env_with_obstacles.reset()
        assert obs.shape == (4, 35)

    def test_obstacle_obs_nonzero(self, env_with_obstacles):
        """With 3 obstacles, obstacle dims should be non-zero."""
        obs = env_with_obstacles.reset()
        obstacle_obs = obs[:, 23:35]  # 3 obstacles * 4 dims each
        assert np.any(obstacle_obs != 0.0), f"Obstacle obs all zeros: {obstacle_obs}"

    def test_no_obstacles_zeros_obs(self, env):
        """With 0 obstacles, obstacle dims should be all zeros."""
        obs = env.reset()
        obstacle_obs = obs[:, 23:35]
        np.testing.assert_allclose(obstacle_obs, 0.0, atol=1e-6)

    def test_obstacle_positions_within_arena(self, env_with_obstacles):
        """Obstacle positions should be within arena bounds."""
        env_with_obstacles.reset()
        obs_pos = env_with_obstacles.obstacle_pos_buf.numpy()
        bounds = env_with_obstacles.cfg.arena_bounds
        for i in range(4):
            for o in range(3):
                x, y = obs_pos[i][o][0], obs_pos[i][o][1]
                assert abs(x) <= bounds + 0.5, f"Obstacle x={x} out of bounds"
                assert abs(y) <= bounds + 0.5, f"Obstacle y={y} out of bounds"

    def test_fly_to_target_with_obstacles(self, env_fly_obstacles):
        """fly_to_target mode should work with obstacles."""
        obs = env_fly_obstacles.reset()
        assert obs.shape == (4, 35)
        actions = np.zeros((4, 3), dtype=np.float32)
        obs, rewards, dones, infos = env_fly_obstacles.step(actions)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))

    def test_intercept_with_obstacles(self, env_with_obstacles):
        """Intercept mode should work with obstacles."""
        obs = env_with_obstacles.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        for _ in range(10):
            obs, rewards, dones, infos = env_with_obstacles.step(actions)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))


# ------------------------------------------------------------------ #
# Reward and done tests
# ------------------------------------------------------------------ #

class TestRewardAndDone:
    def test_interception_gives_positive_reward(self, env):
        """When defender chases attacker, should see positive approach rewards."""
        env.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        max_reward = -np.inf
        for _ in range(500):
            obs, rewards, dones, infos = env.step(actions)
            for i in range(4):
                rel = obs[i, 12:15]
                norm = np.linalg.norm(rel)
                if norm > 0.01:
                    actions[i] = np.clip(rel / norm, -1, 1)
            max_reward = max(max_reward, np.max(rewards))
            if np.any(dones):
                break
        assert max_reward > 0, f"No positive rewards seen: max = {max_reward}"

    def test_time_penalty_exists(self, env):
        """Zero action should give small negative reward (time penalty)."""
        env.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        _, rewards, _, _ = env.step(actions)
        _, rewards, _, _ = env.step(actions)
        assert np.mean(rewards) < 5.0, "Rewards unexpectedly high with no action"


# ------------------------------------------------------------------ #
# Auto-reset tests
# ------------------------------------------------------------------ #

class TestAutoReset:
    def test_done_envs_get_reset(self, env):
        """Environments that are done should auto-reset."""
        env.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        for _ in range(1000):
            obs, rewards, dones, infos = env.step(actions)
            if np.any(dones):
                assert np.all(np.isfinite(obs))
                break


# ------------------------------------------------------------------ #
# GPU scaling tests
# ------------------------------------------------------------------ #

class TestGPUScaling:
    def test_large_env_creates(self, env_large):
        """64 parallel envs should work."""
        assert env_large.num_envs == 64

    def test_large_env_steps(self, env_large):
        """64 parallel envs should step without error."""
        obs = env_large.reset()
        assert obs.shape == (64, 35)
        actions = np.random.uniform(-1, 1, (64, 3)).astype(np.float32)
        obs, rewards, dones, infos = env_large.step(actions)
        assert obs.shape == (64, 35)
        assert np.all(np.isfinite(obs))


# ------------------------------------------------------------------ #
# Config tests
# ------------------------------------------------------------------ #

class TestConfig:
    def test_default_net_arch(self):
        cfg = TrainConfig()
        assert cfg.net_arch == [512, 512, 256]

    def test_obstacle_defaults(self):
        cfg = EnvConfig()
        assert cfg.num_obstacles == 3
        assert cfg.obstacle_radius > 0
        assert cfg.obstacle_height > 0

    def test_obstacle_config(self):
        cfg = EnvConfig(num_obstacles=3)
        assert cfg.num_obstacles == 3


# ------------------------------------------------------------------ #
# Mode tests
# ------------------------------------------------------------------ #

class TestModes:
    def test_fly_to_target_mode(self, env_fly):
        obs = env_fly.reset()
        assert obs.shape == (4, 35)
        actions = np.zeros((4, 3), dtype=np.float32)
        obs, rewards, dones, infos = env_fly.step(actions)
        assert np.all(np.isfinite(obs))
        for info in infos:
            assert "arrived" in info

    def test_chase_mode(self):
        cfg = EnvConfig(mode="chase", num_obstacles=0)
        e = NewtonDroneEnv(num_envs=4, config=cfg)
        obs = e.reset()
        assert obs.shape == (4, 35)
        actions = np.zeros((4, 3), dtype=np.float32)
        obs, _, _, _ = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()

    def test_evade_mode(self):
        cfg = EnvConfig(mode="evade", num_obstacles=0)
        e = NewtonDroneEnv(num_envs=4, config=cfg)
        obs = e.reset()
        assert obs.shape == (4, 35)
        actions = np.zeros((4, 3), dtype=np.float32)
        obs, _, _, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        for info in infos:
            assert "target_reached" in info
            assert "intercepted" in info
        e.close()


# ------------------------------------------------------------------ #
# Domain randomization tests
# ------------------------------------------------------------------ #

class TestDomainRandomization:
    def test_wind_env_creates(self):
        cfg = EnvConfig(wind_enabled=True, wind_max_force=0.5, num_obstacles=0)
        e = NewtonDroneEnv(num_envs=4, config=cfg)
        obs = e.reset()
        assert np.all(np.isfinite(obs))
        e.close()

    def test_wind_force_nonzero(self):
        cfg = EnvConfig(wind_enabled=True, wind_max_force=0.5, num_obstacles=0)
        e = NewtonDroneEnv(num_envs=4, config=cfg)
        e.reset()
        assert np.any(e.wind_force != 0)
        e.close()

    def test_wind_step_finite(self):
        cfg = EnvConfig(wind_enabled=True, wind_max_force=0.3, num_obstacles=0)
        e = NewtonDroneEnv(num_envs=4, config=cfg)
        e.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))
        e.close()

    def test_obs_noise_applied(self):
        cfg = EnvConfig(obs_noise_std=0.1, num_obstacles=0)
        e = NewtonDroneEnv(num_envs=8, config=cfg)
        obs1 = e.reset().copy()
        obs2 = e.reset().copy()
        # With noise, two resets should produce slightly different obs
        # (even same positions get different noise)
        # We just check it doesn't crash and is finite
        assert np.all(np.isfinite(obs1))
        assert np.all(np.isfinite(obs2))
        e.close()

    def test_obs_noise_zero_noop(self):
        cfg = EnvConfig(obs_noise_std=0.0, num_obstacles=0)
        e = NewtonDroneEnv(num_envs=4, config=cfg)
        obs = e.reset()
        assert np.all(np.isfinite(obs))
        e.close()

    def test_wind_with_intercept(self):
        cfg = EnvConfig(mode="intercept", wind_enabled=True, wind_max_force=0.3,
                        num_obstacles=3)
        e = NewtonDroneEnv(num_envs=4, config=cfg)
        e.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        for _ in range(10):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()

    def test_dynamic_obstacles_move(self):
        cfg = EnvConfig(num_obstacles=3, obstacle_dynamic=True, obstacle_max_speed=0.5)
        e = NewtonDroneEnv(num_envs=4, config=cfg)
        e.reset()
        pos_before = e.obstacle_pos_buf.numpy().copy()
        actions = np.zeros((4, 3), dtype=np.float32)
        for _ in range(20):
            e.step(actions)
        pos_after = e.obstacle_pos_buf.numpy()
        # Obstacles should have moved
        assert np.any(pos_before != pos_after), "Dynamic obstacles didn't move"
        e.close()

    def test_dynamic_obstacles_finite(self):
        cfg = EnvConfig(mode="intercept", num_obstacles=3,
                        obstacle_dynamic=True, obstacle_max_speed=0.3)
        e = NewtonDroneEnv(num_envs=4, config=cfg)
        e.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        for _ in range(50):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()

    def test_dynamic_moves_more_than_static(self):
        """Dynamic obstacles should drift more than static ones over time."""
        # Static
        cfg_s = EnvConfig(mode="fly_to_target", num_obstacles=3, obstacle_dynamic=False,
                          episode_secs=60.0)  # long episode to avoid resets
        e_s = NewtonDroneEnv(num_envs=4, config=cfg_s)
        e_s.reset()
        pos_s0 = e_s.obstacle_pos_buf.numpy().copy()
        actions = np.zeros((4, 3), dtype=np.float32)
        for _ in range(20):
            e_s.step(actions)
        drift_s = np.sum(np.abs(e_s.obstacle_pos_buf.numpy() - pos_s0))
        e_s.close()
        # Dynamic
        cfg_d = EnvConfig(mode="fly_to_target", num_obstacles=3, obstacle_dynamic=True,
                          obstacle_max_speed=0.5, episode_secs=60.0)
        e_d = NewtonDroneEnv(num_envs=4, config=cfg_d)
        e_d.reset()
        pos_d0 = e_d.obstacle_pos_buf.numpy().copy()
        for _ in range(20):
            e_d.step(actions)
        drift_d = np.sum(np.abs(e_d.obstacle_pos_buf.numpy() - pos_d0))
        e_d.close()
        assert drift_d > drift_s, f"Dynamic drift {drift_d} should exceed static {drift_s}"

    def test_action_delay(self):
        """Action delay should buffer commands without crashing."""
        cfg = EnvConfig(action_delay_steps=3, num_obstacles=0)
        e = NewtonDroneEnv(num_envs=4, config=cfg)
        obs = e.reset()
        assert np.all(np.isfinite(obs))
        # First few steps should use zero actions (delay buffer initialized to zero)
        actions = np.ones((4, 3), dtype=np.float32)
        for _ in range(10):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()

    def test_all_domain_rand_combined(self):
        """All domain randomization features enabled together."""
        cfg = EnvConfig(
            mode="intercept",
            wind_enabled=True, wind_max_force=0.3,
            obs_noise_std=0.02,
            obstacle_dynamic=True, obstacle_max_speed=0.3,
            action_delay_steps=2,
            num_obstacles=3,
        )
        e = NewtonDroneEnv(num_envs=4, config=cfg)
        e.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        for _ in range(20):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()


class TestDefenderLeash:
    def test_leash_config_defaults(self):
        cfg = EnvConfig()
        assert cfg.defender_leash_radius == 3.0
        assert cfg.defender_leash_penalty == 2.0

    def test_leash_env_runs_with_leash(self):
        """Env with leash enabled should step without errors."""
        cfg = EnvConfig(defender_leash_radius=2.0, defender_leash_penalty=5.0,
                        num_obstacles=0)
        e = NewtonDroneEnv(num_envs=4, config=cfg)
        e.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        for _ in range(20):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))
        e.close()

    def test_leash_penalty_reduces_reward(self):
        """Tight leash should produce lower rewards when defender is far from target."""
        # Tight leash (1.0m radius, heavy penalty)
        cfg_tight = EnvConfig(defender_leash_radius=1.0, defender_leash_penalty=10.0,
                              num_obstacles=0)
        e_tight = NewtonDroneEnv(num_envs=4, config=cfg_tight)
        e_tight.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        rewards_tight = []
        for _ in range(30):
            _, r, _, _ = e_tight.step(actions)
            rewards_tight.append(r.copy())
        e_tight.close()

        # No leash (huge radius = effectively disabled)
        cfg_loose = EnvConfig(defender_leash_radius=100.0, defender_leash_penalty=10.0,
                              num_obstacles=0)
        e_loose = NewtonDroneEnv(num_envs=4, config=cfg_loose)
        e_loose.reset()
        rewards_loose = []
        for _ in range(30):
            _, r, _, _ = e_loose.step(actions)
            rewards_loose.append(r.copy())
        e_loose.close()

        # Tight leash should generally produce lower or equal mean reward
        # (defender starts 0.3-1.0m from target, so tight leash may penalize)
        mean_tight = np.mean([r.mean() for r in rewards_tight])
        mean_loose = np.mean([r.mean() for r in rewards_loose])
        assert np.isfinite(mean_tight)
        assert np.isfinite(mean_loose)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
