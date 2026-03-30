"""Unit tests for the Newton GPU-accelerated swarm drone environment."""

import numpy as np
import pytest
import warp as wp

wp.init()

from envs.newton_swarm_env import NewtonSwarmEnv, OBS_DIM
from train.config import EnvConfig


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def env_2v2():
    """Create a 2v2 swarm environment."""
    cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0)
    e = NewtonSwarmEnv(num_worlds=4, config=cfg)
    yield e
    e.close()


@pytest.fixture
def env_2v2_obstacles():
    """Create a 2v2 swarm environment with obstacles."""
    cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=3)
    e = NewtonSwarmEnv(num_worlds=4, config=cfg)
    yield e
    e.close()


@pytest.fixture
def env_3v3():
    """Create a 3v3 swarm environment."""
    cfg = EnvConfig(num_defenders=3, num_attackers=3, num_obstacles=0)
    e = NewtonSwarmEnv(num_worlds=2, config=cfg)
    yield e
    e.close()


@pytest.fixture
def env_1v1():
    """Create a 1v1 swarm environment (should work like original)."""
    cfg = EnvConfig(num_defenders=1, num_attackers=1, num_obstacles=0)
    e = NewtonSwarmEnv(num_worlds=4, config=cfg)
    yield e
    e.close()


@pytest.fixture
def env_wind():
    """Create env with wind domain randomization."""
    cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                    wind_enabled=True, wind_max_force=0.5)
    e = NewtonSwarmEnv(num_worlds=4, config=cfg)
    yield e
    e.close()


# ------------------------------------------------------------------ #
# Init tests
# ------------------------------------------------------------------ #

class TestSwarmInit:
    def test_2v2_creates(self, env_2v2):
        assert env_2v2 is not None

    def test_2v2_num_envs(self, env_2v2):
        # 4 worlds * 2 defenders = 8
        assert env_2v2.num_envs == 8

    def test_3v3_num_envs(self, env_3v3):
        # 2 worlds * 3 defenders = 6
        assert env_3v3.num_envs == 6

    def test_obs_shape(self, env_2v2):
        assert env_2v2.observation_space.shape == (OBS_DIM,)

    def test_action_shape(self, env_2v2):
        assert env_2v2.action_space.shape == (3,)

    def test_bodies_per_world_2v2(self, env_2v2):
        assert env_2v2.bodies_per_world == 4  # 2 def + 2 att + 0 obs

    def test_bodies_per_world_2v2_obstacles(self, env_2v2_obstacles):
        assert env_2v2_obstacles.bodies_per_world == 7  # 2+2+3


# ------------------------------------------------------------------ #
# Reset tests
# ------------------------------------------------------------------ #

class TestSwarmReset:
    def test_reset_shape_2v2(self, env_2v2):
        obs = env_2v2.reset()
        assert obs.shape == (8, OBS_DIM)

    def test_reset_shape_3v3(self, env_3v3):
        obs = env_3v3.reset()
        assert obs.shape == (6, OBS_DIM)

    def test_reset_finite(self, env_2v2):
        obs = env_2v2.reset()
        assert np.all(np.isfinite(obs))

    def test_reset_clears_eliminated(self, env_2v2):
        env_2v2.reset()
        assert not np.any(env_2v2.eliminated_attackers)


# ------------------------------------------------------------------ #
# Step tests
# ------------------------------------------------------------------ #

class TestSwarmStep:
    def test_step_shapes(self, env_2v2):
        env_2v2.reset()
        actions = np.zeros((8, 3), dtype=np.float32)
        obs, rewards, dones, infos = env_2v2.step(actions)
        assert obs.shape == (8, OBS_DIM)
        assert rewards.shape == (8,)
        assert dones.shape == (8,)
        assert len(infos) == 8

    def test_step_finite(self, env_2v2):
        env_2v2.reset()
        actions = np.zeros((8, 3), dtype=np.float32)
        obs, rewards, dones, infos = env_2v2.step(actions)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))

    def test_step_3v3(self, env_3v3):
        env_3v3.reset()
        actions = np.zeros((6, 3), dtype=np.float32)
        obs, rewards, dones, infos = env_3v3.step(actions)
        assert obs.shape == (6, OBS_DIM)

    def test_info_keys(self, env_2v2):
        env_2v2.reset()
        actions = np.zeros((8, 3), dtype=np.float32)
        _, _, _, infos = env_2v2.step(actions)
        for info in infos:
            assert "intercepted" in info
            assert "target_reached" in info
            assert "active_attackers" in info

    def test_1v1_works(self, env_1v1):
        obs = env_1v1.reset()
        assert obs.shape == (4, OBS_DIM)
        actions = np.zeros((4, 3), dtype=np.float32)
        obs, rewards, dones, infos = env_1v1.step(actions)
        assert obs.shape == (4, OBS_DIM)


# ------------------------------------------------------------------ #
# Interception tests
# ------------------------------------------------------------------ #

class TestSwarmIntercept:
    def test_chase_produces_positive_rewards(self, env_2v2):
        """Defenders chasing attackers should get positive approach rewards."""
        obs = env_2v2.reset()
        max_reward = -np.inf
        actions = np.zeros((8, 3), dtype=np.float32)
        for _ in range(200):
            obs, rewards, dones, infos = env_2v2.step(actions)
            # Simple chase: move toward nearest attacker info
            for i in range(8):
                # Attacker 0 position is at obs[13:16]
                att_pos = obs[i, 13:16]
                my_pos = obs[i, 0:3]
                rel = att_pos - my_pos
                norm = np.linalg.norm(rel)
                if norm > 0.01:
                    actions[i] = np.clip(rel / norm, -1, 1)
            max_reward = max(max_reward, np.max(rewards))
            if np.any(dones):
                break
        assert max_reward > 0, f"No positive rewards: {max_reward}"


# ------------------------------------------------------------------ #
# Domain randomization tests
# ------------------------------------------------------------------ #

class TestDomainRandom:
    def test_wind_env_steps(self, env_wind):
        obs = env_wind.reset()
        actions = np.zeros((8, 3), dtype=np.float32)
        obs, rewards, dones, infos = env_wind.step(actions)
        assert np.all(np.isfinite(obs))

    def test_wind_force_nonzero(self, env_wind):
        env_wind.reset()
        assert np.any(env_wind.wind_force != 0)


# ------------------------------------------------------------------ #
# Obstacle tests
# ------------------------------------------------------------------ #

class TestSwarmObstacles:
    def test_obstacles_present(self, env_2v2_obstacles):
        obs = env_2v2_obstacles.reset()
        # Obstacle dims [48:57] should have non-zero values
        obstacle_obs = obs[:, 48:57]
        assert np.any(obstacle_obs != 0.0)

    def test_step_with_obstacles(self, env_2v2_obstacles):
        env_2v2_obstacles.reset()
        actions = np.zeros((8, 3), dtype=np.float32)
        obs, rewards, dones, infos = env_2v2_obstacles.step(actions)
        assert np.all(np.isfinite(obs))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
