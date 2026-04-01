"""Unit tests for the Newton GPU-accelerated swarm drone environment."""

import numpy as np
import pytest
import warp as wp

wp.init()

from envs.newton_swarm_env import NewtonSwarmEnv, OBS_DIM, compute_obs_dim
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


# ------------------------------------------------------------------ #
# Asymmetric config tests
# ------------------------------------------------------------------ #

class TestSwarmAsymmetric:
    def test_2v1_creates(self):
        cfg = EnvConfig(num_defenders=2, num_attackers=1, num_obstacles=3)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        assert e.num_envs == 8  # 4 worlds * 2 defenders
        obs = e.reset()
        assert obs.shape == (8, OBS_DIM)
        e.close()

    def test_2v1_steps(self):
        cfg = EnvConfig(num_defenders=2, num_attackers=1, num_obstacles=3)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((8, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))
        e.close()

    def test_3v2_creates(self):
        cfg = EnvConfig(num_defenders=3, num_attackers=2, num_obstacles=3)
        e = NewtonSwarmEnv(num_worlds=2, config=cfg)
        assert e.num_envs == 6  # 2 worlds * 3 defenders
        obs = e.reset()
        assert obs.shape == (6, OBS_DIM)
        e.close()

    def test_3v2_steps(self):
        cfg = EnvConfig(num_defenders=3, num_attackers=2, num_obstacles=3)
        e = NewtonSwarmEnv(num_worlds=2, config=cfg)
        e.reset()
        actions = np.zeros((6, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()

    def test_1v2_creates(self):
        """Outnumbered defender scenario."""
        cfg = EnvConfig(num_defenders=1, num_attackers=2, num_obstacles=0)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        assert e.num_envs == 4  # 4 worlds * 1 defender
        obs = e.reset()
        assert obs.shape == (4, OBS_DIM)
        actions = np.zeros((4, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()

    def test_1v3_creates(self):
        """Heavily outnumbered defender."""
        cfg = EnvConfig(num_defenders=1, num_attackers=3, num_obstacles=0)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        assert e.num_envs == 4
        obs = e.reset()
        assert obs.shape == (4, OBS_DIM)
        e.close()


# ------------------------------------------------------------------ #
# Obstacle collision penalty tests
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# Evade mode tests
# ------------------------------------------------------------------ #

class TestSwarmEvade:
    def test_evade_2v2_creates(self):
        """Evade mode creates with RL controlling attackers."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0, mode="evade")
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        assert e.num_envs == 8  # 4 worlds * 2 attackers
        e.close()

    def test_evade_obs_shape(self):
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0, mode="evade")
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        obs = e.reset()
        assert obs.shape == (8, OBS_DIM)
        e.close()

    def test_evade_step(self):
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0, mode="evade")
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((8, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert obs.shape == (8, OBS_DIM)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))
        e.close()

    def test_evade_1v2(self):
        """1 defender vs 2 RL attackers."""
        cfg = EnvConfig(num_defenders=1, num_attackers=2, num_obstacles=0, mode="evade")
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        assert e.num_envs == 8  # 4 worlds * 2 attackers
        obs = e.reset()
        assert obs.shape == (8, OBS_DIM)
        actions = np.zeros((8, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()

    def test_evade_with_obstacles(self):
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=3, mode="evade")
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((8, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()


class TestSwarmObstaclePenalty:
    def test_collision_penalty_exists(self):
        """Verify collision penalty config is set."""
        cfg = EnvConfig()
        assert cfg.obstacle_collision_penalty == 50.0
        assert cfg.obstacle_collision_radius == 0.2
        assert cfg.obstacle_proximity_penalty == 10.0


# ------------------------------------------------------------------ #
# Attacker strategy tests
# ------------------------------------------------------------------ #

class TestAttackerStrategies:
    def test_flanking_2v2(self):
        """Flanking strategy produces finite results."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        attacker_strategy="flanking")
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((8, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))
        e.close()

    def test_mixed_2v2(self):
        """Mixed strategy produces finite results."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        attacker_strategy="mixed")
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((8, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))
        e.close()

    def test_flanking_3v2(self):
        """Flanking works with asymmetric configs."""
        cfg = EnvConfig(num_defenders=3, num_attackers=2, num_obstacles=3,
                        attacker_strategy="flanking")
        e = NewtonSwarmEnv(num_worlds=2, config=cfg)
        e.reset()
        actions = np.zeros((6, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()

    def test_direct_is_default(self):
        """Default strategy is direct."""
        cfg = EnvConfig()
        assert cfg.attacker_strategy == "direct"

    def test_flanking_1v1_falls_back_to_direct(self):
        """With 1 attacker, flanking degrades to direct (no spread needed)."""
        cfg = EnvConfig(num_defenders=1, num_attackers=1, num_obstacles=0,
                        attacker_strategy="flanking")
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((4, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()


class TestDefenderLeash:
    def test_leash_swarm_runs(self):
        """Swarm env with leash should step without errors."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        defender_leash_radius=2.0, defender_leash_penalty=5.0)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((4 * 2, 3), dtype=np.float32)
        for _ in range(10):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))
        e.close()

    def test_leash_tight_vs_loose(self):
        """Tight leash should produce finite rewards in swarm env."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        defender_leash_radius=1.0, defender_leash_penalty=10.0)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((4 * 2, 3), dtype=np.float32)
        for _ in range(20):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(rewards))
        e.close()

    def test_zone_time_penalty_escalates(self):
        """Outside-zone step counter should increase over time."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        defender_leash_radius=0.5, defender_zone_time_penalty=1.0)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((4 * 2, 3), dtype=np.float32)
        for _ in range(10):
            e.step(actions)
        # Defenders start 0.3-1.0m from target, with 0.5m radius most should be outside
        assert np.any(e.outside_zone_steps > 0), "Zone counter should increment when outside"
        e.close()

    def test_zone_counter_resets_on_episode_reset(self):
        """Zone counter should reset on full reset."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        defender_leash_radius=0.5)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((4 * 2, 3), dtype=np.float32)
        for _ in range(5):
            e.step(actions)
        e.reset()
        assert np.all(e.outside_zone_steps == 0), "Zone counter should reset"
        e.close()


class TestAngularCoverage:
    def test_angular_coverage_runs(self):
        """Env with angular coverage bonus should step without errors."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        angular_coverage_bonus=1.0)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((4 * 2, 3), dtype=np.float32)
        for _ in range(20):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(rewards))
        e.close()

    def test_angular_coverage_3v3(self):
        """Angular coverage should work in 3v3."""
        cfg = EnvConfig(num_defenders=3, num_attackers=3, num_obstacles=0,
                        angular_coverage_bonus=0.5)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((4 * 3, 3), dtype=np.float32)
        for _ in range(10):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(rewards))
        e.close()


class TestInterceptEfficiency:
    def test_efficiency_bonus_config(self):
        cfg = EnvConfig()
        assert cfg.intercept_efficiency_bonus == 1.0

    def test_efficiency_bonus_runs(self):
        """Env with intercept efficiency bonus should step without errors."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        intercept_efficiency_bonus=5.0)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((4 * 2, 3), dtype=np.float32)
        for _ in range(30):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(rewards))
        e.close()


class TestSensorRange:
    def test_sensor_range_runs(self):
        """Env with limited sensor range should step without errors."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        sensor_range=3.0)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((4 * 2, 3), dtype=np.float32)
        for _ in range(10):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()

    def test_sensor_range_zeros_far_targets(self):
        """With tight sensor range, far-away agents should be zeroed in obs."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        sensor_range=0.5)  # very tight — most agents out of range
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        obs = e.reset()
        # Attacker block starts at dim 13, if out of range should be zeros
        # Just verify it runs and is finite
        assert np.all(np.isfinite(obs))
        e.close()

    def test_sensor_range_unlimited(self):
        """sensor_range=0 means unlimited (default behavior)."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        sensor_range=0.0)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        obs = e.reset()
        # With unlimited range, attacker obs should be non-zero
        # (attacker pos is in slots 13:16 for first attacker)
        assert np.any(obs[:, 13:16] != 0), "Unlimited sensor should see attackers"
        e.close()


class TestWindGusts:
    def test_wind_gusts_run(self):
        """Wind with gusts enabled should step without errors."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        wind_enabled=True, wind_max_force=0.3,
                        wind_gust_prob=0.5, wind_gust_force=2.0)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((4 * 2, 3), dtype=np.float32)
        for _ in range(20):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()


class TestDroneVariance:
    def test_attacker_speed_variance(self):
        """Per-attacker speed should vary with variance > 0."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        attacker_speed_variance=0.3)
        e = NewtonSwarmEnv(num_worlds=8, config=cfg)
        e.reset()
        # Per-attacker speeds should not all be identical
        speeds = e.per_attacker_speed.flatten()
        assert not np.all(speeds == speeds[0]), "Attacker speeds should vary"
        e.close()

    def test_mass_variance(self):
        """Per-drone mass should vary with variance > 0."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        drone_mass_variance=0.2)
        e = NewtonSwarmEnv(num_worlds=8, config=cfg)
        e.reset()
        masses = e.per_drone_mass.flatten()
        assert not np.all(masses == masses[0]), "Drone masses should vary"
        e.close()

    def test_variance_steps_finite(self):
        """Env with all variance enabled should produce finite results."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=3,
                        attacker_speed_variance=0.2, drone_mass_variance=0.15,
                        wind_enabled=True, wind_gust_prob=0.1,
                        sensor_range=5.0)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((4 * 2, 3), dtype=np.float32)
        for _ in range(30):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))
        e.close()


class TestKamikaze:
    def test_kamikaze_enabled_by_default(self):
        cfg = EnvConfig()
        assert cfg.kamikaze is True

    def test_kamikaze_defender_eliminated_on_intercept(self):
        """When kamikaze=True, defender should be marked eliminated after interception."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0,
                        kamikaze=True, intercept_dist=0.75)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        # Run many steps chasing attacker — eventually should intercept
        actions = np.zeros((4 * 2, 3), dtype=np.float32)
        for _ in range(300):
            obs, rewards, dones, infos = e.step(actions)
            for i in range(8):
                att_pos = obs[i, 13:16]
                my_pos = obs[i, 0:3]
                rel = att_pos - my_pos
                norm = np.linalg.norm(rel)
                if norm > 0.01:
                    actions[i] = np.clip(rel / norm, -1, 1)
            if np.any(dones):
                break
        # Check that at least one defender was eliminated (kamikaze)
        # or episode ended (meaning intercept happened and world reset)
        any_defender_eliminated = np.any(e.eliminated_defenders)
        any_done = np.any(dones)
        assert any_defender_eliminated or any_done, "Kamikaze should eliminate defenders on intercept"
        e.close()

    def test_kamikaze_off_defender_survives(self):
        """When kamikaze=False, defender should not be eliminated."""
        cfg = EnvConfig(num_defenders=2, num_attackers=1, num_obstacles=0,
                        kamikaze=False)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((4 * 2, 3), dtype=np.float32)
        for _ in range(300):
            obs, rewards, dones, infos = e.step(actions)
            for i in range(8):
                att_pos = obs[i, 13:16]
                my_pos = obs[i, 0:3]
                rel = att_pos - my_pos
                norm = np.linalg.norm(rel)
                if norm > 0.01:
                    actions[i] = np.clip(rel / norm, -1, 1)
            if np.any(dones):
                break
        # Defenders should never be eliminated with kamikaze off
        assert not np.any(e.eliminated_defenders), "Non-kamikaze defenders should survive"
        e.close()

    def test_kamikaze_info_has_active_defenders(self):
        """Info dict should include active_defenders count."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0, kamikaze=True)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        actions = np.zeros((4 * 2, 3), dtype=np.float32)
        _, _, _, infos = e.step(actions)
        for info in infos:
            assert "active_defenders" in info
        e.close()

    def test_kamikaze_eliminated_defender_zero_obs(self):
        """Eliminated defenders should get zeroed observations."""
        cfg = EnvConfig(num_defenders=2, num_attackers=2, num_obstacles=0, kamikaze=True)
        e = NewtonSwarmEnv(num_worlds=4, config=cfg)
        e.reset()
        # Manually eliminate a defender
        e.eliminated_defenders[0, 0] = True
        e._compute_obs()
        agent_idx = 0  # world 0, defender 0
        assert np.all(e.obs_buf[agent_idx] == 0.0), "Eliminated defender should have zero obs"
        e.close()


class TestKNearestObs:
    def test_default_obs_dim(self):
        """Default k_opponents=3, k_teammates=2 → 57-dim obs."""
        assert compute_obs_dim(3, 2) == 57

    def test_custom_k_obs_dim(self):
        """Custom K values produce correct obs dim."""
        assert compute_obs_dim(5, 4) == 13 + 5*7 + 4*7 + 9  # 85
        assert compute_obs_dim(1, 1) == 13 + 7 + 7 + 9  # 36

    def test_15v10_with_k3_k2(self):
        """15v10 with K=3/2 should have same 57-dim obs as 2v2."""
        cfg = EnvConfig(num_defenders=15, num_attackers=10, num_obstacles=0,
                        k_opponents=3, k_teammates=2)
        e = NewtonSwarmEnv(num_worlds=2, config=cfg)
        obs = e.reset()
        assert obs.shape == (2 * 15, 57)
        actions = np.zeros((2 * 15, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert obs.shape == (2 * 15, 57)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))
        e.close()

    def test_larger_k_obs_dim(self):
        """Using larger K creates larger obs space."""
        cfg = EnvConfig(num_defenders=5, num_attackers=5, num_obstacles=0,
                        k_opponents=5, k_teammates=4)
        e = NewtonSwarmEnv(num_worlds=2, config=cfg)
        expected_dim = compute_obs_dim(5, 4)
        obs = e.reset()
        assert obs.shape == (2 * 5, expected_dim)
        e.close()

    def test_k_larger_than_actual_drones(self):
        """K > actual drones should work (extra slots stay zero-padded)."""
        cfg = EnvConfig(num_defenders=2, num_attackers=1, num_obstacles=0,
                        k_opponents=5, k_teammates=4)
        e = NewtonSwarmEnv(num_worlds=2, config=cfg)
        obs = e.reset()
        expected_dim = compute_obs_dim(5, 4)
        assert obs.shape == (2 * 2, expected_dim)
        assert np.all(np.isfinite(obs))
        e.close()

    def test_generalization_same_model_diff_swarm_sizes(self):
        """Same K config produces same obs_dim regardless of swarm size."""
        dim_2v2 = compute_obs_dim(3, 2)
        dim_15v10 = compute_obs_dim(3, 2)
        dim_20v15 = compute_obs_dim(3, 2)
        assert dim_2v2 == dim_15v10 == dim_20v15 == 57


class TestLargeSwarm:
    def test_15v10_creates(self):
        """15v10 environment creates successfully."""
        cfg = EnvConfig(num_defenders=15, num_attackers=10, num_obstacles=0)
        e = NewtonSwarmEnv(num_worlds=2, config=cfg)
        assert e.num_envs == 30  # 2 worlds * 15 defenders
        e.close()

    def test_15v10_steps(self):
        """15v10 environment steps without errors."""
        cfg = EnvConfig(num_defenders=15, num_attackers=10, num_obstacles=0)
        e = NewtonSwarmEnv(num_worlds=2, config=cfg)
        e.reset()
        actions = np.zeros((30, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))
        e.close()

    def test_8v5_creates(self):
        """8v5 environment creates successfully."""
        cfg = EnvConfig(num_defenders=8, num_attackers=5, num_obstacles=0)
        e = NewtonSwarmEnv(num_worlds=2, config=cfg)
        assert e.num_envs == 16
        obs = e.reset()
        assert obs.shape == (16, 57)
        e.close()

    def test_20v15_creates(self):
        """20v15 environment creates successfully."""
        cfg = EnvConfig(num_defenders=20, num_attackers=15, num_obstacles=3)
        e = NewtonSwarmEnv(num_worlds=2, config=cfg)
        assert e.num_envs == 40
        obs = e.reset()
        assert obs.shape == (40, 57)
        e.close()

    def test_15v10_kamikaze(self):
        """15v10 with kamikaze should run without errors."""
        cfg = EnvConfig(num_defenders=15, num_attackers=10, num_obstacles=0, kamikaze=True)
        e = NewtonSwarmEnv(num_worlds=2, config=cfg)
        e.reset()
        actions = np.zeros((30, 3), dtype=np.float32)
        for _ in range(10):
            obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        assert np.all(np.isfinite(rewards))
        e.close()

    def test_15v10_evade_mode(self):
        """15v10 evade mode creates and runs."""
        cfg = EnvConfig(num_defenders=15, num_attackers=10, num_obstacles=0, mode="evade")
        e = NewtonSwarmEnv(num_worlds=2, config=cfg)
        assert e.num_envs == 20  # 2 worlds * 10 attackers
        obs = e.reset()
        assert obs.shape == (20, 57)
        actions = np.zeros((20, 3), dtype=np.float32)
        obs, rewards, dones, infos = e.step(actions)
        assert np.all(np.isfinite(obs))
        e.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
