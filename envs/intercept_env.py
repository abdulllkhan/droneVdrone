"""
Drone Interception Environment

A defender drone (RL agent) must intercept an attacker drone that is
flying toward a target position. The defender succeeds by colliding
with the attacker before it reaches the target.

Inherits from gym-pybullet-drones BaseAviary for realistic 3D
quadrotor physics and wraps it as a single-agent Gymnasium env
compatible with Stable Baselines3.
"""

import numpy as np
import pybullet as p
from gymnasium import spaces

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from train.config import EnvConfig


class DroneInterceptEnv(BaseAviary):
    """
    Single-agent RL environment for drone interception.

    Drone 0 = Defender (RL agent)
    Drone 1 = Attacker (scripted PID, flies toward target)

    Observation (22-dim):
        [0:3]   defender position (x, y, z)
        [3:6]   defender velocity (vx, vy, vz)
        [6:9]   defender orientation (roll, pitch, yaw)
        [9:12]  attacker position (x, y, z)
        [12:15] attacker velocity (vx, vy, vz)
        [15:18] relative position (attacker - defender)
        [18]    distance to attacker
        [19:22] target position (x, y, z)

    Action (4-dim):
        Motor commands in [-1, 1], mapped to [0, MAX_RPM] internally.
    """

    OBS_DIM = 22

    def __init__(self, gui=False, config=None):
        self.cfg = config or EnvConfig()
        self.target_pos = np.array(self.cfg.target_pos, dtype=np.float32)

        # Pre-compute for parent init
        init_xyzs = self._random_positions()

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=2,
            initial_xyzs=init_xyzs,
            initial_rpys=np.zeros((2, 3)),
            physics=Physics.PYB_GND_DRAG_DW,
            pyb_freq=self.cfg.pyb_freq,
            ctrl_freq=self.cfg.ctrl_freq,
            gui=gui,
            record=False,
            obstacles=False,
            user_debug_gui=False,
        )

        # PID controller for the scripted attacker
        self.attacker_ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

        # Tracking
        self.prev_dist = None
        # BaseAviary tracks self.step_counter (incremented by PYB_STEPS_PER_CTRL)
        # Max step_counter value for episode timeout:
        self.max_step_counter = int(
            self.cfg.episode_secs * self.cfg.pyb_freq
        )

    # ------------------------------------------------------------------
    # Gymnasium spaces (called by BaseAviary.__init__)
    # ------------------------------------------------------------------

    def _actionSpace(self):
        """Single-agent: 4 motor commands in [-1, 1], mapped to RPMs."""
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        """22-dim observation vector."""
        lo = np.full(self.OBS_DIM, -np.inf, dtype=np.float32)
        hi = np.full(self.OBS_DIM, np.inf, dtype=np.float32)
        return spaces.Box(low=lo, high=hi, dtype=np.float32)

    # ------------------------------------------------------------------
    # Core Gymnasium interface
    # ------------------------------------------------------------------

    def step(self, action):
        """
        Step the environment with the defender's action.

        Args:
            action: (4,) array in [-1, 1] — defender motor commands.
        """
        # Build (2, 4) action array expected by BaseAviary
        full_action = np.zeros((2, 4), dtype=np.float32)
        full_action[0] = np.asarray(action, dtype=np.float32)
        # Attacker slot is a placeholder — _preprocessAction computes PID
        full_action[1] = 0.5

        # BaseAviary.step() calls _preprocessAction (ours), runs physics,
        # then calls our _computeObs/Reward/Terminated/Truncated/Info
        return super().step(full_action)

    def reset(self, seed=None, options=None):
        """Reset with randomized initial positions."""
        self.INIT_XYZS = self._random_positions()
        self.prev_dist = None

        # Re-create PID controller to clear integral state
        self.attacker_ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

        # super().reset() calls _housekeeping (resets step_counter),
        # then calls _computeObs() and _computeInfo()
        return super().reset(seed=seed, options=options)

    # ------------------------------------------------------------------
    # Action preprocessing
    # ------------------------------------------------------------------

    def _preprocessAction(self, action):
        """
        Convert RL action to RPMs for both drones.

        Args:
            action: (2, 4) array — row 0 is defender [-1,1], row 1 is ignored.

        Returns:
            (2, 4) RPM array.
        """
        rpms = np.zeros((2, 4))

        # Defender: map [-1, 1] → [0, MAX_RPM]
        rpms[0] = np.clip((action[0] + 1.0) / 2.0 * self.MAX_RPM, 0, self.MAX_RPM)

        # Attacker: PID control toward target
        rpms[1] = self._compute_attacker_rpms()

        return rpms

    def _compute_attacker_rpms(self):
        """Compute attacker RPMs using PID control toward target."""
        state = self._getDroneStateVector(1)
        rpms, _, _ = self.attacker_ctrl.computeControlFromState(
            control_timestep=self.CTRL_TIMESTEP,
            state=state,
            target_pos=self.target_pos,
            target_rpy=np.zeros(3),
        )
        return rpms * self.cfg.attacker_speed_scale

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _computeObs(self):
        """Build 22-dim observation vector."""
        defender = self._getDroneStateVector(0)
        attacker = self._getDroneStateVector(1)

        def_pos = defender[0:3]
        def_vel = defender[10:13]
        def_rpy = defender[7:10]
        att_pos = attacker[0:3]
        att_vel = attacker[10:13]

        rel_pos = att_pos - def_pos
        dist = np.array([np.linalg.norm(rel_pos)])

        obs = np.concatenate([
            def_pos,            # 3
            def_vel,            # 3
            def_rpy,            # 3
            att_pos,            # 3
            att_vel,            # 3
            rel_pos,            # 3
            dist,               # 1
            self.target_pos,    # 3
        ])  # Total: 22

        return obs.astype(np.float32)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _computeReward(self):
        """
        Dense reward function:
        - Approach reward: positive for closing distance to attacker
        - Interception bonus: +100 on collision
        - Failure penalty: -100 if attacker reaches target
        - Time penalty: -0.01 per step to encourage speed
        """
        def_pos = self.pos[0]
        att_pos = self.pos[1]

        dist = np.linalg.norm(att_pos - def_pos)

        # Interception bonus
        if dist < self.cfg.intercept_dist:
            return 100.0

        # Failure penalty
        att_to_target = np.linalg.norm(att_pos - self.target_pos)
        if att_to_target < self.cfg.target_reached_dist:
            return -100.0

        # Dense approach reward (reward for getting closer)
        approach_reward = 0.0
        if self.prev_dist is not None:
            approach_reward = (self.prev_dist - dist) * 10.0
        self.prev_dist = dist

        # Time penalty
        time_penalty = -0.01

        return approach_reward + time_penalty

    # ------------------------------------------------------------------
    # Termination / truncation
    # ------------------------------------------------------------------

    def _computeTerminated(self):
        """Episode ends on interception or target reached."""
        dist = np.linalg.norm(self.pos[0] - self.pos[1])
        if dist < self.cfg.intercept_dist:
            return True

        att_to_target = np.linalg.norm(self.pos[1] - self.target_pos)
        if att_to_target < self.cfg.target_reached_dist:
            return True

        return False

    def _computeTruncated(self):
        """Episode truncates on time limit or out-of-bounds."""
        if self.step_counter >= self.max_step_counter:
            return True

        for i in range(2):
            if (abs(self.pos[i, 0]) > self.cfg.arena_bounds
                    or abs(self.pos[i, 1]) > self.cfg.arena_bounds
                    or self.pos[i, 2] < self.cfg.altitude_min
                    or self.pos[i, 2] > self.cfg.altitude_max):
                return True

        return False

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _computeInfo(self):
        """Return episode metrics."""
        dist = np.linalg.norm(self.pos[0] - self.pos[1])
        att_to_target = np.linalg.norm(self.pos[1] - self.target_pos)
        return {
            "distance_to_attacker": dist,
            "attacker_to_target": att_to_target,
            "intercepted": dist < self.cfg.intercept_dist,
            "target_reached": att_to_target < self.cfg.target_reached_dist,
            "step": self.step_counter,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _random_positions(self):
        """Generate random initial positions for both drones."""
        rng = np.random.default_rng()

        # Attacker: starts far from target
        angle = rng.uniform(0, 2 * np.pi)
        dist = rng.uniform(*self.cfg.attacker_dist_range)
        att_pos = np.array([
            self.target_pos[0] + dist * np.cos(angle),
            self.target_pos[1] + dist * np.sin(angle),
            rng.uniform(*self.cfg.attacker_alt_range),
        ])

        # Defender: starts near the target
        def_angle = rng.uniform(0, 2 * np.pi)
        def_dist = rng.uniform(*self.cfg.defender_dist_range)
        def_pos = np.array([
            self.target_pos[0] + def_dist * np.cos(def_angle),
            self.target_pos[1] + def_dist * np.sin(def_angle),
            rng.uniform(*self.cfg.defender_alt_range),
        ])

        return np.array([def_pos, att_pos])
