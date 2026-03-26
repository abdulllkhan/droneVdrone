"""
Visualize a trained model in PyBullet GUI.

Supports all training modes: fly_to_target, chase, intercept.
Loads the model, runs episodes, and shows the drone(s) in 3D.

Usage:
    python -m eval.visualize --model models/phase1_fly_final.zip --mode fly_to_target
    python -m eval.visualize --model models/phase3_defender_final.zip --mode intercept
    python -m eval.visualize --model models/phase1_fly_final.zip --mode fly_to_target --episodes 5
"""

import argparse
import time

import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from stable_baselines3 import SAC

from train.config import EnvConfig


class VisualizeEnv(BaseAviary):
    """PyBullet environment for visualizing trained models."""

    def __init__(self, mode="fly_to_target", config=None):
        self.cfg = config or EnvConfig()
        self.vis_mode = mode
        self.target_pos = np.array(self.cfg.target_pos, dtype=np.float32)

        if mode == "fly_to_target":
            num_drones = 1
            # Random target
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(1.0, self.cfg.target_randomize_range)
            self.nav_target = np.array([
                dist * np.cos(angle),
                dist * np.sin(angle),
                np.random.uniform(*self.cfg.target_alt_range),
            ], dtype=np.float32)
            init_xyzs = np.array([[0.0, 0.0, 1.0]])
        elif mode == "chase":
            num_drones = 1
            self.nav_target = np.array([2.0, 0.0, 1.0], dtype=np.float32)
            self.nav_target_vel = np.array([0.1, 0.2, 0.0], dtype=np.float32)
            init_xyzs = np.array([[0.0, 0.0, 1.0]])
        else:  # intercept
            num_drones = 2
            init_xyzs = np.array([
                [0.0, 0.5, 1.0],
                [2.5, 0.0, 1.0],
            ])

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=num_drones,
            initial_xyzs=init_xyzs,
            initial_rpys=np.zeros((num_drones, 3)),
            physics=Physics.PYB_GND_DRAG_DW,
            pyb_freq=self.cfg.pyb_freq,
            ctrl_freq=self.cfg.ctrl_freq,
            gui=True,
            record=False,
            obstacles=False,
            user_debug_gui=True,
        )

        self.controllers = [
            DSLPIDControl(drone_model=DroneModel.CF2X)
            for _ in range(num_drones)
        ]
        self.pid_targets = init_xyzs.copy()
        self.speed_limit = 0.15 * 30.0 * (1000.0 / 3600.0)
        self.max_step_counter = int(self.cfg.episode_secs * self.cfg.ctrl_freq)

        self._draw_markers()

    def _actionSpace(self):
        from gymnasium import spaces
        return spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

    def _observationSpace(self):
        from gymnasium import spaces
        return spaces.Box(low=-np.inf, high=np.inf, shape=(19,), dtype=np.float32)

    def _computeObs(self):
        """Build 19-dim obs matching the Newton env format."""
        state0 = self._getDroneStateVector(0)
        pos = state0[0:3]
        vel = state0[10:13]

        if self.vis_mode in ("fly_to_target", "chase"):
            tgt = self.nav_target
            tgt_vel = self.nav_target_vel if self.vis_mode == "chase" else np.zeros(3)
            rel = tgt - pos
            dist = np.linalg.norm(rel)
            obs = np.zeros(19, dtype=np.float32)
            obs[0:3] = pos
            obs[3:6] = vel
            obs[6:9] = tgt
            obs[9:12] = tgt_vel
            obs[12:15] = rel
            obs[15] = dist
            return obs
        else:  # intercept
            state1 = self._getDroneStateVector(1)
            att_pos = state1[0:3]
            att_vel = state1[10:13]
            rel = att_pos - pos
            dist = np.linalg.norm(rel)
            obs = np.concatenate([
                pos, vel, att_pos, att_vel,
                rel, [dist], self.target_pos,
            ]).astype(np.float32)
            return obs

    def _preprocessAction(self, action):
        """Convert RL velocity action to PID RPMs."""
        rpms = np.zeros((self.NUM_DRONES, 4))

        # RL drone (drone 0): velocity command → PID target
        vel_cmd = np.array(action, dtype=np.float32) * self.speed_limit
        state0 = self._getDroneStateVector(0)
        self.pid_targets[0] = state0[0:3] + vel_cmd * (1.0 / self.cfg.ctrl_freq)
        self.pid_targets[0] = np.clip(
            self.pid_targets[0],
            [-self.cfg.arena_bounds, -self.cfg.arena_bounds, self.cfg.altitude_min + 0.05],
            [self.cfg.arena_bounds, self.cfg.arena_bounds, self.cfg.altitude_max],
        )
        rpms[0], _, _ = self.controllers[0].computeControlFromState(
            control_timestep=self.CTRL_TIMESTEP,
            state=state0,
            target_pos=self.pid_targets[0],
            target_rpy=np.zeros(3),
        )

        # Attacker (drone 1, intercept mode): scripted pursuit
        if self.NUM_DRONES > 1:
            state1 = self._getDroneStateVector(1)
            to_target = self.target_pos - state1[0:3]
            dist = np.linalg.norm(to_target)
            if dist > 0.01:
                att_vel = to_target / dist * self.cfg.attacker_speed_scale * 0.8
                self.pid_targets[1] = state1[0:3] + att_vel * (1.0 / self.cfg.ctrl_freq)
            rpms[1], _, _ = self.controllers[1].computeControlFromState(
                control_timestep=self.CTRL_TIMESTEP,
                state=state1,
                target_pos=self.pid_targets[1],
                target_rpy=np.zeros(3),
            )

        return rpms

    def _computeReward(self):
        return 0.0

    def _computeTerminated(self):
        state0 = self._getDroneStateVector(0)
        if self.vis_mode in ("fly_to_target", "chase"):
            dist = np.linalg.norm(state0[0:3] - self.nav_target)
            return dist < self.cfg.arrival_dist
        else:
            state1 = self._getDroneStateVector(1)
            drone_dist = np.linalg.norm(state0[0:3] - state1[0:3])
            att_to_tgt = np.linalg.norm(state1[0:3] - self.target_pos)
            return drone_dist < self.cfg.intercept_dist or att_to_tgt < self.cfg.target_reached_dist

    def _computeTruncated(self):
        return self.step_counter >= self.max_step_counter

    def _computeInfo(self):
        state0 = self._getDroneStateVector(0)
        if self.vis_mode in ("fly_to_target", "chase"):
            dist = np.linalg.norm(state0[0:3] - self.nav_target)
            return {"dist_to_target": dist, "arrived": dist < self.cfg.arrival_dist}
        else:
            state1 = self._getDroneStateVector(1)
            drone_dist = np.linalg.norm(state0[0:3] - state1[0:3])
            att_to_tgt = np.linalg.norm(state1[0:3] - self.target_pos)
            return {
                "intercepted": drone_dist < self.cfg.intercept_dist,
                "target_reached": att_to_tgt < self.cfg.target_reached_dist,
            }

    def update_moving_target(self):
        """Move target for chase mode."""
        if self.vis_mode != "chase":
            return
        self.nav_target += self.nav_target_vel * (1.0 / self.cfg.ctrl_freq)
        bounds = self.cfg.arena_bounds * 0.8
        for ax in range(2):
            if abs(self.nav_target[ax]) > bounds:
                self.nav_target_vel[ax] *= -1
                self.nav_target[ax] = np.clip(self.nav_target[ax], -bounds, bounds)
        alt_lo, alt_hi = self.cfg.target_alt_range
        if self.nav_target[2] < alt_lo or self.nav_target[2] > alt_hi:
            self.nav_target_vel[2] *= -1
            self.nav_target[2] = np.clip(self.nav_target[2], alt_lo, alt_hi)

    def randomize_target(self):
        """New random target for fly_to_target mode."""
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(1.0, self.cfg.target_randomize_range)
        self.nav_target = np.array([
            dist * np.cos(angle),
            dist * np.sin(angle),
            np.random.uniform(*self.cfg.target_alt_range),
        ], dtype=np.float32)

    def _draw_markers(self):
        """Draw target and arena markers."""
        if self.vis_mode in ("fly_to_target", "chase"):
            self._draw_nav_target()
        else:
            self._draw_defend_target()
        self._draw_arena()

    def _draw_nav_target(self):
        t = self.nav_target
        size = 0.2
        color = [0, 1, 0]  # Green for nav target
        p.addUserDebugLine([t[0]-size, t[1], t[2]], [t[0]+size, t[1], t[2]],
                           color, lineWidth=4, physicsClientId=self.CLIENT)
        p.addUserDebugLine([t[0], t[1]-size, t[2]], [t[0], t[1]+size, t[2]],
                           color, lineWidth=4, physicsClientId=self.CLIENT)
        p.addUserDebugLine([t[0], t[1], t[2]-size], [t[0], t[1], t[2]+size],
                           color, lineWidth=4, physicsClientId=self.CLIENT)
        p.addUserDebugText("TARGET", [t[0], t[1], t[2]+0.3],
                           textColorRGB=color, textSize=1.5, physicsClientId=self.CLIENT)

    def _draw_defend_target(self):
        t = self.target_pos
        size = 0.15
        color = [1, 0, 0]
        p.addUserDebugLine([t[0]-size, t[1], t[2]], [t[0]+size, t[1], t[2]],
                           color, lineWidth=3, physicsClientId=self.CLIENT)
        p.addUserDebugLine([t[0], t[1]-size, t[2]], [t[0], t[1]+size, t[2]],
                           color, lineWidth=3, physicsClientId=self.CLIENT)
        p.addUserDebugLine([t[0], t[1], t[2]-size], [t[0], t[1], t[2]+size],
                           color, lineWidth=3, physicsClientId=self.CLIENT)
        p.addUserDebugText("DEFEND", [t[0], t[1], t[2]+0.25],
                           textColorRGB=color, textSize=1.5, physicsClientId=self.CLIENT)

    def _draw_arena(self):
        b = self.cfg.arena_bounds
        color = [0.5, 0.5, 0.5]
        corners = [[-b, -b, 0], [b, -b, 0], [b, b, 0], [-b, b, 0]]
        for i in range(4):
            p.addUserDebugLine(corners[i], corners[(i+1) % 4],
                               color, lineWidth=1, physicsClientId=self.CLIENT)


def visualize(args):
    cfg = EnvConfig()
    model = SAC.load(args.model)
    print(f"Loaded model: {args.model}")
    print(f"Mode: {args.mode}")

    for ep in range(args.episodes):
        env = VisualizeEnv(mode=args.mode, config=cfg)
        obs, info = env.reset()
        env._draw_markers()

        done = False
        steps = 0
        total_reward = 0.0
        print(f"\n--- Episode {ep+1}/{args.episodes} ---")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            total_reward += reward

            if args.mode == "chase":
                env.update_moving_target()

            # Real-time pacing
            time.sleep(1.0 / cfg.ctrl_freq)

            # HUD
            if steps % 10 == 0:
                if args.mode in ("fly_to_target", "chase"):
                    print(f"  step={steps:4d} | dist={info.get('dist_to_target', 0):.2f}m", end="\r")
                else:
                    print(f"  step={steps:4d}", end="\r")

        # Outcome
        if args.mode in ("fly_to_target", "chase"):
            if info.get("arrived"):
                print(f"\n  ARRIVED at target! (steps={steps})")
            else:
                print(f"\n  Timeout/bounds. (steps={steps})")
        else:
            if info.get("intercepted"):
                print(f"\n  INTERCEPTED! (steps={steps})")
            elif info.get("target_reached"):
                print(f"\n  Target reached by attacker. (steps={steps})")
            else:
                print(f"\n  Timeout. (steps={steps})")

        env.close()

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize trained model in PyBullet")
    parser.add_argument("--model", type=str, required=True, help="Path to model .zip")
    parser.add_argument("--mode", choices=["fly_to_target", "chase", "intercept"],
                        default="fly_to_target", help="Visualization mode")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes")
    args = parser.parse_args()
    visualize(args)
