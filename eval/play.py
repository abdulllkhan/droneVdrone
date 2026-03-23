"""
Manual Play Mode — YOU are the attacker, the RL model is the defender.

Fly your drone toward the target while the trained defender tries to
intercept you. Uses PyBullet GUI with keyboard controls.

Controls:
    W / S     — forward / backward
    A / D     — left / right
    SPACE     — up
    LEFT SHIFT — down
    R         — reset episode
    ESC       — quit

Usage:
    python -m eval.play --model models/best_model.zip
    python -m eval.play   (no model = random defender, good for testing controls)
"""

import argparse
import time

import numpy as np
import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from train.config import EnvConfig
import os


# PyBullet key codes
KEY_W = ord('w')
KEY_A = ord('a')
KEY_S = ord('s')
KEY_D = ord('d')
KEY_R = ord('r')
KEY_SPACE = 32
KEY_SHIFT = 65306
KEY_ESC = 27


class PlayableInterceptEnv(BaseAviary):
    """
    Two-drone environment where:
    - Drone 0 = Defender (RL model or random)
    - Drone 1 = Attacker (human keyboard control)

    The human tries to reach the target while the AI defends.
    """

    def __init__(self, config=None):
        self.cfg = config or EnvConfig()
        self.target_pos = np.array(self.cfg.target_pos, dtype=np.float32)

        # Fixed starting positions for playability
        init_xyzs = np.array([
            [0.0, 0.5, 1.0],    # Defender: near target
            [2.5, 0.0, 1.0],    # Attacker: far from target
        ])

        super().__init__(
            drone_model=DroneModel.CF2X,
            num_drones=2,
            initial_xyzs=init_xyzs,
            initial_rpys=np.zeros((2, 3)),
            physics=Physics.PYB_GND_DRAG_DW,
            pyb_freq=self.cfg.pyb_freq,
            ctrl_freq=self.cfg.ctrl_freq,
            gui=True,
            record=False,
            obstacles=False,
            user_debug_gui=True,
        )

        # PID controllers for keyboard-driven flight
        self.attacker_ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)
        self.defender_ctrl = DSLPIDControl(drone_model=DroneModel.CF2X)

        # Human control state
        self.human_target = np.array([2.5, 0.0, 1.0])  # PID target for attacker
        self.move_speed = 0.05  # meters per control step

        # Draw target marker
        self._draw_target()

        self.max_step_counter = int(self.cfg.episode_secs * self.cfg.pyb_freq)

    def _actionSpace(self):
        from gymnasium import spaces
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        from gymnasium import spaces
        return spaces.Box(low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32)

    def _computeObs(self):
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
            def_pos, def_vel, def_rpy,
            att_pos, att_vel,
            rel_pos, dist,
            self.target_pos,
        ])
        return obs.astype(np.float32)

    def _preprocessAction(self, action):
        rpms = np.zeros((2, 4))

        # Defender: RL action [-1,1] → RPMs
        rpms[0] = np.clip((action[0] + 1.0) / 2.0 * self.MAX_RPM, 0, self.MAX_RPM)

        # Attacker: keyboard-driven PID
        rpms[1] = self._get_human_rpms()

        return rpms

    def _get_human_rpms(self):
        """Read keyboard and compute attacker RPMs via PID."""
        keys = p.getKeyboardEvents(physicsClientId=self.CLIENT)

        # Update human target based on keys
        if KEY_W in keys and keys[KEY_W] & p.KEY_IS_DOWN:
            self.human_target[0] -= self.move_speed
        if KEY_S in keys and keys[KEY_S] & p.KEY_IS_DOWN:
            self.human_target[0] += self.move_speed
        if KEY_A in keys and keys[KEY_A] & p.KEY_IS_DOWN:
            self.human_target[1] += self.move_speed
        if KEY_D in keys and keys[KEY_D] & p.KEY_IS_DOWN:
            self.human_target[1] -= self.move_speed
        if KEY_SPACE in keys and keys[KEY_SPACE] & p.KEY_IS_DOWN:
            self.human_target[2] += self.move_speed
        if KEY_SHIFT in keys and keys[KEY_SHIFT] & p.KEY_IS_DOWN:
            self.human_target[2] -= self.move_speed

        # Clamp within arena
        self.human_target = np.clip(
            self.human_target,
            [-self.cfg.arena_bounds, -self.cfg.arena_bounds, self.cfg.altitude_min],
            [self.cfg.arena_bounds, self.cfg.arena_bounds, self.cfg.altitude_max],
        )

        state = self._getDroneStateVector(1)
        rpms, _, _ = self.attacker_ctrl.computeControlFromState(
            control_timestep=self.CTRL_TIMESTEP,
            state=state,
            target_pos=self.human_target,
            target_rpy=np.zeros(3),
        )
        return rpms

    def check_keys(self):
        """Check for reset/quit keys. Returns 'reset', 'quit', or None."""
        keys = p.getKeyboardEvents(physicsClientId=self.CLIENT)
        if KEY_ESC in keys and keys[KEY_ESC] & p.KEY_WAS_TRIGGERED:
            return "quit"
        if KEY_R in keys and keys[KEY_R] & p.KEY_WAS_TRIGGERED:
            return "reset"
        return None

    def _computeReward(self):
        return 0.0

    def _computeTerminated(self):
        dist = np.linalg.norm(self.pos[0] - self.pos[1])
        if dist < self.cfg.intercept_dist:
            return True
        att_to_target = np.linalg.norm(self.pos[1] - self.target_pos)
        if att_to_target < self.cfg.target_reached_dist:
            return True
        return False

    def _computeTruncated(self):
        if self.step_counter >= self.max_step_counter:
            return True
        return False

    def _computeInfo(self):
        dist = np.linalg.norm(self.pos[0] - self.pos[1])
        att_to_target = np.linalg.norm(self.pos[1] - self.target_pos)
        return {
            "distance_to_attacker": dist,
            "attacker_to_target": att_to_target,
            "intercepted": dist < self.cfg.intercept_dist,
            "target_reached": att_to_target < self.cfg.target_reached_dist,
        }

    def _draw_target(self):
        """Draw a visual marker at the target position."""
        t = self.target_pos
        size = 0.15
        color = [1, 0, 0]  # Red
        p.addUserDebugLine(
            [t[0]-size, t[1], t[2]], [t[0]+size, t[1], t[2]],
            color, lineWidth=3, physicsClientId=self.CLIENT)
        p.addUserDebugLine(
            [t[0], t[1]-size, t[2]], [t[0], t[1]+size, t[2]],
            color, lineWidth=3, physicsClientId=self.CLIENT)
        p.addUserDebugLine(
            [t[0], t[1], t[2]-size], [t[0], t[1], t[2]+size],
            color, lineWidth=3, physicsClientId=self.CLIENT)
        # Label
        p.addUserDebugText(
            "TARGET", [t[0], t[1], t[2]+0.25],
            textColorRGB=color, textSize=1.5, physicsClientId=self.CLIENT)


def play(args):
    cfg = EnvConfig()
    env = PlayableInterceptEnv(config=cfg)

    # Load defender model if provided
    model = None
    if args.model:
        model = SAC.load(args.model)
        print(f"Loaded defender model: {args.model}")

        # Load VecNormalize if available
        vec_norm_path = os.path.join(os.path.dirname(args.model), "vec_normalize.pkl")
        if os.path.exists(vec_norm_path):
            # We need to wrap for normalization
            dummy_vec = DummyVecEnv([lambda: env])
            vec_norm = VecNormalize.load(vec_norm_path, dummy_vec)
            vec_norm.training = False
            vec_norm.norm_reward = False
            print("Loaded observation normalization stats")
    else:
        print("No model loaded — defender uses random actions")

    print(f"""
{'='*60}
  DRONE vs DRONE — Play Mode
{'='*60}
  YOU are the ATTACKER (drone on the right).
  Fly toward the RED TARGET and avoid the defender!

  Controls:
    W/S     — forward / backward
    A/D     — left / right
    SPACE   — up
    SHIFT   — down
    R       — reset episode
    ESC     — quit

  Click on the PyBullet window to give it keyboard focus.
{'='*60}
""")

    episode = 0
    while True:
        episode += 1
        obs, info = env.reset()
        env.human_target = np.array([2.5, 0.0, 1.0])  # Reset attacker position

        # Re-draw target after reset
        env._draw_target()

        done = False
        steps = 0
        print(f"--- Episode {episode} ---")

        while not done:
            # Check for reset/quit
            key_action = env.check_keys()
            if key_action == "quit":
                print("\nQuitting...")
                env.close()
                return
            if key_action == "reset":
                print("  Resetting...")
                break

            # Get defender action
            if model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

            # Real-time pacing
            time.sleep(1.0 / cfg.ctrl_freq)

            # HUD update every 10 steps
            if steps % 10 == 0:
                dist = info["distance_to_attacker"]
                att_dist = info["attacker_to_target"]
                print(f"  step={steps:4d} | drone_dist={dist:.2f}m | "
                      f"to_target={att_dist:.2f}m", end="\r")

        # Episode outcome
        if info.get("intercepted"):
            print(f"\n  INTERCEPTED! Defender wins. (steps={steps})")
        elif info.get("target_reached"):
            print(f"\n  TARGET REACHED! You win! (steps={steps})")
        else:
            print(f"\n  Time/bounds expired. (steps={steps})")

        print()

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play against the defender drone")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained defender model (.zip). "
                             "If omitted, defender uses random actions.")
    args = parser.parse_args()
    play(args)
