"""
Game Mode — fly drones with Xbox controller or keyboard in PyBullet GUI.

Modes:
    free_flight  — One drone, no combat. Validate physics feel.
    attack       — You (Xbox) are the attacker, RL/random defends.
    defend       — You (Xbox) are the defender, scripted attacker.

Usage:
    python -m eval.game --mode free_flight
    python -m eval.game --mode free_flight --keyboard
    python -m eval.game --mode attack --model models/best_model.zip
    python -m eval.game --mode defend --model models/best_model.zip
"""

import argparse
import time
import os

import numpy as np
import pybullet as p
from gym_pybullet_drones.envs.BaseAviary import BaseAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

from train.config import EnvConfig


# PyBullet key codes
KEY_W = ord('w')
KEY_A = ord('a')
KEY_S = ord('s')
KEY_D = ord('d')
KEY_R = ord('r')
KEY_SPACE = 32
KEY_SHIFT = 65306
KEY_ESC = 27


class GameEnv(BaseAviary):
    """PyBullet environment for interactive game modes."""

    def __init__(self, mode="free_flight", config=None):
        self.cfg = config or EnvConfig()
        self.game_mode = mode
        self.target_pos = np.array(self.cfg.target_pos, dtype=np.float32)

        num_drones = 1 if mode == "free_flight" else 2

        if mode == "free_flight":
            init_xyzs = np.array([[0.0, 0.0, 1.0]])
        elif mode == "attack":
            init_xyzs = np.array([
                [0.0, 0.5, 1.0],    # Defender (AI)
                [2.5, 0.0, 1.0],    # Attacker (human)
            ])
        else:  # defend
            init_xyzs = np.array([
                [0.0, 0.5, 1.0],    # Defender (human)
                [2.5, 0.0, 1.0],    # Attacker (scripted)
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

        # PID controllers
        self.controllers = [
            DSLPIDControl(drone_model=DroneModel.CF2X)
            for _ in range(num_drones)
        ]

        # PID targets (position-based control)
        self.pid_targets = init_xyzs.copy()
        self.move_speed = 0.05  # meters per control step

        self.max_step_counter = int(self.cfg.episode_secs * self.cfg.pyb_freq)

        # Score tracking
        self.score = {"human": 0, "ai": 0, "timeout": 0}

        # Draw target and arena markers
        if mode != "free_flight":
            self._draw_target()
        self._draw_arena()

    def _actionSpace(self):
        from gymnasium import spaces
        return spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _observationSpace(self):
        from gymnasium import spaces
        obs_dim = 6 if self.game_mode == "free_flight" else 19
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _computeObs(self):
        state0 = self._getDroneStateVector(0)

        if self.game_mode == "free_flight":
            return np.concatenate([state0[0:3], state0[10:13]]).astype(np.float32)

        state1 = self._getDroneStateVector(1)
        def_pos = state0[0:3]
        def_vel = state0[10:13]
        att_pos = state1[0:3]
        att_vel = state1[10:13]
        rel_pos = att_pos - def_pos
        dist = np.array([np.linalg.norm(rel_pos)])
        return np.concatenate([
            def_pos, def_vel, att_pos, att_vel,
            rel_pos, dist, self.target_pos,
        ]).astype(np.float32)

    def _preprocessAction(self, action):
        # Action is not used — we drive RPMs directly from apply_input()
        rpms = np.zeros((self.NUM_DRONES, 4))
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            rpms[i], _, _ = self.controllers[i].computeControlFromState(
                control_timestep=self.CTRL_TIMESTEP,
                state=state,
                target_pos=self.pid_targets[i],
                target_rpy=np.zeros(3),
            )
        return rpms

    def apply_input(self, drone_idx, velocity):
        """Apply velocity command to a drone's PID target.

        Args:
            drone_idx: which drone to control
            velocity: (vx, vy, vz) in [-1, 1]
        """
        self.pid_targets[drone_idx] += velocity * self.move_speed
        # Clamp within arena
        self.pid_targets[drone_idx] = np.clip(
            self.pid_targets[drone_idx],
            [-self.cfg.arena_bounds, -self.cfg.arena_bounds, self.cfg.altitude_min + 0.1],
            [self.cfg.arena_bounds, self.cfg.arena_bounds, self.cfg.altitude_max],
        )

    def get_keyboard_velocity(self):
        """Read keyboard and return (vx, vy, vz) in [-1, 1]."""
        keys = p.getKeyboardEvents(physicsClientId=self.CLIENT)
        vx, vy, vz = 0.0, 0.0, 0.0

        if KEY_W in keys and keys[KEY_W] & p.KEY_IS_DOWN:
            vx = -1.0
        if KEY_S in keys and keys[KEY_S] & p.KEY_IS_DOWN:
            vx = 1.0
        if KEY_A in keys and keys[KEY_A] & p.KEY_IS_DOWN:
            vy = 1.0
        if KEY_D in keys and keys[KEY_D] & p.KEY_IS_DOWN:
            vy = -1.0
        if KEY_SPACE in keys and keys[KEY_SPACE] & p.KEY_IS_DOWN:
            vz = 1.0
        if KEY_SHIFT in keys and keys[KEY_SHIFT] & p.KEY_IS_DOWN:
            vz = -1.0

        return np.array([vx, vy, vz], dtype=np.float32)

    def check_special_keys(self):
        """Check for reset/quit. Returns 'reset', 'quit', or None."""
        keys = p.getKeyboardEvents(physicsClientId=self.CLIENT)
        if KEY_ESC in keys and keys[KEY_ESC] & p.KEY_WAS_TRIGGERED:
            return "quit"
        if KEY_R in keys and keys[KEY_R] & p.KEY_WAS_TRIGGERED:
            return "reset"
        return None

    def get_status(self):
        """Get current game status dict."""
        if self.game_mode == "free_flight":
            state = self._getDroneStateVector(0)
            return {
                "pos": state[0:3],
                "vel": state[10:13],
                "alt": state[2],
            }

        state0 = self._getDroneStateVector(0)
        state1 = self._getDroneStateVector(1)
        drone_dist = np.linalg.norm(state0[0:3] - state1[0:3])
        att_to_target = np.linalg.norm(state1[0:3] - self.target_pos)
        return {
            "drone_dist": drone_dist,
            "att_to_target": att_to_target,
            "intercepted": drone_dist < self.cfg.intercept_dist,
            "target_reached": att_to_target < self.cfg.target_reached_dist,
        }

    def _computeReward(self):
        return 0.0

    def _computeTerminated(self):
        if self.game_mode == "free_flight":
            return False
        status = self.get_status()
        return status["intercepted"] or status["target_reached"]

    def _computeTruncated(self):
        if self.step_counter >= self.max_step_counter:
            return True
        return False

    def _computeInfo(self):
        return self.get_status()

    def _draw_target(self):
        t = self.target_pos
        size = 0.15
        color = [1, 0, 0]
        p.addUserDebugLine(
            [t[0]-size, t[1], t[2]], [t[0]+size, t[1], t[2]],
            color, lineWidth=3, physicsClientId=self.CLIENT)
        p.addUserDebugLine(
            [t[0], t[1]-size, t[2]], [t[0], t[1]+size, t[2]],
            color, lineWidth=3, physicsClientId=self.CLIENT)
        p.addUserDebugLine(
            [t[0], t[1], t[2]-size], [t[0], t[1], t[2]+size],
            color, lineWidth=3, physicsClientId=self.CLIENT)
        p.addUserDebugText(
            "TARGET", [t[0], t[1], t[2]+0.25],
            textColorRGB=color, textSize=1.5, physicsClientId=self.CLIENT)

    def _draw_arena(self):
        b = self.cfg.arena_bounds
        color = [0.5, 0.5, 0.5]
        corners = [[-b, -b, 0], [b, -b, 0], [b, b, 0], [-b, b, 0]]
        for i in range(4):
            p.addUserDebugLine(
                corners[i], corners[(i+1) % 4],
                color, lineWidth=1, physicsClientId=self.CLIENT)


def run_game(args):
    cfg = EnvConfig()
    use_controller = not args.keyboard
    controller = None

    if use_controller:
        try:
            from eval.controller import XboxController
            controller = XboxController()
        except (ImportError, RuntimeError) as e:
            print(f"Controller not available: {e}")
            print("Falling back to keyboard controls.")
            use_controller = False

    # Load RL model if provided
    model = None
    if args.model:
        from stable_baselines3 import SAC
        model = SAC.load(args.model)
        print(f"Loaded model: {args.model}")

    env = GameEnv(mode=args.mode, config=cfg)

    # Determine which drone the human controls
    if args.mode == "free_flight":
        human_drone = 0
        ai_drone = None
    elif args.mode == "attack":
        human_drone = 1  # attacker
        ai_drone = 0     # defender
    else:  # defend
        human_drone = 0  # defender
        ai_drone = 1     # attacker

    input_name = "Xbox Controller" if use_controller else "Keyboard (WASD + Space/Shift)"
    role = {"free_flight": "PILOT", "attack": "ATTACKER", "defend": "DEFENDER"}[args.mode]

    print(f"""
{'='*60}
  DRONE vs DRONE — Game Mode: {args.mode.upper()}
{'='*60}
  You are the {role}.
  Input: {input_name}
  {'AI Defender: ' + (args.model or 'random') if args.mode == 'attack' else ''}
  {'Scripted Attacker' if args.mode == 'defend' else ''}

  Controls:
    {"Left stick = XY, Triggers = Z, B = reset, Start = quit" if use_controller else "WASD = XY, Space/Shift = Z, R = reset, ESC = quit"}

  {'Click on PyBullet window for keyboard focus.' if not use_controller else ''}
{'='*60}
""")

    episode = 0
    while True:
        episode += 1
        obs, info = env.reset()

        # Reset PID targets to initial positions
        if args.mode == "free_flight":
            env.pid_targets[0] = np.array([0.0, 0.0, 1.0])
        elif args.mode == "attack":
            env.pid_targets[0] = np.array([0.0, 0.5, 1.0])
            env.pid_targets[1] = np.array([2.5, 0.0, 1.0])
        else:
            env.pid_targets[0] = np.array([0.0, 0.5, 1.0])
            env.pid_targets[1] = np.array([2.5, 0.0, 1.0])

        if args.mode != "free_flight":
            env._draw_target()

        done = False
        steps = 0
        print(f"--- Episode {episode} --- Score: You={env.score['human']} AI={env.score['ai']} Timeout={env.score['timeout']}")

        while not done:
            # Check quit/reset
            if use_controller:
                buttons = controller.get_buttons()
                if buttons["quit"]:
                    print("\nQuit.")
                    env.close()
                    if controller:
                        controller.close()
                    return
                if buttons["reset"]:
                    print("  Reset.")
                    break
            else:
                key_action = env.check_special_keys()
                if key_action == "quit":
                    print("\nQuit.")
                    env.close()
                    return
                if key_action == "reset":
                    print("  Reset.")
                    break

            # Get human input
            if use_controller:
                vel = controller.get_velocity()
            else:
                vel = env.get_keyboard_velocity()

            env.apply_input(human_drone, vel)

            # AI/scripted control for second drone
            if ai_drone is not None:
                if args.mode == "defend":
                    # Scripted attacker: fly toward target
                    att_state = env._getDroneStateVector(ai_drone)
                    to_target = env.target_pos - att_state[0:3]
                    dist = np.linalg.norm(to_target)
                    if dist > 0.01:
                        att_vel = to_target / dist * env.cfg.attacker_speed_scale
                        env.pid_targets[ai_drone] += att_vel * env.move_speed
                elif model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                    # Convert RL velocity to PID target update
                    ai_vel = action[:3] if len(action) >= 3 else action
                    env.apply_input(ai_drone, ai_vel)

            # Step physics (action is unused, RPMs come from PID targets)
            dummy_action = np.zeros(4, dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(dummy_action)
            done = terminated or truncated
            steps += 1

            # Real-time pacing
            time.sleep(1.0 / cfg.ctrl_freq)

            # HUD
            if steps % 10 == 0:
                status = env.get_status()
                if args.mode == "free_flight":
                    pos = status["pos"]
                    print(f"  pos=({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}) alt={status['alt']:.2f}m", end="\r")
                else:
                    print(f"  step={steps:4d} | drone_dist={status['drone_dist']:.2f}m | "
                          f"to_target={status['att_to_target']:.2f}m", end="\r")

        # Episode outcome
        if args.mode != "free_flight":
            status = env.get_status()
            if status.get("intercepted"):
                if args.mode == "defend":
                    env.score["human"] += 1
                    print(f"\n  INTERCEPTED! You win! (steps={steps})")
                else:
                    env.score["ai"] += 1
                    print(f"\n  INTERCEPTED! AI wins. (steps={steps})")
            elif status.get("target_reached"):
                if args.mode == "attack":
                    env.score["human"] += 1
                    print(f"\n  TARGET REACHED! You win! (steps={steps})")
                else:
                    env.score["ai"] += 1
                    print(f"\n  TARGET REACHED! Attacker wins. (steps={steps})")
            else:
                env.score["timeout"] += 1
                print(f"\n  Timeout/bounds. (steps={steps})")
        else:
            print(f"\n  Episode ended. (steps={steps})")
        print()

    env.close()
    if controller:
        controller.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone Game Mode")
    parser.add_argument("--mode", choices=["free_flight", "attack", "defend"],
                        default="free_flight", help="Game mode")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to trained RL model (.zip)")
    parser.add_argument("--keyboard", action="store_true",
                        help="Use keyboard instead of Xbox controller")
    args = parser.parse_args()
    run_game(args)
