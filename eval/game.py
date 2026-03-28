"""
Interactive drone game — validate physics before RL training.

Modes:
  free      — Fly one drone around with keyboard. No combat.
  defend    — You control defender (blue), RL attacker (red) goes for target (green).
  attack    — You control attacker (red), RL defender (blue) tries to intercept.
  watch     — RL defender vs RL attacker. Just observe.

Controls:
  WASD      — Horizontal movement (forward/back/left/right)
  Space     — Ascend
  LShift    — Descend
  R         — Reset episode
  Q / ESC   — Quit
  +/-       — Zoom in/out
  Arrow keys — Rotate camera

Usage:
  python -m eval.game                                          # free flight
  python -m eval.game --mode watch --defender models/best_model.zip --attacker models/adv_attacker_round1.zip
  python -m eval.game --mode defend --attacker models/mix_attacker_round2.zip
  python -m eval.game --mode attack --defender models/defender_best_generalist.zip
"""

import argparse
import math
import sys

import numpy as np
import pygame
import warp as wp
from stable_baselines3 import SAC

from envs.newton_env import NewtonDroneEnv
from train.config import EnvConfig


# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (40, 40, 40)
DARK_GRAY = (25, 25, 25)
BLUE = (60, 140, 255)       # Defender
RED = (255, 70, 70)         # Attacker
GREEN = (70, 255, 70)       # Target
YELLOW = (255, 255, 80)     # Trail
CYAN = (80, 220, 255)       # Info text
GRID_COLOR = (30, 30, 30)


class DroneGame:
    def __init__(self, mode="free", defender_path=None, attacker_path=None,
                 arena_size=8.0):
        self.mode = mode
        self.arena_size = arena_size
        self.screen_w, self.screen_h = 1200, 800

        # Camera
        self.cam_azimuth = 45.0   # degrees
        self.cam_elevation = 35.0
        self.cam_zoom = 80.0      # pixels per meter

        # Physics env (1 environment only)
        use_two = mode in ("defend", "attack", "watch")
        env_mode = "intercept" if use_two else "fly_to_target"

        cfg_kwargs = {
            "mode": env_mode,
            "arena_bounds": arena_size,
            "episode_secs": 30.0,
        }
        if use_two and attacker_path:
            cfg_kwargs["attacker_model_path"] = attacker_path

        self.cfg = EnvConfig(**cfg_kwargs)
        self.env = NewtonDroneEnv(num_envs=1, config=self.cfg)

        # Load RL models
        self.defender_model = None
        self.attacker_model = None
        if defender_path and mode in ("attack", "watch"):
            self.defender_model = SAC.load(defender_path)
        if attacker_path and mode in ("defend", "watch"):
            self.attacker_model = SAC.load(attacker_path)

        self.use_two = use_two
        self.bodies_per_world = self.env.bodies_per_world

        # Trail history
        self.trail_defender = []
        self.trail_attacker = []
        self.max_trail = 500

        # Stats
        self.episode_steps = 0
        self.episodes = 0
        self.interceptions = 0
        self.target_reached = 0

    def world_to_screen(self, pos3d):
        """Project 3D position to 2D screen using isometric-like view."""
        x, y, z = pos3d
        az = math.radians(self.cam_azimuth)
        el = math.radians(self.cam_elevation)

        # Rotate around Z axis (azimuth)
        rx = x * math.cos(az) - y * math.sin(az)
        ry = x * math.sin(az) + y * math.cos(az)

        # Project with elevation
        sx = rx * self.cam_zoom
        sy = (-ry * math.sin(el) - z * math.cos(el)) * self.cam_zoom

        # Center on screen
        return (
            int(self.screen_w // 2 + sx),
            int(self.screen_h // 2 + sy),
        )

    def draw_grid(self, screen):
        """Draw ground plane grid."""
        step = 1.0
        n = int(self.arena_size)
        for i in range(-n, n + 1):
            p1 = self.world_to_screen((i * step, -n * step, 0))
            p2 = self.world_to_screen((i * step, n * step, 0))
            pygame.draw.line(screen, GRID_COLOR, p1, p2, 1)
            p1 = self.world_to_screen((-n * step, i * step, 0))
            p2 = self.world_to_screen((n * step, i * step, 0))
            pygame.draw.line(screen, GRID_COLOR, p1, p2, 1)

    def draw_axes(self, screen):
        """Draw XYZ axes at origin."""
        origin = self.world_to_screen((0, 0, 0))
        for axis, color, label in [
            ((1, 0, 0), RED, "X"),
            ((0, 1, 0), GREEN, "Y"),
            ((0, 0, 1), BLUE, "Z"),
        ]:
            end = self.world_to_screen(axis)
            pygame.draw.line(screen, color, origin, end, 2)

    def draw_drone(self, screen, pos, color, label, size=10):
        """Draw a drone as a diamond with altitude line."""
        sx, sy = self.world_to_screen(pos)

        # Ground shadow
        gx, gy = self.world_to_screen((pos[0], pos[1], 0))
        pygame.draw.line(screen, (color[0]//3, color[1]//3, color[2]//3),
                         (sx, sy), (gx, gy), 1)
        pygame.draw.circle(screen, (color[0]//3, color[1]//3, color[2]//3),
                           (gx, gy), 3)

        # Drone body (diamond shape)
        points = [
            (sx, sy - size),
            (sx + size, sy),
            (sx, sy + size),
            (sx - size, sy),
        ]
        pygame.draw.polygon(screen, color, points)
        pygame.draw.polygon(screen, WHITE, points, 1)

        # Label
        font = pygame.font.SysFont("monospace", 12)
        text = font.render(f"{label} ({pos[2]:.1f}m)", True, color)
        screen.blit(text, (sx + size + 4, sy - 8))

    def draw_target(self, screen, pos):
        """Draw target as a pulsating star."""
        sx, sy = self.world_to_screen(pos)
        pulse = 8 + 3 * math.sin(pygame.time.get_ticks() / 200.0)
        size = int(pulse)

        for i in range(5):
            angle = math.radians(i * 72 - 90)
            ex = sx + int(size * 1.5 * math.cos(angle))
            ey = sy + int(size * 1.5 * math.sin(angle))
            pygame.draw.line(screen, GREEN, (sx, sy), (ex, ey), 2)

        pygame.draw.circle(screen, GREEN, (sx, sy), size // 2)

        # Ground marker
        gx, gy = self.world_to_screen((pos[0], pos[1], 0))
        pygame.draw.circle(screen, (0, 80, 0), (gx, gy), 4, 1)

    def draw_trail(self, screen, trail, color):
        """Draw fading trail."""
        n = len(trail)
        if n < 2:
            return
        for i in range(1, n):
            alpha = i / n
            c = (int(color[0] * alpha * 0.5),
                 int(color[1] * alpha * 0.5),
                 int(color[2] * alpha * 0.5))
            p1 = self.world_to_screen(trail[i - 1])
            p2 = self.world_to_screen(trail[i])
            pygame.draw.line(screen, c, p1, p2, 1)

    def draw_hud(self, screen, def_pos, att_pos, target_pos):
        """Draw heads-up display with game info."""
        font = pygame.font.SysFont("monospace", 14)
        y = 10
        lines = [
            (f"Mode: {self.mode.upper()}", CYAN),
            (f"Episode: {self.episodes}  Steps: {self.episode_steps}", WHITE),
            (f"Interceptions: {self.interceptions}  Target reached: {self.target_reached}", WHITE),
            ("", WHITE),
        ]

        if def_pos is not None:
            lines.append((f"Defender: ({def_pos[0]:+.2f}, {def_pos[1]:+.2f}, {def_pos[2]:.2f})", BLUE))
        if att_pos is not None:
            lines.append((f"Attacker: ({att_pos[0]:+.2f}, {att_pos[1]:+.2f}, {att_pos[2]:.2f})", RED))
        if target_pos is not None:
            lines.append((f"Target:   ({target_pos[0]:+.2f}, {target_pos[1]:+.2f}, {target_pos[2]:.2f})", GREEN))

        if def_pos is not None and att_pos is not None:
            dist = np.linalg.norm(np.array(def_pos) - np.array(att_pos))
            lines.append((f"Def-Att dist: {dist:.2f}m", YELLOW))
        if att_pos is not None and target_pos is not None:
            dist = np.linalg.norm(np.array(att_pos) - np.array(target_pos))
            lines.append((f"Att-Target dist: {dist:.2f}m", YELLOW))

        lines.append(("", WHITE))
        lines.append(("WASD=move  Space/Shift=up/down  R=reset  Q=quit", GRAY))
        lines.append(("Arrows=rotate camera  +/-=zoom", GRAY))

        for text, color in lines:
            surf = font.render(text, True, color)
            screen.blit(surf, (10, y))
            y += 18

    def draw_arena_bounds(self, screen):
        """Draw arena boundary on ground."""
        b = self.arena_size
        corners = [(-b, -b, 0), (b, -b, 0), (b, b, 0), (-b, b, 0)]
        for i in range(4):
            p1 = self.world_to_screen(corners[i])
            p2 = self.world_to_screen(corners[(i + 1) % 4])
            pygame.draw.line(screen, (60, 0, 0), p1, p2, 1)

    def draw_intercept_radius(self, screen, pos):
        """Draw interception threshold circle around a position."""
        r = self.cfg.intercept_dist
        n_pts = 20
        points = []
        for i in range(n_pts + 1):
            angle = 2 * math.pi * i / n_pts
            p = (pos[0] + r * math.cos(angle),
                 pos[1] + r * math.sin(angle),
                 pos[2])
            points.append(self.world_to_screen(p))
        if len(points) > 1:
            pygame.draw.lines(screen, (80, 80, 0), False, points, 1)

    def get_keyboard_velocity(self, keys):
        """Convert keyboard input to velocity vector."""
        vx, vy, vz = 0.0, 0.0, 0.0
        speed = 1.0

        az = math.radians(self.cam_azimuth)
        fwd_x, fwd_y = -math.sin(az), -math.cos(az)
        right_x, right_y = math.cos(az), -math.sin(az)

        if keys[pygame.K_w]:
            vx += fwd_x * speed
            vy += fwd_y * speed
        if keys[pygame.K_s]:
            vx -= fwd_x * speed
            vy -= fwd_y * speed
        if keys[pygame.K_a]:
            vx -= right_x * speed
            vy -= right_y * speed
        if keys[pygame.K_d]:
            vx += right_x * speed
            vy += right_y * speed
        if keys[pygame.K_SPACE]:
            vz += speed
        if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
            vz -= speed

        return np.array([vx, vy, vz], dtype=np.float32)

    def get_positions(self):
        """Read current drone positions from physics state."""
        wp.synchronize()
        body_q = self.env.state_0.body_q.numpy()

        def_pos = body_q[0][:3].tolist()
        att_pos = None
        if self.use_two:
            att_pos = body_q[1][:3].tolist()

        target_pos = list(self.cfg.target_pos)
        return def_pos, att_pos, target_pos

    def run(self):
        pygame.init()
        screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption(f"DroneVDrone — {self.mode.upper()}")
        clock = pygame.time.Clock()

        obs = self.env.reset()
        self.episodes = 1
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    if event.key == pygame.K_r:
                        obs = self.env.reset()
                        self.trail_defender.clear()
                        self.trail_attacker.clear()
                        self.episode_steps = 0
                        self.episodes += 1

            keys = pygame.key.get_pressed()

            # Camera controls
            if keys[pygame.K_LEFT]:
                self.cam_azimuth -= 2
            if keys[pygame.K_RIGHT]:
                self.cam_azimuth += 2
            if keys[pygame.K_UP]:
                self.cam_elevation = min(89, self.cam_elevation + 2)
            if keys[pygame.K_DOWN]:
                self.cam_elevation = max(5, self.cam_elevation - 2)
            if keys[pygame.K_EQUALS] or keys[pygame.K_PLUS]:
                self.cam_zoom = min(200, self.cam_zoom + 2)
            if keys[pygame.K_MINUS]:
                self.cam_zoom = max(20, self.cam_zoom - 2)

            # Compute action based on mode
            if self.mode == "free":
                vel = self.get_keyboard_velocity(keys)
                action = vel.reshape(1, 3)
            elif self.mode == "defend":
                vel = self.get_keyboard_velocity(keys)
                action = vel.reshape(1, 3)
            elif self.mode == "attack":
                # In intercept mode, drone0 is defender. For human-as-attacker,
                # we send zeros for defender and the env's internal attacker model
                # handles drone1. This mode needs the attacker to be human-controlled
                # which requires env changes. For now, human controls defender.
                vel = self.get_keyboard_velocity(keys)
                action = vel.reshape(1, 3)
            elif self.mode == "watch":
                if self.defender_model:
                    action, _ = self.defender_model.predict(obs, deterministic=True)
                else:
                    action = np.zeros((1, 3), dtype=np.float32)

            obs, rewards, dones, infos = self.env.step(action)
            self.episode_steps += 1

            if dones[0]:
                info = infos[0]
                if info.get("intercepted"):
                    self.interceptions += 1
                elif info.get("target_reached"):
                    self.target_reached += 1
                self.trail_defender.clear()
                self.trail_attacker.clear()
                self.episode_steps = 0
                self.episodes += 1

            # Get positions
            def_pos, att_pos, target_pos = self.get_positions()

            # Update trails
            self.trail_defender.append(tuple(def_pos))
            if len(self.trail_defender) > self.max_trail:
                self.trail_defender.pop(0)
            if att_pos is not None:
                self.trail_attacker.append(tuple(att_pos))
                if len(self.trail_attacker) > self.max_trail:
                    self.trail_attacker.pop(0)

            # --- DRAW ---
            screen.fill(BLACK)
            self.draw_grid(screen)
            self.draw_arena_bounds(screen)
            self.draw_axes(screen)

            self.draw_trail(screen, self.trail_defender, BLUE)
            if att_pos is not None:
                self.draw_trail(screen, self.trail_attacker, RED)

            self.draw_target(screen, target_pos)

            if att_pos is not None:
                self.draw_intercept_radius(screen, att_pos)

            self.draw_drone(screen, def_pos, BLUE, "DEF", size=12)
            if att_pos is not None:
                self.draw_drone(screen, att_pos, RED, "ATK", size=12)

            # Distance line between drones
            if att_pos is not None:
                p1 = self.world_to_screen(def_pos)
                p2 = self.world_to_screen(att_pos)
                dist = np.linalg.norm(np.array(def_pos) - np.array(att_pos))
                color = GREEN if dist < self.cfg.intercept_dist else YELLOW
                pygame.draw.line(screen, color, p1, p2, 1)

            self.draw_hud(screen, def_pos, att_pos, target_pos)

            pygame.display.flip()
            clock.tick(self.env.fps)

        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description="Interactive Drone Game")
    parser.add_argument("--mode", type=str, default="free",
                        choices=["free", "defend", "attack", "watch"])
    parser.add_argument("--defender", type=str, default=None,
                        help="Path to defender model (watch/attack modes)")
    parser.add_argument("--attacker", type=str, default=None,
                        help="Path to attacker model (watch/defend modes)")
    parser.add_argument("--arena", type=float, default=8.0,
                        help="Arena size in meters")
    args = parser.parse_args()

    wp.init()

    game = DroneGame(
        mode=args.mode,
        defender_path=args.defender,
        attacker_path=args.attacker,
        arena_size=args.arena,
    )
    game.run()


if __name__ == "__main__":
    main()
