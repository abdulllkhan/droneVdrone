"""Live AEGIS viewer: watch a trained swarm policy defend in real time.

Top-down tactical view in the render_aegis GIF style (terrain, protected targets,
defenders ^, attackers v, intercept / breach / terrain-loss markers, assignment
lines, trails) with a HUD and keyboard control of the scenario. Per-model env
settings come from eval.aegis_common.model_profile (CLI flags override them).
The policy loads once; scenario edits rebuild the single-world env.

Usage:
    python -m eval.aegis_viewer                       # aegis_vark, 15v15, 3 targets
    python -m eval.aegis_viewer --model models/aegis_adv_defender_cf_lab_best/best_model.zip
    python -m eval.aegis_viewer --model ... --nm 10x20 --strategy saturation --num-targets 2
    python -m eval.aegis_viewer --model models/aegis_adv_attacker_cf_lab_best/best_model.zip
        (an RL attacker plays the 'rl' strategy against the defender of its round)

Keys: Space pause | Right/. step | Up/Down speed | R new seed | S strategy |
      T terrain | 1-4/K targets | N/Shift+N defenders | M/Shift+M attackers |
      V target speed | A assignment lines | C trails | H help | Q/Esc quit
"""

import argparse
import math
import os
import sys
import textwrap
import time
import warnings
from collections import deque

import numpy as np
import pygame
from pygame import gfxdraw

from eval.aegis_common import (DEFAULT_MODEL, STRATEGIES, config_kwargs, default_scenario,
                               frame, load_policy, make_env, model_name, model_profile,
                               paired_defender, world_meta)
from train.train_aegis import parse_overrides

SPEEDS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)   # x real time (env control dt)
TARGET_SPEEDS = (0.0, 5.0, 10.0)                 # convoy target, m/s
HOLD_SECS = 2.5         # outcome banner before auto-continuing
REBUILD_DELAY = 0.35    # debounce scenario edits before rebuilding the env
TRAIL_STEPS = 18        # 1.5 s at 12 Hz, as in the GIFs
FPS = 60

# render_aegis GIF palette
BG = (13, 17, 23)
GRID = (26, 32, 40)
GRID_AXIS = (40, 47, 57)
SPINE = (60, 66, 74)
HILL = (59, 59, 47)
HILL_EDGE = (92, 92, 70)
BOX = (68, 71, 94)
TARGET = (255, 215, 0)
DEF = (47, 129, 247)
DEF_REARM = (30, 70, 130)
DEF_EDGE = (165, 200, 255)
ATT = (248, 81, 73)
ATT_EDGE = (255, 179, 173)
DEAD_DEF = (96, 104, 114)
KILLED_ATT = (150, 66, 66)
BREACH = (255, 107, 0)
TERRAIN_LOSS = (214, 166, 64)
LINE = (139, 148, 158)
TEXT = (230, 237, 243)
DIM = (125, 133, 144)
ACCENT = (88, 166, 255)
PANEL = (22, 27, 34)
PANEL_EDGE = (48, 54, 61)
GOOD = (63, 185, 80)
BAD = (248, 81, 73)
WARN = (210, 153, 34)

HELP = (("Space", "pause / resume"),
        ("Right  .", "single step (while paused); skip banner"),
        ("Up / Down", "faster / slower (1x = real time)"),
        ("R", "new episode with the next seed"),
        ("S / Shift+S", "cycle attacker strategy"),
        ("T", "cycle terrain"),
        ("1-4  K", "number of protected targets"),
        ("N / Shift+N", "defenders +1 / -1"),
        ("M / Shift+M", "attackers +1 / -1"),
        ("V", "target speed 0 / 5 / 10 m/s (1 target)"),
        ("A", "assignment lines on / off"),
        ("C", "trails on / off"),
        ("H  F1", "this help"),
        ("Q  Esc", "quit"))


def outcome_title(o, k=1):
    """Episode outcome dict -> (banner title, color)."""
    if o.get("win"):
        return ("TARGETS DEFENDED" if k > 1 else "TARGET DEFENDED"), GOOD
    if o.get("breaches"):
        return f"BREACHED x{o['breaches']}", BAD
    if o.get("defenders_left", 1) == 0:
        return "DEFENDERS LOST", BAD
    if o.get("timeout"):
        return "TIMEOUT", WARN
    return "EPISODE OVER", TEXT


def _cycle(seq, cur, step=1):
    i = seq.index(cur) if cur in seq else -1
    return seq[(i + step) % len(seq)]


def _blend(c, t):
    return tuple(int(b + (x - b) * t) for x, b in zip(c, BG))


class Viewer:
    """Pygame front end over a single-world AegisSwarmEnv driven by one PPO policy."""

    def __init__(self, model, model_path, profile, scenario, attacker_model="",
                 overrides=None, speed=1.0, size=820, paused=False):
        pygame.init()
        self.map_px, self.mx, self.my = size, 52, 14
        self.panel_x = self.mx + size + 18
        self.screen = pygame.display.set_mode((self.panel_x + 390, self.my + size + 34))
        pygame.display.set_caption(f"AEGIS Viewer - {model_name(model_path)}")
        font = pygame.font.SysFont
        self.f_title = font("monospace", 22, bold=True)
        self.f_big = font("monospace", 30, bold=True)
        self.f_md = font("monospace", 16)
        self.f_hd = font("monospace", 14, bold=True)
        self.f_sm = font("monospace", 14)
        self.f_xs = font("monospace", 12)
        self.clock = pygame.time.Clock()

        self.model, self.model_path, self.prof = model, model_path, profile
        self.attacker_model, self.overrides = attacker_model, dict(overrides or {})
        self.sc, self.pending, self.pending_at = dict(scenario), None, 0.0
        self.strategies = STRATEGIES + (("rl",) if attacker_model else ())
        self.terrains = tuple(dict.fromkeys((scenario["terrain"],) + profile["terrains"]))
        self.speed_idx = int(np.argmin([abs(s - speed) for s in SPEEDS]))
        self.paused, self.show_lines, self.show_trails, self.show_help = paused, True, True, False
        self.env, self._terminal, self.outcome, self.last_outcome = None, None, None, None
        self.msg, self.msg_until = "", 0.0
        self._rate, self._rate_steps, self._rate_wall = 0.0, 0, 0.0
        self.rebuild()

    # ------------------------------------------------------------------ #
    # Simulation
    # ------------------------------------------------------------------ #

    def rebuild(self, keep_stats=False):
        """(Re)create the env for the current scenario; the policy stays loaded."""
        self._drop_env()
        self.env = make_env(config_kwargs(self.prof, self.sc, self.attacker_model,
                                          self.overrides))
        if self.env.observation_space.shape != self.model.observation_space.shape:
            raise SystemExit(f"policy obs {self.model.observation_space.shape} != env obs "
                             f"{self.env.observation_space.shape} (RL attackers go in "
                             f"--attacker-model)")
        self._hook_terminal(self.env)
        self.dt, self.b = self.env.dt, self.env.cfg.arena_bounds
        self.scale = self.map_px / (2.0 * self.b)
        self.obs = self.env.reset()
        if not keep_stats:
            self.stats = dict(episodes=0, wins=0, intercepts=0, breaches=0)
            self.episode, self.last_outcome = 0, None
        self.outcome = None
        self._start_episode()

    def _hook_terminal(self, env):
        """step_wait() auto-resets a finished world (via _spawn_world) before it
        returns, so grab world 0's final state on the way into the respawn."""
        spawn = env._spawn_world

        def capture_then_spawn(w):
            if w == 0:
                self._terminal = self._snapshot()
            spawn(w)
        env._spawn_world = capture_then_spawn

    def _drop_env(self):
        if self.env is not None:
            self.env.__dict__.pop("_spawn_world", None)   # break the hook's ref cycle
            self.env.close()
            self.env = None

    def _snapshot(self):
        env, c = self.env, self.env.cfg
        f = frame(env)
        f["rearm"] = env.rearm[0] > 0
        # the env's defender bad-death test (OOB or inside terrain), to tell
        # terrain losses from kamikaze intercepts when a defender drops out
        p = f["dp"]
        lost = ((np.abs(p[:, 0]) > c.arena_bounds) | (np.abs(p[:, 1]) > c.arena_bounds) |
                (p[:, 2] < c.altitude_min) | (p[:, 2] > c.altitude_max))
        shell, below, _ = env._terrain_shell(env.def_pos)
        if shell is not None:
            lost |= ((shell[0] < 5.0) & below[0]).any(axis=-1)
        f["lost"] = lost
        return f

    def _start_episode(self):
        env = self.env
        self.meta = world_meta(env)
        wind = getattr(env, "_wind_np", np.zeros((1, 3)))[0]
        self.meta.update(k=int(env.k_w[0]), terrain=str(env.world_terrain[0]),
                         wind=float(np.hypot(wind[0], wind[1])),
                         sensor=float(env.sensor_w[0]),
                         target_speed=float(env.target_speed_w[0]))
        self.cur = self.prev = self._snapshot()
        self.def_lost = np.zeros(len(self.cur["da"]), dtype=bool)
        self.trail = deque([(self.cur["dp"], self.cur["ap"])], maxlen=TRAIL_STEPS)
        self.target_path = [self.cur["tp"][:2].copy()]
        self.episode += 1
        self.acc, self.hold, self.ended = 0.0, 0.0, False

    def step_once(self):
        """Advance one control step. True if the episode just ended: the env has
        already respawned, self.cur holds the final frame until _start_episode()."""
        self._terminal = None
        acts, _ = self.model.predict(self.obs, deterministic=True)
        self.env.step_async(acts)
        self.obs, _, _, _ = self.env.step_wait()
        outs = self.env.pop_outcomes()
        snap = self._terminal if outs and self._terminal is not None else self._snapshot()
        self.def_lost |= self.cur["da"] & ~snap["da"] & snap["lost"]
        self.prev, self.cur = self.cur, snap
        self.trail.append((snap["dp"], snap["ap"]))
        self.target_path.append(snap["tp"][:2].copy())
        self._rate_steps += 1
        if not outs:
            return False
        o = self.outcome = self.last_outcome = outs[-1]
        self.stats["episodes"] += 1
        self.stats["wins"] += int(o["win"])
        self.stats["intercepts"] += o["intercepts"]
        self.stats["breaches"] += o["breaches"]
        # outcome banner (wall time), shorter when fast-forwarding
        self.ended, self.hold = True, HOLD_SECS / max(1.0, SPEEDS[self.speed_idx] / 2)
        return True

    def update(self, dt):
        """Advance the sim by dt wall seconds x speed (frame-budgeted)."""
        now = time.monotonic()
        if self.pending is not None and now >= self.pending_at:
            self.apply_pending()
        if not (self.paused or self.ended):            # achieved sim speed over
            self._rate_wall += dt                      # 1 s of running wall time
        if self._rate_wall >= 1.0:
            self._rate = self._rate_steps * self.dt / self._rate_wall
            self._rate_steps, self._rate_wall = 0, 0.0
        if self.paused:
            return
        if self.ended:                                 # outcome banner up
            self.hold -= dt
            if self.hold <= 0:
                self._start_episode()
            return
        self.acc += dt * SPEEDS[self.speed_idx]
        budget = time.perf_counter() + 0.04            # keep the UI responsive
        while self.acc >= self.dt:
            self.acc -= self.dt
            if self.step_once():
                self.acc = 0.0
                break
            if time.perf_counter() > budget:          # can't keep up: drop backlog
                self.acc = min(self.acc, self.dt)
                break

    # ------------------------------------------------------------------ #
    # Controls
    # ------------------------------------------------------------------ #

    def flash(self, msg):
        self.msg, self.msg_until = msg, time.monotonic() + 2.5

    def edit(self, **changes):
        """Queue a scenario edit; the env rebuilds after a short debounce."""
        new = dict(self.pending or self.sc, **changes)
        self.pending = None if new == self.sc else new
        self.pending_at = time.monotonic() + REBUILD_DELAY

    def apply_pending(self):
        if self.pending is not None:
            self.sc, self.pending = self.pending, None
            self.rebuild()

    def _edit_team(self, key, d):
        lo, hi = self.prof["n_lim" if key == "n" else "m_lim"]
        v = (self.pending or self.sc)[key] + d
        if lo <= v <= hi:
            self.edit(**{key: v})
        else:
            side = "defenders" if key == "n" else "attackers"
            self.flash(f"{side}: {lo}-{hi} for this model")

    def handle_key(self, key, mod=0):
        """Apply one key press. Returns False to quit."""
        shift = bool(mod & pygame.KMOD_SHIFT)
        sc = self.pending or self.sc
        digits = {pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
                  pygame.K_KP1: 1, pygame.K_KP2: 2, pygame.K_KP3: 3, pygame.K_KP4: 4}
        if key in (pygame.K_q, pygame.K_ESCAPE):
            return False
        if key == pygame.K_SPACE:
            self.paused = not self.paused
            self.prev, self.acc = self.cur, 0.0
        elif key in (pygame.K_RIGHT, pygame.K_PERIOD):
            if self.ended:
                self._start_episode()
            elif self.paused:
                self.step_once()
                self.prev = self.cur
        elif key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.speed_idx = min(self.speed_idx + 1, len(SPEEDS) - 1)
        elif key in (pygame.K_DOWN, pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_idx = max(self.speed_idx - 1, 0)
        elif key == pygame.K_r:
            same = self.pending is None
            self.sc, self.pending = dict(sc, seed=sc["seed"] + 1), None
            self.rebuild(keep_stats=same)
        elif key == pygame.K_s:
            self.edit(strategy=_cycle(self.strategies, sc["strategy"], -1 if shift else 1))
        elif key == pygame.K_t:
            if len(self.terrains) == 1:
                self.flash(f"terrain fixed to '{self.terrains[0]}' for this model")
            self.edit(terrain=_cycle(self.terrains, sc["terrain"], -1 if shift else 1))
        elif key == pygame.K_k or key in digits:
            mx = self.prof["max_targets"]
            k = sc["num_targets"] % mx + 1 if key == pygame.K_k else digits[key]
            if k > mx:
                self.flash(f"this model defends at most {mx} target(s)")
            else:
                self.edit(num_targets=k)
        elif key == pygame.K_n:
            self._edit_team("n", -1 if shift else 1)
        elif key == pygame.K_m:
            self._edit_team("m", -1 if shift else 1)
        elif key == pygame.K_v:
            v = _cycle(TARGET_SPEEDS, sc["target_speed"])
            if v > 0 and sc["num_targets"] > 1:
                self.flash("a moving target needs 1 protected point")
            self.edit(target_speed=v)
        elif key == pygame.K_a:
            self.show_lines = not self.show_lines
        elif key == pygame.K_c:
            self.show_trails = not self.show_trails
        elif key in (pygame.K_h, pygame.K_F1, pygame.K_SLASH):
            self.show_help = not self.show_help
        return True

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #

    def _w2s(self, p):
        """World (..., >=2) -> screen x, y arrays."""
        p = np.asarray(p, dtype=np.float64)
        return (self.mx + (p[..., 0] + self.b) * self.scale,
                self.my + (self.b - p[..., 1]) * self.scale)

    def _display(self):
        """Current frame, positions interpolated between control steps."""
        c, p = self.cur, self.prev
        a = 1.0 if (self.paused or self.ended or p is c) else min(self.acc / self.dt, 1.0)
        if a >= 1.0:
            return c
        f = dict(c)
        for k in ("dp", "ap", "tp"):
            f[k] = p[k] + (c[k] - p[k]) * a
        return f

    def _put(self, text, x, y, font=None, color=TEXT):
        surf = (font or self.f_sm).render(text, True, color)
        self.screen.blit(surf, (x, y))
        return surf.get_width()

    def _tri(self, x, y, s, up, fill, edge):
        k = -1 if up else 1
        pts = [(int(x), int(y + k * s)), (int(x - 0.87 * s), int(y - 0.5 * k * s)),
               (int(x + 0.87 * s), int(y - 0.5 * k * s))]
        gfxdraw.filled_polygon(self.screen, pts, fill)
        gfxdraw.aapolygon(self.screen, pts, edge)

    def _cross(self, x, y, s, color, ring=None):
        pygame.draw.line(self.screen, color, (x - s, y - s), (x + s, y + s), 2)
        pygame.draw.line(self.screen, color, (x - s, y + s), (x + s, y - s), 2)
        if ring:
            gfxdraw.aacircle(self.screen, int(x), int(y), s + 4, ring)

    def _plus(self, x, y, s, color):
        pygame.draw.line(self.screen, color, (x - s, y), (x + s, y), 4)
        pygame.draw.line(self.screen, color, (x, y - s), (x, y + s), 4)

    def _star(self, x, y, r, fill=TARGET, edge=(255, 255, 255)):
        pts = [(int(x + (r if i % 2 == 0 else 0.45 * r) * math.cos(-math.pi / 2 + i * math.pi / 5)),
                int(y + (r if i % 2 == 0 else 0.45 * r) * math.sin(-math.pi / 2 + i * math.pi / 5)))
               for i in range(10)]
        gfxdraw.filled_polygon(self.screen, pts, fill)
        gfxdraw.aapolygon(self.screen, pts, edge)

    def _dashed(self, color, a, b, dash=5, gap=4):
        (x0, y0), (x1, y1) = a, b
        L = math.hypot(x1 - x0, y1 - y0)
        if L < 1:
            return
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        t = 0.0
        while t < L:
            e = min(t + dash, L)
            pygame.draw.line(self.screen, color, (x0 + ux * t, y0 + uy * t),
                             (x0 + ux * e, y0 + uy * e))
            t += dash + gap

    def _trails(self, pts, alive, color):
        """pts (T, D, 2) screen coords oldest->newest; fading polyline per drone."""
        T = len(pts)
        for i in range(1, T):
            c = _blend(color, 0.12 + 0.5 * i / T)
            for d in np.nonzero(alive)[0]:
                pygame.draw.aaline(self.screen, c, pts[i - 1, d], pts[i, d])

    def _draw_map(self, f):
        scr, meta, s = self.screen, self.meta, self.scale
        rect = pygame.Rect(self.mx, self.my, self.map_px, self.map_px)
        scr.set_clip(rect)
        for g in np.arange(-self.b, self.b + 1e-6, 100.0):
            x, y = self._w2s((g, g))
            col = GRID_AXIS if abs(g) < 1e-6 else GRID
            pygame.draw.line(scr, col, (x, self.my), (x, self.my + self.map_px))
            pygame.draw.line(scr, col, (self.mx, y), (self.mx + self.map_px, y))
        if meta["hills"] is not None:
            r = max(2, int(meta["hill_r"] * s))
            for h in meta["hills"]:
                x, y = self._w2s(h)
                gfxdraw.filled_circle(scr, int(x), int(y), r, HILL)
                gfxdraw.aacircle(scr, int(x), int(y), r, HILL_EDGE)
        if meta["boxes"] is not None:
            for bx in meta["boxes"]:
                x0, y0 = self._w2s((bx[0] - bx[2], bx[1] + bx[3]))
                pygame.draw.rect(scr, BOX, (x0, y0, float(2 * bx[2] * s), float(2 * bx[3] * s)))
        k = meta["k"]
        if k == 1 and len(self.target_path) > 1:           # convoy path so far
            xs, ys = self._w2s(np.array(self.target_path))
            pygame.draw.lines(scr, _blend(TARGET, 0.4), False, list(zip(xs, ys)), 2)

        dx, dy = self._w2s(f["dp"])
        ax, ay = self._w2s(f["ap"])
        if self.show_trails and len(self.trail) > 1:
            tr = list(self.trail)
            dp = np.stack([np.stack(self._w2s(t[0]), -1) for t in tr] + [np.stack((dx, dy), -1)])
            ap = np.stack([np.stack(self._w2s(t[1]), -1) for t in tr] + [np.stack((ax, ay), -1)])
            self._trails(dp, f["da"], DEF)
            self._trails(ap, f["aa"], ATT)
        if self.show_lines:
            for d in np.nonzero(f["da"])[0]:
                a = f["assign"][d]
                if a >= 0 and f["aa"][a]:
                    self._dashed(LINE, (dx[d], dy[d]), (ax[a], ay[a]))

        for d in np.nonzero(~f["da"])[0]:
            self._cross(dx[d], dy[d], 4, DEAD_DEF, TERRAIN_LOSS if self.def_lost[d] else None)
        for a in np.nonzero(~f["aa"] & ~f["ab"])[0]:
            self._cross(ax[a], ay[a], 4, KILLED_ATT, TERRAIN_LOSS if f["at"][a] else None)
        for a in np.nonzero(f["ab"])[0]:
            self._plus(ax[a], ay[a], 6, BREACH)
        if k > 1:
            for t in meta["targets"][:k]:
                self._star(*self._w2s(t), 12)
        else:
            self._star(*self._w2s(f["tp"]), 12)
        for d in np.nonzero(f["da"])[0]:
            self._tri(dx[d], dy[d], 7, True, DEF_REARM if f["rearm"][d] else DEF, DEF_EDGE)
        for a in np.nonzero(f["aa"])[0]:
            self._tri(ax[a], ay[a], 7, False, ATT, ATT_EDGE)
        scr.set_clip(None)

        pygame.draw.rect(scr, SPINE, rect, 1)
        for g in np.arange(-self.b + 100.0, self.b - 1e-6, 200.0):
            x, y = self._w2s((g, g))
            lbl = f"{g:.0f}"
            w = self.f_xs.size(lbl)[0]
            self._put(lbl, x - w / 2, self.my + self.map_px + 4, self.f_xs, DIM)
            self._put(lbl, self.mx - w - 6, y - 7, self.f_xs, DIM)
        bar = 100.0 * s                                      # 100 m scale bar
        x0, y0 = self.mx + 14, self.my + self.map_px - 16
        pygame.draw.line(scr, TEXT, (x0, y0), (x0 + bar, y0), 2)
        unit = "100 m = 0.8 m lab" if self.prof["lab"] else "100 m"
        self._put(unit, x0, y0 - 18, self.f_xs, TEXT)

    def _draw_panel(self, f):
        scr, x = self.screen, self.panel_x
        W = scr.get_width() - x - 10
        pygame.draw.rect(scr, PANEL, (x - 8, self.my, W + 8, self.map_px))
        pygame.draw.rect(scr, PANEL_EDGE, (x - 8, self.my, W + 8, self.map_px), 1)
        y = self.my + 10
        sc, prof, meta = self.sc, self.prof, self.meta

        def line(text, color=TEXT, font=None, dy=19, dx=6):
            nonlocal y
            self._put(text, x + dx, y, font or self.f_sm, color)
            y += dy

        def head(text):
            nonlocal y
            y += 6
            pygame.draw.line(scr, PANEL_EDGE, (x, y - 3), (x + W - 10, y - 3))
            line(text, DIM, self.f_hd, 20, 0)

        line("AEGIS", TEXT, self.f_title, 28, 0)
        line(model_name(self.model_path)[:40], TEXT, self.f_md, 21, 0)
        line(prof["label"][:50], ACCENT, self.f_xs, 16, 0)
        for note in textwrap.wrap(prof["note"], 50)[:2]:
            line(note, DIM, self.f_xs, 16, 0)
        if self.attacker_model:
            line(("vs RL attacker " + model_name(self.attacker_model))[:50], BAD, self.f_xs, 16, 0)

        head("SCENARIO")
        line(f"{sc['n']} v {sc['m']}   " + ("kamikaze" if sc["kamikaze"] else "reusable interceptors"))
        line(f"attackers  {sc['strategy']}")
        terr = meta["terrain"]
        if terr == "hills" and meta["hills"] is None:
            terr = "hills (none placed)"               # plain world: num_obstacles=0
        world = terr + (f"  wind {meta['wind']:.2f} N" if meta["wind"] > 0 else "  calm")
        line(f"terrain    {world}")
        if meta["k"] > 1:
            line(f"targets    {meta['k']} points, spread {self.env.cfg.target_spread:.0f} m")
        else:
            mv = meta["target_speed"]
            line("target     " + (f"moving {mv:.0f} m/s" if mv > 0 else "1, static"))
        line("sensors    " + (f"{meta['sensor']:.0f} m" if meta["sensor"] > 0 else "unlimited (mocap)"))
        line(f"seed {sc['seed']}   episode {self.episode}", DIM)
        if self.pending is not None:
            diff = [f"{k}={v}" for k, v in self.pending.items() if v != sc[k]]
            line(("applying: " + ", ".join(diff))[:46], WARN)

        head("EPISODE")
        n, m = len(f["da"]), len(f["aa"])
        t = f["t"] * self.dt
        lab_t = f"  (lab {t * math.sqrt(2):.1f} s)" if prof["lab"] else ""
        line(f"t {t:5.1f} s   step {f['t']}/{self.env.max_steps}{lab_t}")
        rearm = int((f["rearm"] & f["da"]).sum())
        line(f"defenders  {int(f['da'].sum())}/{n} alive" + (f"  ({rearm} rearming)" if rearm else ""))
        line(f"attackers  {int(f['aa'].sum())}/{m} airborne")
        line(f"intercepts {f['ic']}", GOOD if f["ic"] else TEXT)
        br = int(f["ab"].sum())
        line(f"breaches   {br}", BAD if br else TEXT)
        lost, tl = n - int(f["da"].sum()), int(self.def_lost.sum())
        line(f"def. lost  {lost}" + (f"  ({tl} terrain/OOB)" if tl else ""), WARN if lost else TEXT)
        line(f"att. lost to terrain {int(f['at'].sum())}", DIM)

        head("SESSION")
        st = self.stats
        if st["episodes"]:
            line(f"episodes {st['episodes']}   wins {st['wins']} "
                 f"({st['wins'] / st['episodes']:.0%})")
            o = self.last_outcome
            title, col = outcome_title(o, meta["k"])
            line(f"last: {title.lower()}, {o['intercepts']}/{o['m']} intercepted", col)
        else:
            line("no finished episodes yet", DIM)

        head("PLAYBACK")
        state = "PAUSED" if self.paused else (f"sim {self._rate:.1f}x" if self._rate else "sim --")
        line(f"speed {SPEEDS[self.speed_idx]:g}x   {state}   {self.clock.get_fps():.0f} fps",
             WARN if self.paused else TEXT)
        if self.msg and time.monotonic() < self.msg_until:
            line(self.msg[:46], WARN)

        head("LEGEND")
        y0 = y + 8
        self._tri(x + 14, y0, 7, True, DEF, DEF_EDGE)
        self._put("defender", x + 26, y0 - 8, self.f_xs)
        self._tri(x + 112, y0, 7, False, ATT, ATT_EDGE)
        self._put("attacker", x + 124, y0 - 8, self.f_xs)
        self._star(x + 212, y0, 9)
        self._put("target", x + 226, y0 - 8, self.f_xs)
        y0 += 24
        self._cross(x + 14, y0, 4, DEAD_DEF)
        self._cross(x + 32, y0, 4, KILLED_ATT)
        self._put("lost / intercepted", x + 44, y0 - 8, self.f_xs)
        self._plus(x + 212, y0, 6, BREACH)
        self._put("breach", x + 226, y0 - 8, self.f_xs)
        y0 += 24
        self._cross(x + 14, y0, 4, DEAD_DEF, TERRAIN_LOSS)
        self._put("terrain / OOB loss", x + 30, y0 - 8, self.f_xs)
        self._tri(x + 212, y0, 7, True, DEF_REARM, DEF_EDGE)
        self._put("rearming", x + 226, y0 - 8, self.f_xs)
        y0 += 22
        self._dashed(LINE, (x + 6, y0), (x + 40, y0))
        self._put("assignment" + ("" if self.show_lines else " (off)"), x + 48, y0 - 8, self.f_xs)
        for i in range(6):
            pygame.draw.line(scr, _blend(DEF, 0.12 + 0.5 * i / 6), (x + 196 + 2 * i, y0),
                             (x + 198 + 2 * i, y0), 2)
        self._put("trails" + ("" if self.show_trails else " (off)"), x + 226, y0 - 8, self.f_xs)

        self._put("H help  Space pause  Up/Down speed  Q quit", x,
                  self.my + self.map_px - 22, self.f_xs, DIM)

    def _overlay(self, w, h, y, border):
        x = self.mx + (self.map_px - w) // 2
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        box.fill((13, 17, 23, 225))
        self.screen.blit(box, (x, y))
        pygame.draw.rect(self.screen, border, (x, y, w, h), 2)
        return x

    def _draw_banner(self):
        o, k = self.outcome, self.meta["k"]
        title, col = outcome_title(o, k)
        x = self._overlay(int(self.map_px * 0.84), 112, self.my + int(self.map_px * 0.1), col)
        cx = x + int(self.map_px * 0.42)
        lines = ((title, self.f_big, col),
                 (f"{o['intercepts']}/{o['m']} intercepted   {o['breaches']} breached   "
                  f"{o['n'] - o['defenders_left']} defenders lost   {o['length'] * self.dt:.1f} s",
                  self.f_sm, TEXT),
                 (f"next episode in {max(self.hold, 0):.1f} s   (Right: skip, Space: pause)",
                  self.f_xs, DIM))
        y = self.my + int(self.map_px * 0.1) + 12
        for text, font, color in lines:
            w = font.size(text)[0]
            self._put(text, cx - w / 2, y, font, color)
            y += font.get_linesize() + 6

    def _draw_help(self):
        h = 40 + 24 * len(HELP)
        x = self._overlay(int(self.map_px * 0.8), h, self.my + (self.map_px - h) // 2, ACCENT)
        y = self.my + (self.map_px - h) // 2 + 14
        self._put("CONTROLS", x + 20, y, self.f_hd, ACCENT)
        for key, what in HELP:
            y += 24
            self._put(key, x + 20, y, self.f_sm, TEXT)
            self._put(what, x + 170, y, self.f_sm, DIM)

    def draw(self):
        self.screen.fill(BG)
        f = self._display()
        self._draw_map(f)
        self._draw_panel(f)
        if self.ended and self.outcome:
            self._draw_banner()
        if self.show_help:
            self._draw_help()

    # ------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------ #

    def tick(self):
        """One UI frame: events, sim update, draw. False when asked to quit."""
        running = True
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                running = self.handle_key(ev.key, ev.mod) and running
        self.update(min(self.clock.tick(FPS) / 1000.0, 0.1))
        self.draw()
        pygame.display.flip()
        return running

    def run(self, frames=0, screenshot="", warmup=0):
        """Main loop; frames > 0 stops after that many frames (headless runs)."""
        for _ in range(warmup):
            if self.step_once():
                self._start_episode()
        self.prev = self.cur
        self._rate_steps, self._rate_wall = 0, 0.0
        self.clock.tick()                  # don't bill env/model setup to frame 1
        n = 0
        while self.tick():
            n += 1
            if frames and n >= frames:
                break
        if screenshot:
            os.makedirs(os.path.dirname(os.path.abspath(screenshot)), exist_ok=True)
            pygame.image.save(self.screen, screenshot)
        self.close()

    def close(self):
        self._drop_env()
        pygame.quit()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Live AEGIS viewer")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="defender PPO zip (an RL attacker zip plays vs its paired defender)")
    p.add_argument("--nm", default="", help="NxM, e.g. 15x15 (default: per model)")
    p.add_argument("--strategy", default="", choices=["", *STRATEGIES, "rl"])
    p.add_argument("--terrain", default="", choices=["", "hills", "urban", "maze", "none"])
    p.add_argument("--num-targets", type=int, default=0, help="protected points, 1-4")
    p.add_argument("--target-speed", type=float, default=None,
                   help="moving-target (convoy) speed in m/s")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--kamikaze", dest="kamikaze", action="store_true", default=None)
    g.add_argument("--reusable", dest="kamikaze", action="store_false")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--realism", dest="realism", action="store_true", default=None,
                   help="5 hills + wind/gusts + 300 m sensing")
    g.add_argument("--plain", dest="realism", action="store_false", help="no hills/wind")
    p.add_argument("--attacker-model", default="",
                   help="frozen RL attacker zip; enables strategy 'rl'")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="AegisConfig override (repeatable), applied last")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--speed", type=float, default=1.0, help="x real time")
    p.add_argument("--device", default="cuda")
    p.add_argument("--size", type=int, default=820, help="map size in pixels")
    p.add_argument("--paused", action="store_true")
    p.add_argument("--warmup", type=int, default=0, help="fast-forward N control steps")
    p.add_argument("--frames", type=int, default=0, help="quit after N frames (0 = never)")
    p.add_argument("--screenshot", default="", help="save the last frame as PNG on exit")
    return p.parse_args(argv)


def scenario_from_args(profile, args, attacker_model=""):
    """Profile defaults, overridden by whatever the CLI set."""
    sc = default_scenario(profile, seed=args.seed)
    if args.nm:
        sc["n"], sc["m"] = map(int, args.nm.lower().split("x"))
    if attacker_model and not args.strategy:
        sc["strategy"] = "rl"
    for key, val in (("strategy", args.strategy), ("terrain", args.terrain),
                     ("num_targets", args.num_targets)):
        if val:
            sc[key] = val
    for key in ("target_speed", "kamikaze", "realism"):
        if getattr(args, key) is not None:
            sc[key] = getattr(args, key)
    return sc


def main(argv=None):
    args = parse_args(argv)
    model_path, attacker = args.model, args.attacker_model
    prof = model_profile(model_path)
    if prof["attacker"]:                      # watch the RL attacker vs its defender
        model_path, attacker = paired_defender(model_path), model_path
    for path in (model_path, attacker):
        if path and not os.path.exists(path):
            sys.exit(f"model not found: {path}")
    sc = scenario_from_args(prof, args, attacker)
    if sc["strategy"] == "rl" and not attacker:
        sys.exit("--strategy rl needs --attacker-model")
    warnings.filterwarnings("ignore", module="stable_baselines3")
    model = load_policy(model_path, device=args.device)
    print(f"AEGIS viewer: {model_name(model_path)} [{prof['family']}] "
          f"{sc['n']}v{sc['m']} {sc['strategy']} {sc['terrain']} targets={sc['num_targets']}"
          + (f" vs {model_name(attacker)}" if attacker else "") + "  (H = help)")
    viewer = Viewer(model, model_path, prof, sc, attacker_model=attacker,
                    overrides=parse_overrides(args.set), speed=args.speed,
                    size=args.size, paused=args.paused)
    viewer.run(frames=args.frames, screenshot=args.screenshot, warmup=args.warmup)
    return viewer


if __name__ == "__main__":
    main()
