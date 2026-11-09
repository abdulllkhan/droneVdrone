"""
Interactive Model Launcher — browse and watch all trained models.

A Pygame-based UI that:
  - Scans models/ for all trained .zip files (+ AEGIS models/aegis_*/best_model.zip)
  - Groups them by training phase (swarm, adversarial, etc.)
  - Shows model details (size, date, obs dim, architecture)
  - Lets you pick a mode (Watch Swarm, Spawn, 1v1 Watch; AEGIS: Watch, Small
    Team, Large Swarm)
  - Launches the appropriate viewer: AEGIS (PPO) models -> eval.aegis_viewer,
    legacy SAC models -> eval.game

Usage:
    python -m eval.launcher
"""

import math
import os
import re
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pygame

from eval.aegis_common import CHAMPIONS, model_name, model_profile, paired_defender

MODELS_DIR = Path("models")

LEGACY_MODES = ("Watch Swarm", "Interactive Spawn", "1v1 Watch")
AEGIS_MODES = ("Watch", "Small Team", "Large Swarm")

# --- Colors ---
BG = (18, 18, 24)
PANEL_BG = (28, 28, 38)
PANEL_BORDER = (50, 50, 70)
HEADER_BG = (35, 35, 50)
TEXT = (220, 220, 230)
TEXT_DIM = (140, 140, 160)
TEXT_BRIGHT = (255, 255, 255)
ACCENT = (80, 160, 255)
ACCENT_DIM = (50, 100, 180)
GREEN = (70, 220, 120)
RED = (255, 80, 80)
ORANGE = (255, 180, 60)
YELLOW = (255, 230, 80)
SELECTED_BG = (40, 60, 100)
HOVER_BG = (35, 45, 65)
BUTTON_BG = (50, 110, 220)
BUTTON_HOVER = (70, 130, 240)
BUTTON_DISABLED = (50, 50, 60)
SCROLLBAR_BG = (35, 35, 50)
SCROLLBAR_FG = (80, 80, 110)
CATEGORY_BG = (30, 35, 50)
DIVIDER = (45, 45, 60)


def is_aegis(name):
    return "aegis_" in name.lower()


def categorize_model(name):
    """Assign a category to a model based on filename patterns."""
    n = name.lower()
    if is_aegis(n):
        if n in CHAMPIONS:
            return "AEGIS Champions"
        return "AEGIS RL Attackers" if "attacker" in n else "AEGIS (other)"
    if "swarm_15v10" in n and "300m" in n:
        return "Swarm 15v10 (300m)"
    if "swarm_15v10" in n:
        return "Swarm 15v10"
    if "swarm_3v3" in n:
        return "Swarm 3v3"
    if "swarm_2v2" in n:
        return "Swarm 2v2"
    if "swarm" in n:
        return "Swarm (other)"
    if "adv_defender" in n or "adv_attacker" in n:
        return "Adversarial"
    if "mix_" in n:
        return "Mix Training"
    if "phase1" in n:
        return "Phase 1 (Fly)"
    if "phase2" in n:
        return "Phase 2 (Chase)"
    if "phase3" in n:
        return "Phase 3 (Intercept)"
    if "domrand" in n or "obstacle" in n:
        return "Domain Randomization"
    if "defender" in n:
        return "Defender (general)"
    return "Other"


def detect_model_type(name):
    """Detect whether this is a swarm, 1v1, attacker, defender, or AEGIS model."""
    n = name.lower()
    if is_aegis(n):
        return "aegis_attacker" if "attacker" in n else "aegis"
    if "swarm" in n:
        return "swarm"
    if "attacker" in n:
        return "attacker"
    return "defender"


def parse_swarm_config(name):
    """Extract NvM config from filename like swarm_15v10_..."""
    m = re.search(r"(\d+)v(\d+)", name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 2, 2


def is_realistic(name):
    """Check if model was trained for 300m arena."""
    return "300m" in name.lower() or "realistic" in name.lower()


def get_stage_name(name):
    """Extract stage info from filename."""
    for s in ["stage1", "stage2", "stage3", "stage4", "stage5"]:
        if s in name.lower():
            labels = {
                "stage1": "Stage 1 (Static)",
                "stage2": "Stage 2 (Wind)",
                "stage3": "Stage 3 (Dynamic)",
                "stage4": "Stage 4 (Hard)",
                "stage5": "Stage 5 (Real-world)",
            }
            return labels.get(s, s)
    if "final" in name.lower():
        return "Final"
    if "best" in name.lower():
        return "Best"
    return ""


class ModelInfo:
    """Metadata about a trained model."""
    def __init__(self, path):
        self.path = path
        self.name = model_name(path)          # models/<dir>/best_model.zip -> <dir>
        self.filename = path.name
        self.category = categorize_model(self.name)
        self.model_type = detect_model_type(self.name)
        self.is_realistic = is_realistic(self.name)
        self.stage = get_stage_name(self.name)
        self.profile = model_profile(path) if self.is_aegis else None

        # File metadata
        stat = path.stat()
        self.size_mb = stat.st_size / (1024 * 1024)
        self.modified = datetime.fromtimestamp(stat.st_mtime)

        # Swarm config
        if self.model_type == "swarm":
            self.n_def, self.n_att = parse_swarm_config(self.name)
        elif self.profile:
            self.n_def, self.n_att = self.profile["nm"]
        else:
            self.n_def, self.n_att = 1, 1

    @property
    def is_aegis(self):
        return self.model_type in ("aegis", "aegis_attacker")


def modes_for(model):
    return AEGIS_MODES if model.is_aegis else LEGACY_MODES


def build_command(model, mode, python=None):
    """ModelInfo + mode label -> argv. AEGIS (PPO) models open the live
    eval.aegis_viewer (an RL attacker plays vs its paired defender); legacy
    SAC models keep the original eval.game command."""
    cmd = [python or sys.executable]
    if model.is_aegis:
        prof = model.profile
        cmd += ["-m", "eval.aegis_viewer"]
        if prof["attacker"]:
            cmd += ["--model", paired_defender(model.path), "--attacker-model", str(model.path)]
        else:
            cmd += ["--model", str(model.path)]
        if mode == "Small Team":
            (n_lo, n_hi), (m_lo, m_hi) = prof["n_lim"], prof["m_lim"]
            cmd += ["--nm", f"{min(max(3, n_lo), n_hi)}x{min(max(3, m_lo), m_hi)}"]
        elif mode == "Large Swarm":
            cmd += ["--nm", "%dx%d" % prof["nm_large"]]
        return cmd

    cmd += ["-m", "eval.game"]
    if mode == "Watch Swarm":
        cmd += ["--mode", "swarm"]
        cmd += ["--defender", str(model.path)]
        cmd += ["--defenders", str(model.n_def)]
        cmd += ["--attackers", str(model.n_att)]
        if model.is_realistic:
            cmd += ["--realistic"]
        cmd += ["--strategy", "mixed"]

    elif mode == "Interactive Spawn":
        cmd += ["--mode", "spawn"]
        cmd += ["--defender", str(model.path)]
        if model.is_realistic:
            cmd += ["--realistic"]

    elif mode == "1v1 Watch":
        cmd += ["--mode", "watch"]
        cmd += ["--defender", str(model.path)]
    return cmd


class Launcher:
    def __init__(self):
        pygame.init()
        self.screen_w, self.screen_h = 1400, 900
        self.screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        pygame.display.set_caption("DroneVDrone — Model Launcher")

        self.font_lg = pygame.font.SysFont("monospace", 22, bold=True)
        self.font_md = pygame.font.SysFont("monospace", 16)
        self.font_sm = pygame.font.SysFont("monospace", 13)
        self.font_title = pygame.font.SysFont("monospace", 28, bold=True)

        # State
        self.models = []
        self.categories = {}  # category -> [ModelInfo]
        self.collapsed = set()  # collapsed categories
        self.selected = None  # selected ModelInfo
        self.hovered_idx = -1
        self.scroll_offset = 0
        self.max_scroll = 0
        self.mode_idx = 0  # 0=Watch Swarm, 1=Spawn, 2=1v1 Watch (AEGIS: AEGIS_MODES)
        self.modes = list(LEGACY_MODES)
        self.launching = False
        self.status_msg = ""
        self.status_time = 0

        # Panel dimensions
        self.list_x = 20
        self.list_y = 80
        self.list_w = 600
        self.list_h = self.screen_h - 100
        self.detail_x = 640
        self.detail_y = 80
        self.detail_w = self.screen_w - 660
        self.detail_h = self.screen_h - 100

        self.row_height = 32
        self.category_height = 36

        self.scan_models()

    def scan_models(self):
        """Scan models/ directory for .zip files."""
        self.models = []
        self.categories = {}
        if not MODELS_DIR.exists():
            return

        # AEGIS champions live in models/aegis_*_best/best_model.zip
        paths = (sorted(MODELS_DIR.glob("*.zip")) +
                 sorted(MODELS_DIR.glob("aegis_*/best_model.zip")))
        for p in paths:
            info = ModelInfo(p)
            self.models.append(info)
            if info.category not in self.categories:
                self.categories[info.category] = []
            self.categories[info.category].append(info)
        if "AEGIS Champions" in self.categories:
            self.categories["AEGIS Champions"].sort(key=lambda m: CHAMPIONS.index(m.name))

        # Sort categories by priority
        cat_order = [
            "AEGIS Champions", "AEGIS (other)", "AEGIS RL Attackers",
            "Swarm 15v10 (300m)", "Swarm 15v10", "Swarm 3v3", "Swarm 2v2",
            "Swarm (other)", "Adversarial", "Mix Training",
            "Phase 3 (Intercept)", "Phase 2 (Chase)", "Phase 1 (Fly)",
            "Domain Randomization", "Defender (general)", "Other"
        ]
        ordered = {}
        for c in cat_order:
            if c in self.categories:
                ordered[c] = self.categories[c]
        for c in self.categories:
            if c not in ordered:
                ordered[c] = self.categories[c]
        self.categories = ordered

        self.status_msg = f"Found {len(self.models)} models in {len(self.categories)} categories"
        self.status_time = time.time()

    def _build_rows(self):
        """Build flat list of (type, data) for rendering. type is 'category' or 'model'."""
        rows = []
        for cat, models in self.categories.items():
            rows.append(("category", cat, len(models)))
            if cat not in self.collapsed:
                for m in models:
                    rows.append(("model", m, None))
        return rows

    def _total_content_height(self, rows):
        h = 0
        for rtype, *_ in rows:
            h += self.category_height if rtype == "category" else self.row_height
        return h

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    return False
                if event.key == pygame.K_r:
                    self.scan_models()
                if event.key == pygame.K_TAB:
                    self.mode_idx = (self.mode_idx + 1) % len(self.modes)
                if event.key == pygame.K_RETURN and self.selected:
                    self.launch_model()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos

                # Scroll wheel
                if event.button == 4:  # scroll up
                    self.scroll_offset = max(0, self.scroll_offset - 40)
                elif event.button == 5:  # scroll down
                    self.scroll_offset = min(self.max_scroll, self.scroll_offset + 40)

                # Click in model list
                if event.button == 1 and self.list_x <= mx <= self.list_x + self.list_w:
                    self._handle_list_click(mx, my)

                # Click on mode buttons
                if event.button == 1 and my >= self.detail_y + self.detail_h - 180:
                    self._handle_detail_click(mx, my)

            if event.type == pygame.MOUSEMOTION:
                mx, my = event.pos
                self.hovered_idx = -1
                if self.list_x <= mx <= self.list_x + self.list_w:
                    rows = self._build_rows()
                    cy = self.list_y + self.category_height - self.scroll_offset
                    for i, (rtype, data, extra) in enumerate(rows):
                        rh = self.category_height if rtype == "category" else self.row_height
                        if cy <= my < cy + rh and cy >= self.list_y:
                            self.hovered_idx = i
                        cy += rh

        return True

    def _handle_list_click(self, mx, my):
        rows = self._build_rows()
        cy = self.list_y + self.category_height - self.scroll_offset
        for i, (rtype, data, extra) in enumerate(rows):
            rh = self.category_height if rtype == "category" else self.row_height
            if cy <= my < cy + rh and cy >= self.list_y and cy + rh <= self.list_y + self.list_h:
                if rtype == "category":
                    if data in self.collapsed:
                        self.collapsed.discard(data)
                    else:
                        self.collapsed.add(data)
                elif rtype == "model":
                    self.selected = data
                    self.modes = list(modes_for(data))
                    # Auto-select appropriate mode
                    if data.is_aegis or data.model_type == "swarm":
                        self.mode_idx = 0
                    elif data.model_type == "attacker":
                        self.mode_idx = 2
                    else:
                        self.mode_idx = 2
                break
            cy += rh

    def _handle_detail_click(self, mx, my):
        if not self.selected:
            return

        # Mode buttons (3 buttons side by side)
        btn_y = self.detail_y + self.detail_h - 170
        btn_w = (self.detail_w - 30) // 3
        for i in range(3):
            bx = self.detail_x + 10 + i * (btn_w + 10)
            if bx <= mx <= bx + btn_w and btn_y <= my <= btn_y + 40:
                self.mode_idx = i

        # Launch button
        launch_y = self.detail_y + self.detail_h - 100
        launch_x = self.detail_x + 10
        launch_w = self.detail_w - 20
        if launch_x <= mx <= launch_x + launch_w and launch_y <= my <= launch_y + 55:
            self.launch_model()

    def launch_model(self):
        if not self.selected:
            return

        model = self.selected
        mode = self.modes[self.mode_idx]
        cmd = build_command(model, mode)

        self.status_msg = f"Launching: {' '.join(cmd[-6:])}"
        self.status_time = time.time()
        self.launching = True

        # Launch in subprocess so launcher stays open
        try:
            subprocess.Popen(cmd)
        except Exception as e:
            self.status_msg = f"Error: {e}"
            self.status_time = time.time()

        self.launching = False

    def draw(self):
        self.screen.fill(BG)

        # Title bar
        pygame.draw.rect(self.screen, HEADER_BG, (0, 0, self.screen_w, 60))
        title = self.font_title.render("DroneVDrone — Model Launcher", True, TEXT_BRIGHT)
        self.screen.blit(title, (20, 15))

        count = self.font_sm.render(f"{len(self.models)} models  |  R = Refresh  |  Tab = Mode  |  Enter = Launch", True, TEXT_DIM)
        self.screen.blit(count, (self.screen_w - count.get_width() - 20, 22))

        # Status bar
        if self.status_msg and time.time() - self.status_time < 5:
            status = self.font_sm.render(self.status_msg, True, GREEN)
            self.screen.blit(status, (20, 62))

        self._draw_model_list()
        self._draw_detail_panel()

    def _draw_model_list(self):
        # Panel background
        pygame.draw.rect(self.screen, PANEL_BG,
                         (self.list_x, self.list_y, self.list_w, self.list_h))
        pygame.draw.rect(self.screen, PANEL_BORDER,
                         (self.list_x, self.list_y, self.list_w, self.list_h), 1)

        # Header
        header = self.font_md.render("Models", True, ACCENT)
        self.screen.blit(header, (self.list_x + 10, self.list_y + 6))

        # Clip region for scrolling
        clip_rect = pygame.Rect(self.list_x, self.list_y + self.category_height,
                                self.list_w, self.list_h - self.category_height)
        self.screen.set_clip(clip_rect)

        rows = self._build_rows()
        total_h = self._total_content_height(rows)
        visible_h = self.list_h - self.category_height
        self.max_scroll = max(0, total_h - visible_h)
        self.scroll_offset = min(self.scroll_offset, self.max_scroll)

        cy = self.list_y + self.category_height - self.scroll_offset

        for i, row in enumerate(rows):
            rtype = row[0]
            rh = self.category_height if rtype == "category" else self.row_height

            if cy + rh < self.list_y + self.category_height:
                cy += rh
                continue
            if cy > self.list_y + self.list_h:
                break

            if rtype == "category":
                cat_name = row[1]
                cat_count = row[2]
                # Category row
                pygame.draw.rect(self.screen, CATEGORY_BG,
                                 (self.list_x + 1, cy, self.list_w - 2, rh))
                pygame.draw.line(self.screen, DIVIDER,
                                 (self.list_x, cy + rh - 1),
                                 (self.list_x + self.list_w, cy + rh - 1))

                arrow = "▼" if cat_name not in self.collapsed else "▶"
                label = self.font_md.render(f"{arrow} {cat_name} ({cat_count})", True, ACCENT)
                self.screen.blit(label, (self.list_x + 10, cy + 8))

            elif rtype == "model":
                model = row[1]

                # Highlight
                bg = PANEL_BG
                if self.selected == model:
                    bg = SELECTED_BG
                elif i == self.hovered_idx:
                    bg = HOVER_BG

                pygame.draw.rect(self.screen, bg,
                                 (self.list_x + 1, cy, self.list_w - 2, rh))

                # Model name
                display_name = model.name
                if len(display_name) > 50:
                    display_name = display_name[:47] + "..."

                color = TEXT_BRIGHT if self.selected == model else TEXT
                name_surf = self.font_sm.render(display_name, True, color)
                self.screen.blit(name_surf, (self.list_x + 24, cy + 4))

                # Size badge
                size_str = f"{model.size_mb:.1f}MB"
                size_surf = self.font_sm.render(size_str, True, TEXT_DIM)
                self.screen.blit(size_surf, (self.list_x + self.list_w - size_surf.get_width() - 40, cy + 4))

                # Stage/type badge
                if model.stage:
                    badge_color = GREEN if "Final" in model.stage or "Best" in model.stage else ORANGE
                    badge = self.font_sm.render(model.stage, True, badge_color)
                    self.screen.blit(badge, (self.list_x + 24, cy + 17))

                # Realistic / AEGIS badge
                if model.is_aegis:
                    label = "RL ATT" if model.profile["attacker"] else (
                        "LAB" if model.profile["lab"] else "AEGIS")
                    r_badge = self.font_sm.render(label, True, ACCENT)
                    self.screen.blit(r_badge, (self.list_x + self.list_w - 100, cy + 17))
                elif model.is_realistic:
                    r_badge = self.font_sm.render("300m", True, YELLOW)
                    self.screen.blit(r_badge, (self.list_x + self.list_w - 100, cy + 17))

            cy += rh

        self.screen.set_clip(None)

        # Scrollbar
        if self.max_scroll > 0:
            sb_x = self.list_x + self.list_w - 8
            sb_h = self.list_h - self.category_height
            sb_y = self.list_y + self.category_height
            pygame.draw.rect(self.screen, SCROLLBAR_BG, (sb_x, sb_y, 6, sb_h))

            thumb_h = max(20, int(sb_h * visible_h / total_h))
            thumb_y = sb_y + int((sb_h - thumb_h) * self.scroll_offset / self.max_scroll)
            pygame.draw.rect(self.screen, SCROLLBAR_FG, (sb_x, thumb_y, 6, thumb_h), border_radius=3)

    def _draw_detail_panel(self):
        # Panel background
        pygame.draw.rect(self.screen, PANEL_BG,
                         (self.detail_x, self.detail_y, self.detail_w, self.detail_h))
        pygame.draw.rect(self.screen, PANEL_BORDER,
                         (self.detail_x, self.detail_y, self.detail_w, self.detail_h), 1)

        if not self.selected:
            msg = self.font_md.render("Select a model to view details", True, TEXT_DIM)
            self.screen.blit(msg, (self.detail_x + 20, self.detail_y + self.detail_h // 2 - 10))
            return

        model = self.selected
        x = self.detail_x + 20
        y = self.detail_y + 15

        # Model name
        name_surf = self.font_lg.render(model.name, True, TEXT_BRIGHT)
        self.screen.blit(name_surf, (x, y))
        y += 35

        # Divider
        pygame.draw.line(self.screen, DIVIDER, (x, y), (x + self.detail_w - 40, y))
        y += 15

        # Details
        if model.is_aegis:
            prof = model.profile
            kind = "RL attacker (PPO)" if prof["attacker"] else "AEGIS swarm policy (PPO)"
            arena = ("8x8 m lab (1:125 unit map)" if prof["lab"]
                     else "1 km AEGIS arena")
            details = [
                ("Category", model.category, ACCENT),
                ("Type", kind, TEXT),
                ("File", str(model.path), TEXT_DIM),
                ("Size", f"{model.size_mb:.1f} MB", TEXT),
                ("Modified", model.modified.strftime("%Y-%m-%d %H:%M"), TEXT),
                ("Arena", arena, YELLOW),
                ("Profile", prof["label"], GREEN),
                ("Default", f"{model.n_def}v{model.n_att} {prof['strategy']}, "
                            f"{prof['terrain']}, {prof['num_targets']} target(s)"
                            + (f", {prof['target_speed']:.0f} m/s" if prof["target_speed"] else ""),
                 ACCENT),
                ("Trained", prof["note"], TEXT_DIM),
            ]
            if prof["attacker"]:
                details.append(("Versus", prof["defender"], RED))
        else:
            details = [
                ("Category", model.category, ACCENT),
                ("Type", model.model_type.capitalize(), TEXT),
                ("File", model.filename, TEXT_DIM),
                ("Size", f"{model.size_mb:.1f} MB", TEXT),
                ("Modified", model.modified.strftime("%Y-%m-%d %H:%M"), TEXT),
                ("Arena", "300m realistic" if model.is_realistic else "8m small", YELLOW if model.is_realistic else TEXT),
            ]
        if model.stage:
            details.append(("Stage", model.stage, GREEN))
        if model.model_type == "swarm":
            details.append(("Config", f"{model.n_def} defenders vs {model.n_att} attackers", ACCENT))

        for label, value, color in details:
            label_surf = self.font_sm.render(f"{label}:", True, TEXT_DIM)
            value_surf = self.font_md.render(value, True, color)
            self.screen.blit(label_surf, (x, y))
            self.screen.blit(value_surf, (x + 100, y - 1))
            y += 26

        # --- Mode selection buttons ---
        y = self.detail_y + self.detail_h - 170
        btn_w = (self.detail_w - 50) // 3
        mx, my_pos = pygame.mouse.get_pos()

        for i, mode_name in enumerate(self.modes):
            bx = x + i * (btn_w + 10)
            is_selected = (i == self.mode_idx)
            is_hover = (bx <= mx <= bx + btn_w and y <= my_pos <= y + 40)

            bg = ACCENT if is_selected else (HOVER_BG if is_hover else PANEL_BG)
            border_col = ACCENT if is_selected else PANEL_BORDER
            pygame.draw.rect(self.screen, bg, (bx, y, btn_w, 40), border_radius=5)
            pygame.draw.rect(self.screen, border_col, (bx, y, btn_w, 40), 1, border_radius=5)

            text_col = TEXT_BRIGHT if is_selected else TEXT
            mode_surf = self.font_sm.render(mode_name, True, text_col)
            self.screen.blit(mode_surf, (bx + (btn_w - mode_surf.get_width()) // 2, y + 12))

        # --- Launch button ---
        launch_y = self.detail_y + self.detail_h - 100
        launch_w = self.detail_w - 40
        is_hover = (x <= mx <= x + launch_w and launch_y <= my_pos <= launch_y + 55)
        bg = BUTTON_HOVER if is_hover else BUTTON_BG

        pygame.draw.rect(self.screen, bg, (x, launch_y, launch_w, 55), border_radius=8)

        launch_text = f"Launch: {self.modes[self.mode_idx]}"
        launch_surf = self.font_lg.render(launch_text, True, TEXT_BRIGHT)
        self.screen.blit(launch_surf, (x + (launch_w - launch_surf.get_width()) // 2, launch_y + 15))

        # Subtitle with command hint (AEGIS commands carry full paths: wrap them)
        cmd_hint = self._get_cmd_hint()
        width = max(20, (self.detail_w - 40) // max(1, self.font_sm.size("M")[0]))
        for i, part in enumerate(textwrap.wrap(cmd_hint, width)[:2]):
            hint_surf = self.font_sm.render(part, True, TEXT_DIM)
            self.screen.blit(hint_surf, (x, launch_y + 62 + 16 * i))

    def _get_cmd_hint(self):
        if not self.selected:
            return ""
        model = self.selected
        mode = self.modes[self.mode_idx]
        if model.is_aegis:
            return " ".join(build_command(model, mode)[2:])
        if mode == "Watch Swarm":
            return f"eval.game --mode swarm --defender {model.filename} --defenders {model.n_def} --attackers {model.n_att}"
        elif mode == "Interactive Spawn":
            return f"eval.game --mode spawn --defender {model.filename}"
        else:
            return f"eval.game --mode watch --defender {model.filename}"

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            running = self.handle_events()
            self.draw()
            pygame.display.flip()
            clock.tick(30)

        pygame.quit()


def main():
    launcher = Launcher()
    launcher.run()


if __name__ == "__main__":
    main()
