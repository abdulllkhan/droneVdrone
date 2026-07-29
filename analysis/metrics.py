"""
Training Metrics Tracker — Collect per-episode diagnostics during training.

Tracks multiple metrics simultaneously to give a full picture of defender behavior:
  - Intercept rate: % of attackers intercepted per episode
  - Breach rate: % of episodes where an attacker reaches the target
  - Timeout rate: % of episodes that end by time limit (no resolution)
  - Mean closing speed: are defenders actually approaching attackers?
  - Clustering score: avg pairwise defender distance (low = bad clustering)
  - Death cause breakdown: kamikaze (good) vs obstacle/ground/OOB (bad)
  - Defender utilization: % of defenders that make at least one intercept
  - Time to first intercept: how quickly defenders engage
  - Coverage spread: angular coverage of defenders around target

Usage:
    tracker = MetricsTracker(num_worlds, N, M)
    # In reward loop:
    tracker.record_step(world_id, step_data)
    # On episode end:
    tracker.record_episode_end(world_id, cause)
    # Periodic summary:
    report = tracker.summarize(reset=True)
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
import json
import os
import time


@dataclass
class EpisodeStats:
    """Per-episode statistics accumulated during an episode."""
    steps: int = 0
    intercepts: int = 0
    breached: bool = False
    timed_out: bool = False
    # Death causes for defenders
    kamikaze_deaths: int = 0
    obstacle_deaths: int = 0
    ground_deaths: int = 0
    oob_deaths: int = 0  # out of bounds
    # Behavior metrics (accumulated per step, averaged at end)
    closing_speeds: list = field(default_factory=list)
    pairwise_distances: list = field(default_factory=list)
    defender_speeds: list = field(default_factory=list)
    # Coverage
    angular_coverages: list = field(default_factory=list)
    # Which defenders intercepted
    intercepting_defenders: set = field(default_factory=set)
    # Time to first intercept
    first_intercept_step: int = -1
    # Total defenders and attackers
    total_defenders: int = 0
    total_attackers: int = 0
    # Reward components (accumulated)
    total_reward: float = 0.0


class MetricsTracker:
    """Collects and summarizes training diagnostics across worlds."""

    def __init__(self, num_worlds, num_defenders, num_attackers):
        self.num_worlds = num_worlds
        self.N = num_defenders
        self.M = num_attackers
        self._episodes = {w: EpisodeStats(total_defenders=num_defenders,
                                           total_attackers=num_attackers)
                          for w in range(num_worlds)}
        self._completed = []  # finished episodes
        self._step_count = 0

    def record_step(self, world_id, def_positions, att_positions, target,
                    eliminated_defenders, eliminated_attackers,
                    def_velocities=None):
        """Record per-step metrics for one world.

        Args:
            def_positions: (N, 3) array of defender positions
            att_positions: (M, 3) array of attacker positions
            target: (3,) target position
            eliminated_defenders: (N,) bool array
            eliminated_attackers: (M,) bool array
            def_velocities: optional (N, 3) array
        """
        ep = self._episodes[world_id]
        ep.steps += 1

        active_defs = [d for d in range(self.N) if not eliminated_defenders[d]]
        active_atts = [a for a in range(self.M) if not eliminated_attackers[a]]

        if not active_defs or not active_atts:
            return

        # Closing speed: avg rate defenders approach nearest attacker
        closing_speeds = []
        for d in active_defs:
            min_dist = float('inf')
            for a in active_atts:
                dist = np.linalg.norm(def_positions[d] - att_positions[a])
                min_dist = min(min_dist, dist)
            closing_speeds.append(min_dist)
        ep.closing_speeds.append(np.mean(closing_speeds))

        # Pairwise defender distance (clustering metric)
        if len(active_defs) > 1:
            dists = []
            for i, d1 in enumerate(active_defs):
                for d2 in active_defs[i+1:]:
                    dists.append(np.linalg.norm(def_positions[d1] - def_positions[d2]))
            ep.pairwise_distances.append(np.mean(dists))

        # Defender speeds
        if def_velocities is not None:
            speeds = [np.linalg.norm(def_velocities[d]) for d in active_defs]
            ep.defender_speeds.append(np.mean(speeds))

        # Angular coverage around target
        if len(active_defs) >= 2:
            angles = []
            for d in active_defs:
                rel = def_positions[d] - target
                angles.append(np.arctan2(rel[1], rel[0]))
            angles.sort()
            if len(angles) >= 2:
                gaps = [angles[i+1] - angles[i] for i in range(len(angles)-1)]
                gaps.append(2 * np.pi - (angles[-1] - angles[0]))
                max_gap = max(gaps)
                coverage = 1.0 - (max_gap / (2 * np.pi))
                ep.angular_coverages.append(coverage)

    def record_intercept(self, world_id, defender_id, step_in_episode):
        """Record a successful interception (kamikaze)."""
        ep = self._episodes[world_id]
        ep.intercepts += 1
        ep.kamikaze_deaths += 1
        ep.intercepting_defenders.add(defender_id)
        if ep.first_intercept_step < 0:
            ep.first_intercept_step = step_in_episode

    def record_death(self, world_id, defender_id, cause):
        """Record a defender death from a non-intercept cause.

        cause: 'obstacle', 'ground', 'oob'
        """
        ep = self._episodes[world_id]
        if cause == 'obstacle':
            ep.obstacle_deaths += 1
        elif cause == 'ground':
            ep.ground_deaths += 1
        elif cause == 'oob':
            ep.oob_deaths += 1

    def record_episode_end(self, world_id, cause='timeout'):
        """Mark episode as complete and store it.

        cause: 'all_intercepted' (team win), 'breached' (target reached),
               'all_defenders_dead', 'timeout', 'oob'
        """
        ep = self._episodes[world_id]
        if cause == 'breached':
            ep.breached = True
        elif cause == 'timeout':
            ep.timed_out = True
        self._completed.append(ep)
        # Reset for new episode
        self._episodes[world_id] = EpisodeStats(
            total_defenders=self.N, total_attackers=self.M
        )

    def record_reward(self, world_id, reward):
        """Accumulate total reward for an episode."""
        self._episodes[world_id].total_reward += reward

    def summarize(self, reset=True):
        """Compute aggregate statistics over completed episodes.

        Returns a dict of metrics. If reset=True, clears completed episodes.
        """
        episodes = self._completed
        if not episodes:
            return {"n_episodes": 0}

        n = len(episodes)

        # Rates
        intercept_rate = np.mean([ep.intercepts / max(ep.total_attackers, 1) for ep in episodes])
        breach_rate = np.mean([float(ep.breached) for ep in episodes])
        timeout_rate = np.mean([float(ep.timed_out) for ep in episodes])
        win_rate = np.mean([float(ep.intercepts == ep.total_attackers) for ep in episodes])

        # Closing speed trend (are they getting closer over time?)
        avg_closing = []
        for ep in episodes:
            if len(ep.closing_speeds) >= 2:
                first_half = np.mean(ep.closing_speeds[:len(ep.closing_speeds)//2])
                second_half = np.mean(ep.closing_speeds[len(ep.closing_speeds)//2:])
                avg_closing.append(first_half - second_half)  # positive = closing
        closing_trend = np.mean(avg_closing) if avg_closing else 0.0

        # Clustering
        avg_pairwise = []
        for ep in episodes:
            if ep.pairwise_distances:
                avg_pairwise.append(np.mean(ep.pairwise_distances))
        clustering_score = np.mean(avg_pairwise) if avg_pairwise else 0.0

        # Speed
        avg_speed = []
        for ep in episodes:
            if ep.defender_speeds:
                avg_speed.append(np.mean(ep.defender_speeds))
        mean_speed = np.mean(avg_speed) if avg_speed else 0.0

        # Death breakdown
        total_kamikaze = sum(ep.kamikaze_deaths for ep in episodes)
        total_obstacle = sum(ep.obstacle_deaths for ep in episodes)
        total_ground = sum(ep.ground_deaths for ep in episodes)
        total_oob = sum(ep.oob_deaths for ep in episodes)
        total_deaths = total_kamikaze + total_obstacle + total_ground + total_oob

        # Utilization
        utilizations = []
        for ep in episodes:
            if ep.total_defenders > 0:
                utilizations.append(len(ep.intercepting_defenders) / ep.total_defenders)
        defender_utilization = np.mean(utilizations) if utilizations else 0.0

        # Time to first intercept
        ttfi = [ep.first_intercept_step for ep in episodes if ep.first_intercept_step >= 0]
        mean_ttfi = np.mean(ttfi) if ttfi else -1

        # Coverage
        coverages = []
        for ep in episodes:
            if ep.angular_coverages:
                coverages.append(np.mean(ep.angular_coverages))
        mean_coverage = np.mean(coverages) if coverages else 0.0

        # Episode length
        mean_length = np.mean([ep.steps for ep in episodes])

        # Reward
        mean_reward = np.mean([ep.total_reward for ep in episodes])

        report = {
            "n_episodes": n,
            "intercept_rate": float(intercept_rate),
            "breach_rate": float(breach_rate),
            "timeout_rate": float(timeout_rate),
            "win_rate": float(win_rate),
            "closing_trend": float(closing_trend),
            "clustering_score": float(clustering_score),
            "mean_defender_speed": float(mean_speed),
            "death_kamikaze": total_kamikaze,
            "death_obstacle": total_obstacle,
            "death_ground": total_ground,
            "death_oob": total_oob,
            "death_total": total_deaths,
            "kamikaze_ratio": float(total_kamikaze / max(total_deaths, 1)),
            "defender_utilization": float(defender_utilization),
            "mean_time_to_first_intercept": float(mean_ttfi),
            "mean_angular_coverage": float(mean_coverage),
            "mean_episode_length": float(mean_length),
            "mean_episode_reward": float(mean_reward),
        }

        if reset:
            self._completed = []

        return report

    @staticmethod
    def format_report(report, phase_name=""):
        """Pretty-print a metrics report."""
        if report["n_episodes"] == 0:
            return f"[{phase_name}] No completed episodes to report."

        lines = []
        if phase_name:
            lines.append(f"{'='*60}")
            lines.append(f"  METRICS REPORT: {phase_name}")
            lines.append(f"{'='*60}")
        lines.append(f"  Episodes: {report['n_episodes']}")
        lines.append(f"  Win rate:        {report['win_rate']:.1%}")
        lines.append(f"  Intercept rate:  {report['intercept_rate']:.1%}")
        lines.append(f"  Breach rate:     {report['breach_rate']:.1%}")
        lines.append(f"  Timeout rate:    {report['timeout_rate']:.1%}")
        lines.append(f"  ---")
        lines.append(f"  Closing trend:   {report['closing_trend']:+.3f} (>0 = approaching)")
        lines.append(f"  Clustering:      {report['clustering_score']:.2f}m avg pairwise dist")
        lines.append(f"  Defender speed:  {report['mean_defender_speed']:.2f} m/s")
        lines.append(f"  Coverage:        {report['mean_angular_coverage']:.1%}")
        lines.append(f"  Utilization:     {report['defender_utilization']:.1%}")
        lines.append(f"  ---")
        lines.append(f"  Deaths: {report['death_total']} total")
        lines.append(f"    Kamikaze (good): {report['death_kamikaze']} ({report['kamikaze_ratio']:.0%})")
        lines.append(f"    Obstacle:        {report['death_obstacle']}")
        lines.append(f"    Ground:          {report['death_ground']}")
        lines.append(f"    Out of bounds:   {report['death_oob']}")
        lines.append(f"  ---")
        lines.append(f"  Time to 1st intercept: {report['mean_time_to_first_intercept']:.0f} steps")
        lines.append(f"  Episode length:  {report['mean_episode_length']:.0f} steps")
        lines.append(f"  Episode reward:  {report['mean_episode_reward']:.0f}")
        return "\n".join(lines)

    def save_report(self, report, filepath, phase_name=""):
        """Append a metrics report to a JSON-lines log file."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase_name,
            **report,
        }
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a") as f:
            f.write(json.dumps(entry) + "\n")


def diagnose(report):
    """Analyze a metrics report and return actionable diagnostics.

    Returns a list of (issue, severity, suggestion) tuples.
    """
    issues = []

    if report["n_episodes"] < 10:
        return [("insufficient_data", "info", "Need more episodes for reliable diagnostics")]

    # Clustering
    if report["clustering_score"] < 0.5:
        issues.append(("clustering", "critical",
                        "Defenders are clustering heavily (avg pairwise dist < 0.5m). "
                        "Increase anti-clustering penalty or add spread bonus."))
    elif report["clustering_score"] < 1.0:
        issues.append(("mild_clustering", "warning",
                        "Defenders show mild clustering. Monitor if it worsens."))

    # No closing
    if report["closing_trend"] <= 0:
        issues.append(("no_closing", "critical",
                        "Defenders are NOT approaching attackers. "
                        "Closing reward may be too weak or other signals dominate."))

    # Low speed
    if report["mean_defender_speed"] < 0.1:
        issues.append(("stationary", "critical",
                        "Defenders are nearly stationary. "
                        "They may be stuck in a local minimum. Check reward magnitudes."))

    # Low intercept rate
    if report["intercept_rate"] < 0.1:
        issues.append(("low_intercept", "critical",
                        "Very few interceptions happening. "
                        "Dense pursuit reward needs to be strengthened."))

    # High breach rate
    if report["breach_rate"] > 0.5:
        issues.append(("high_breach", "warning",
                        "Attackers reaching target in >50% of episodes. "
                        "Defenders not engaging fast enough."))

    # Bad deaths
    if report["death_total"] > 0 and report["kamikaze_ratio"] < 0.8:
        bad_deaths = report["death_obstacle"] + report["death_ground"] + report["death_oob"]
        issues.append(("bad_deaths", "warning",
                        f"{bad_deaths} non-intercept deaths ({1-report['kamikaze_ratio']:.0%}). "
                        "Increase penalties for obstacle/ground/OOB deaths."))

    # Low utilization
    if report["defender_utilization"] < 0.2:
        issues.append(("low_utilization", "warning",
                        "Most defenders never intercept. "
                        "They may be clustering or ignoring attackers."))

    # High timeout
    if report["timeout_rate"] > 0.6:
        issues.append(("high_timeout", "warning",
                        "Most episodes time out without resolution. "
                        "Defenders not engaging. Urgency signal may be too weak."))

    if not issues:
        issues.append(("healthy", "info", "All metrics look reasonable."))

    return issues
