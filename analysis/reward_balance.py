"""
Reward Balance Optimizer — find reward parameters where good behavior
dominates bad behavior by the widest margin.

Models 6 defender behaviors as Markov chains and computes total episode
reward for each. Searches for reward configs where:
  1. Fast interceptor >> Eternal chaser (by at least 3x)
  2. Fast interceptor >> Passive defender (by at least 5x)
  3. Fast interceptor >> Clusterer (by at least 5x)
  4. Team spread >> All-chase-same (by at least 2x)
  5. Fast interceptor >> Kamikaze avoider (by at least 3x)

Usage:
    python -m analysis.reward_balance
"""

import itertools
import numpy as np
from dataclasses import dataclass, field


# --- Episode parameters (300m arena, 15v10) ---
EPISODE_STEPS = 1440       # 30 sec × 48 Hz
ARENA_BOUNDS = 150.0       # ±150m
ATTACKER_SPEED = 18.0      # m/s
DEFENDER_SPEED = 15.0      # m/s
DT = 1.0 / 48.0           # timestep
INTERCEPT_DIST = 2.0       # meters
NUM_DEFENDERS = 15
NUM_ATTACKERS = 10
INITIAL_DIST = 80.0        # avg starting distance defender→attacker
ATTACKER_START_DIST = 85.0 # avg attacker distance from target


@dataclass
class RewardConfig:
    """Reward parameters to optimize."""
    # Event-based
    intercept_bonus: float = 150.0        # #1: This defender intercepts
    teammate_intercept_bonus: float = 30.0 # #2: Teammate intercepts
    target_reached_penalty: float = -300.0 # #3: Attacker reaches target
    all_eliminated_bonus: float = 500.0   # #4: All attackers eliminated
    all_defenders_dead_penalty: float = -100.0  # #5: Team loses

    # Per-step
    closing_scale: float = 200.0          # #7: Reward per meter closed
    inaction_penalty: float = -1.0        # #8: Not closing distance
    priority_bonus_max: float = 3.0       # #9: Near dangerous attacker
    separation_penalty_max: float = -3.0  # #10: Too close to teammate
    overlap_penalty: float = -2.0         # #11: Same target as teammate
    time_penalty: float = -0.2            # #14: Per-step urgency


def simulate_fast_interceptor(cfg: RewardConfig):
    """Good behavior: closes on attacker and intercepts quickly."""
    total = 0.0
    dist = INITIAL_DIST
    closing_rate = DEFENDER_SPEED + ATTACKER_SPEED  # head-on closing
    # But attacker is going toward target, not toward defender
    # Net closing = defender_speed - attacker's lateral component
    # On average, closing at ~defender_speed since both converge on intercept point
    closing_per_step = DEFENDER_SPEED * DT  # ~0.3125 m/step

    intercept_step = int(dist / closing_per_step)  # ~256 steps to intercept
    intercept_step = min(intercept_step, EPISODE_STEPS)

    for step in range(EPISODE_STEPS):
        if step < intercept_step:
            # Closing on attacker
            dist -= closing_per_step
            dist = max(dist, INTERCEPT_DIST)
            total += closing_per_step * cfg.closing_scale
            # Priority bonus (attacker getting closer to target)
            threat = max(0, 1.0 - (ATTACKER_START_DIST - ATTACKER_SPEED * DT * step) / ARENA_BOUNDS)
            total += cfg.priority_bonus_max * threat * 0.3  # partial
            total += cfg.time_penalty
        elif step == intercept_step:
            # Intercept happens
            total += cfg.intercept_bonus
            # Defender is eliminated (kamikaze) — no more rewards
            break

    return total, intercept_step


def simulate_eternal_chaser(cfg: RewardConfig):
    """Bad behavior: closes distance but maintains ~5m gap, never intercepts."""
    total = 0.0
    dist = INITIAL_DIST
    closing_per_step = DEFENDER_SPEED * DT
    min_chase_dist = 5.0  # maintains this gap

    for step in range(EPISODE_STEPS):
        if dist > min_chase_dist:
            dist -= closing_per_step
            total += closing_per_step * cfg.closing_scale
        else:
            # Hovering at 5m, not closing
            # Small oscillation: sometimes +0.05m, sometimes -0.05m
            total += cfg.inaction_penalty

        total += cfg.time_penalty

    # Attacker eventually reaches target (not intercepted)
    attacker_travel_time = int(ATTACKER_START_DIST / (ATTACKER_SPEED * DT))
    if attacker_travel_time < EPISODE_STEPS:
        total += cfg.target_reached_penalty

    return total, EPISODE_STEPS


def simulate_passive_ring(cfg: RewardConfig):
    """Bad behavior: stays near target, doesn't pursue."""
    total = 0.0

    for step in range(EPISODE_STEPS):
        # Not closing distance at all
        total += cfg.inaction_penalty
        total += cfg.time_penalty

    # Attackers reach target
    total += cfg.target_reached_penalty

    return total, EPISODE_STEPS


def simulate_clusterer(cfg: RewardConfig):
    """Bad behavior: groups with other defenders, moves slowly."""
    total = 0.0
    dist = INITIAL_DIST
    closing_per_step = DEFENDER_SPEED * DT * 0.3  # slow, hesitant

    for step in range(EPISODE_STEPS):
        if dist > 20.0:
            dist -= closing_per_step
            total += closing_per_step * cfg.closing_scale
        else:
            total += cfg.inaction_penalty

        # Separation penalty (close to other defenders)
        total += cfg.separation_penalty_max * 0.7  # 70% of max, most steps
        # Overlap penalty (same target as teammates)
        total += cfg.overlap_penalty
        total += cfg.time_penalty

    # Too slow, attacker reaches target
    total += cfg.target_reached_penalty

    return total, EPISODE_STEPS


def simulate_kamikaze_avoider(cfg: RewardConfig):
    """Bad behavior: chases but veers away at last moment to avoid death."""
    total = 0.0
    dist = INITIAL_DIST
    closing_per_step = DEFENDER_SPEED * DT

    for step in range(EPISODE_STEPS):
        if dist > 8.0:
            # Closing normally
            dist -= closing_per_step
            total += closing_per_step * cfg.closing_scale
        elif dist > 3.0:
            # Slowing down, oscillating
            dist -= closing_per_step * 0.1
            total += closing_per_step * 0.1 * cfg.closing_scale
        else:
            # Pulling away
            dist += closing_per_step * 0.5
            total += cfg.inaction_penalty

        total += cfg.time_penalty

    # Never intercepts, attacker reaches target
    total += cfg.target_reached_penalty

    return total, EPISODE_STEPS


def simulate_team_spread(cfg: RewardConfig):
    """Good behavior: each defender targets a different attacker."""
    # Each defender intercepts one attacker, spread across 10 attackers
    # With 15 defenders and 10 attackers, ~1.5 defenders per attacker
    # Good spread: 10 defenders each solo one attacker, 5 on backup
    total = 0.0

    # Primary interceptor (10 of 15 defenders)
    primary_reward, intercept_step = simulate_fast_interceptor(cfg)
    total += primary_reward * 10

    # After interception, remaining teammates get teammate bonus
    total += cfg.teammate_intercept_bonus * 10 * 5  # 5 backups × 10 events

    # All attackers eliminated
    total += cfg.all_eliminated_bonus * 15  # all 15 get this

    return total / 15  # per-defender average


def simulate_all_chase_same(cfg: RewardConfig):
    """Bad behavior: all 15 defenders chase the same attacker."""
    total = 0.0
    dist = INITIAL_DIST
    closing_per_step = DEFENDER_SPEED * DT
    intercept_step = int(dist / closing_per_step)

    for step in range(min(intercept_step, EPISODE_STEPS)):
        dist -= closing_per_step
        total += closing_per_step * cfg.closing_scale
        # All 14 teammates also chasing same target
        total += cfg.overlap_penalty
        # Close to other defenders
        total += cfg.separation_penalty_max * 0.5
        total += cfg.time_penalty

    # One gets the intercept, this one probably doesn't
    total += cfg.teammate_intercept_bonus  # teammate got it

    # 9 other attackers reach target
    total += cfg.target_reached_penalty * 9

    return total


def evaluate_config(cfg: RewardConfig):
    """Evaluate a reward config and return behavior margins."""
    fast_r, fast_steps = simulate_fast_interceptor(cfg)
    chase_r, _ = simulate_eternal_chaser(cfg)
    passive_r, _ = simulate_passive_ring(cfg)
    cluster_r, _ = simulate_clusterer(cfg)
    kamikaze_r, _ = simulate_kamikaze_avoider(cfg)
    spread_r = simulate_team_spread(cfg)
    same_r = simulate_all_chase_same(cfg)

    results = {
        "fast_interceptor": fast_r,
        "eternal_chaser": chase_r,
        "passive_ring": passive_r,
        "clusterer": cluster_r,
        "kamikaze_avoider": kamikaze_r,
        "team_spread": spread_r,
        "all_chase_same": same_r,
        "intercept_steps": fast_steps,
    }

    # Compute margins (ratio of good/bad, want > 1)
    margins = {}
    if chase_r != 0:
        margins["fast_vs_chaser"] = fast_r / chase_r if chase_r > 0 else fast_r - chase_r
    if passive_r != 0:
        margins["fast_vs_passive"] = fast_r / passive_r if passive_r > 0 else fast_r - passive_r
    if cluster_r != 0:
        margins["fast_vs_cluster"] = fast_r / cluster_r if cluster_r > 0 else fast_r - cluster_r
    if kamikaze_r != 0:
        margins["fast_vs_kamikaze"] = fast_r / kamikaze_r if kamikaze_r > 0 else fast_r - kamikaze_r
    if same_r != 0:
        margins["spread_vs_same"] = spread_r / same_r if same_r > 0 else spread_r - same_r

    return results, margins


def score_config(cfg: RewardConfig):
    """Score a config. Higher is better. Returns (score, details)."""
    results, margins = evaluate_config(cfg)

    fast = results["fast_interceptor"]
    chase = results["eternal_chaser"]
    passive = results["passive_ring"]
    cluster = results["clusterer"]
    kamikaze = results["kamikaze_avoider"]
    spread = results["team_spread"]
    same = results["all_chase_same"]

    score = 0.0
    penalties = []

    # HARD CONSTRAINTS: fast interceptor must have POSITIVE reward
    if fast <= 0:
        return -10000, "fast_interceptor reward is negative or zero"

    # HARD CONSTRAINTS: bad behaviors must have NEGATIVE reward
    bad_behaviors = [
        ("eternal_chaser", chase),
        ("passive_ring", passive),
        ("clusterer", cluster),
        ("kamikaze_avoider", kamikaze),
    ]
    for name, val in bad_behaviors:
        if val >= 0:
            penalties.append(f"{name} is positive ({val:.0f})")
            score -= 1000

    # SOFT: maximize gap between fast interceptor and bad behaviors
    for name, val in bad_behaviors:
        gap = fast - val
        if gap > 0:
            score += np.log1p(gap) * 10  # log scale to avoid extreme configs
        else:
            score -= 100

    # SOFT: spread should beat all-chase-same
    if spread > same:
        score += np.log1p(spread - same) * 5
    else:
        score -= 200

    # SOFT: fast should dominate chase by at least 3x
    if chase < 0 and fast > 0:
        score += 50  # good: fast positive, chase negative
    elif chase > 0 and fast > chase * 3:
        score += 30

    # SOFT: prefer configs where intercept_bonus is meaningful
    # (> 10% of total fast_interceptor reward)
    intercept_fraction = cfg.intercept_bonus / max(abs(fast), 1)
    if intercept_fraction > 0.1:
        score += 20

    detail = f"fast={fast:.0f} chase={chase:.0f} passive={passive:.0f} cluster={cluster:.0f} kamikaze={kamikaze:.0f} spread={spread:.0f} same={same:.0f}"
    if penalties:
        detail += f" | ISSUES: {'; '.join(penalties)}"

    return score, detail


def grid_search():
    """Search over reward parameter space."""
    print("=" * 80)
    print("REWARD BALANCE OPTIMIZER")
    print("Finding reward configs where good behavior dominates bad behavior")
    print("=" * 80)

    # First, show current config
    print("\n--- CURRENT CONFIG ---")
    current = RewardConfig()
    score, detail = score_config(current)
    results, margins = evaluate_config(current)
    print(f"Score: {score:.1f}")
    print(f"Details: {detail}")
    print(f"Intercept at step: {results['intercept_steps']}")
    for k, v in margins.items():
        print(f"  {k}: {v:.2f}")

    # Search space
    intercept_values = [500, 1000, 2000, 5000, 10000]
    teammate_values = [10, 50, 100, 200]
    closing_values = [5, 10, 20, 50]  # much lower than 200
    inaction_values = [-0.1, -0.5, -1.0, -2.0]
    target_penalty_values = [-500, -1000, -2000, -5000]
    all_elim_values = [1000, 2000, 5000, 10000]
    time_penalty_values = [-0.01, -0.05, -0.1]
    overlap_values = [-0.5, -1.0, -2.0]
    separation_values = [-1.0, -2.0, -5.0]
    priority_values = [0.5, 1.0, 2.0]

    best_score = -float('inf')
    best_cfg = None
    best_detail = ""
    total_configs = 0

    print(f"\nSearching {len(intercept_values)}×{len(closing_values)}×{len(target_penalty_values)}×{len(all_elim_values)}×{len(inaction_values)}×{len(time_penalty_values)}×{len(teammate_values)}×{len(overlap_values)} configs...")

    for intercept in intercept_values:
        for closing in closing_values:
            for target_pen in target_penalty_values:
                for all_elim in all_elim_values:
                    for inaction in inaction_values:
                        for time_pen in time_penalty_values:
                            for teammate in teammate_values:
                                for overlap in overlap_values:
                                    for sep in separation_values:
                                        for priority in priority_values:
                                            cfg = RewardConfig(
                                                intercept_bonus=intercept,
                                                teammate_intercept_bonus=teammate,
                                                closing_scale=closing,
                                                inaction_penalty=inaction,
                                                target_reached_penalty=target_pen,
                                                all_eliminated_bonus=all_elim,
                                                time_penalty=time_pen,
                                                overlap_penalty=overlap,
                                                separation_penalty_max=sep,
                                                priority_bonus_max=priority,
                                            )
                                            score, detail = score_config(cfg)
                                            total_configs += 1

                                            if score > best_score:
                                                best_score = score
                                                best_cfg = cfg
                                                best_detail = detail

    print(f"\nSearched {total_configs:,} configs")
    print(f"\n{'=' * 80}")
    print("BEST CONFIG FOUND")
    print(f"{'=' * 80}")
    print(f"Score: {best_score:.1f}")
    print(f"Details: {best_detail}")
    print()
    print(f"  intercept_bonus:        {best_cfg.intercept_bonus:>8.0f}  (defender intercepts attacker)")
    print(f"  teammate_intercept:     {best_cfg.teammate_intercept_bonus:>8.0f}  (teammate gets a kill)")
    print(f"  target_reached_penalty: {best_cfg.target_reached_penalty:>8.0f}  (attacker reaches target)")
    print(f"  all_eliminated_bonus:   {best_cfg.all_eliminated_bonus:>8.0f}  (team wins)")
    print(f"  closing_scale:          {best_cfg.closing_scale:>8.1f}  (per meter closed per step)")
    print(f"  inaction_penalty:       {best_cfg.inaction_penalty:>8.1f}  (not closing per step)")
    print(f"  priority_bonus_max:     {best_cfg.priority_bonus_max:>8.1f}  (near dangerous attacker per step)")
    print(f"  separation_penalty:     {best_cfg.separation_penalty_max:>8.1f}  (too close to teammate per step)")
    print(f"  overlap_penalty:        {best_cfg.overlap_penalty:>8.1f}  (same target as teammate per step)")
    print(f"  time_penalty:           {best_cfg.time_penalty:>8.2f}  (per step urgency)")

    # Show all behavior rewards under best config
    print(f"\n--- BEHAVIOR REWARDS UNDER BEST CONFIG ---")
    results, margins = evaluate_config(best_cfg)
    behaviors = [
        ("Fast Interceptor (GOOD)", results["fast_interceptor"]),
        ("Eternal Chaser (BAD)", results["eternal_chaser"]),
        ("Passive Ring (BAD)", results["passive_ring"]),
        ("Clusterer (BAD)", results["clusterer"]),
        ("Kamikaze Avoider (BAD)", results["kamikaze_avoider"]),
        ("Team Spread (GOOD)", results["team_spread"]),
        ("All Chase Same (BAD)", results["all_chase_same"]),
    ]

    max_name = max(len(b[0]) for b in behaviors)
    for name, val in behaviors:
        bar_len = int(min(max(val / 100, -50), 50))
        if val > 0:
            bar = "█" * min(bar_len, 50)
            print(f"  {name:<{max_name}} : {val:>+10.0f}  {bar}")
        else:
            bar = "░" * min(abs(bar_len), 50)
            print(f"  {name:<{max_name}} : {val:>+10.0f}  {bar}")

    print(f"\n--- MARGINS ---")
    for k, v in margins.items():
        print(f"  {k}: {v:.2f}")

    # Top 10 configs
    print(f"\n--- TOP 10 CONFIGS ---")
    all_results = []
    for intercept in intercept_values:
        for closing in closing_values:
            for target_pen in target_penalty_values:
                for all_elim in all_elim_values:
                    for inaction in inaction_values:
                        for time_pen in time_penalty_values:
                            for teammate in teammate_values:
                                for overlap in overlap_values:
                                    for sep in separation_values:
                                        for priority in priority_values:
                                            cfg = RewardConfig(
                                                intercept_bonus=intercept,
                                                teammate_intercept_bonus=teammate,
                                                closing_scale=closing,
                                                inaction_penalty=inaction,
                                                target_reached_penalty=target_pen,
                                                all_eliminated_bonus=all_elim,
                                                time_penalty=time_pen,
                                                overlap_penalty=overlap,
                                                separation_penalty_max=sep,
                                                priority_bonus_max=priority,
                                            )
                                            score, detail = score_config(cfg)
                                            all_results.append((score, cfg, detail))

    all_results.sort(key=lambda x: -x[0])
    for i, (score, cfg, detail) in enumerate(all_results[:10]):
        print(f"\n  #{i+1} (score={score:.1f}):")
        print(f"    intercept={cfg.intercept_bonus:.0f} teammate={cfg.teammate_intercept_bonus:.0f} "
              f"closing={cfg.closing_scale:.0f} inaction={cfg.inaction_penalty:.1f} "
              f"target_pen={cfg.target_reached_penalty:.0f} all_elim={cfg.all_eliminated_bonus:.0f} "
              f"time={cfg.time_penalty:.2f} overlap={cfg.overlap_penalty:.1f} "
              f"sep={cfg.separation_penalty_max:.1f} priority={cfg.priority_bonus_max:.1f}")
        print(f"    {detail}")

    return best_cfg


if __name__ == "__main__":
    best = grid_search()
