"""
Swarm Evaluation Matrix — Cross-evaluate defender × attacker pairs in NvM scenarios.

Runs every combination of defender model vs attacker model (scripted + RL)
and prints a results matrix with interception rates.

Usage:
    # Auto-discover models in models/ directory
    python -m eval.eval_swarm_matrix --defenders 2 --attackers 2

    # Specific models only
    python -m eval.eval_swarm_matrix --defenders 2 --attackers 2 \
        --defender-models models/swarm_2v2_final.zip models/swarm_adv_defender_2v2_round1.zip \
        --attacker-models models/swarm_adv_attacker_2v2_round1.zip

    # 3v2 asymmetric
    python -m eval.eval_swarm_matrix --defenders 3 --attackers 2 --episodes 20
"""

import argparse
import glob
import os

import numpy as np
import warp as wp
from stable_baselines3 import SAC

from envs.newton_swarm_env import NewtonSwarmEnv
from train.config import EnvConfig


def eval_swarm_matchup(defender_path, attacker_path, n_def, n_att,
                       num_episodes=20, num_worlds=4, num_obstacles=3,
                       max_steps=480, realworld=False, attacker_strategy="direct"):
    """Run episodes for one defender vs one attacker configuration.

    Returns dict with outcome counts and avg steps.
    """
    env_kwargs = dict(
        num_defenders=n_def,
        num_attackers=n_att,
        num_obstacles=num_obstacles,
        attacker_strategy=attacker_strategy,
    )
    if realworld:
        env_kwargs.update(
            wind_enabled=True,
            wind_max_force=0.5,
            wind_gust_prob=0.03,
            wind_gust_force=1.0,
            obs_noise_std=0.03,
            sensor_range=5.0,
            attacker_speed_variance=0.15,
            drone_mass_variance=0.1,
        )
    if isinstance(attacker_path, str) and attacker_path:
        env_kwargs["attacker_model_path"] = attacker_path

    cfg = EnvConfig(**env_kwargs)
    # Use num_worlds=1 for clean per-episode evaluation (avoids cross-world confusion)
    env = NewtonSwarmEnv(num_worlds=1, config=cfg)
    model = SAC.load(defender_path)

    outcomes = {"all_intercepted": 0, "target_reached": 0, "timeout": 0}
    total_steps = 0
    total_rewards = 0.0

    for ep in range(num_episodes):
        obs = env.reset()
        ep_reward = 0.0
        outcome = "timeout"
        steps = max_steps

        for step in range(max_steps):
            actions, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            ep_reward += float(np.mean(rewards))

            if np.any(dones):
                # With num_worlds=1, infos[0] is always the right world
                info = infos[0]
                if info.get("all_eliminated"):
                    outcome = "all_intercepted"
                elif info.get("target_reached"):
                    outcome = "target_reached"
                steps = step + 1
                break

        outcomes[outcome] += 1
        total_steps += steps
        total_rewards += ep_reward

    env.close()

    return {
        "intercepted": outcomes["all_intercepted"],
        "target_reached": outcomes["target_reached"],
        "timeout": outcomes["timeout"],
        "avg_steps": total_steps / num_episodes,
        "avg_reward": total_rewards / num_episodes,
        "episodes": num_episodes,
    }


def discover_models(pattern, model_dir="models"):
    """Find model files matching a glob pattern."""
    paths = sorted(glob.glob(os.path.join(model_dir, pattern)))
    return [p for p in paths if os.path.isfile(p)]


def main():
    parser = argparse.ArgumentParser(description="Swarm Evaluation Matrix")
    parser.add_argument("--defenders", type=int, default=2)
    parser.add_argument("--attackers", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-worlds", type=int, default=4)
    parser.add_argument("--obstacles", type=int, default=3)
    parser.add_argument("--defender-models", nargs="+", default=None,
                        help="Specific defender model paths")
    parser.add_argument("--attacker-models", nargs="+", default=None,
                        help="Specific attacker model paths (RL)")
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--realworld", action="store_true",
                        help="Evaluate under real-world conditions (wind, gusts, sensor limits, variance)")
    parser.add_argument("--strategies", nargs="+", default=["direct", "flanking", "mixed"],
                        help="Scripted attacker strategies to test")
    args = parser.parse_args()

    wp.init()

    n_def = args.defenders
    n_att = args.attackers
    tag = f"{n_def}v{n_att}"

    # Discover defender models
    if args.defender_models:
        def_paths = args.defender_models
    else:
        def_paths = discover_models(f"swarm*{tag}*.zip", args.model_dir)
        # Also check for adv/mix defender models
        def_paths += discover_models(f"swarm_adv_defender_{tag}*.zip", args.model_dir)
        def_paths += discover_models(f"swarm_mix_def_{tag}*.zip", args.model_dir)
        # Deduplicate preserving order
        seen = set()
        unique = []
        for p in def_paths:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        def_paths = unique

    if not def_paths:
        print(f"No defender models found for {tag}. Use --defender-models.")
        return

    # Discover attacker models — scripted strategies
    att_entries = {}
    for strat in args.strategies:
        att_entries[f"Scripted({strat})"] = ("scripted", strat)
    if args.attacker_models:
        for p in args.attacker_models:
            name = os.path.basename(p).replace(".zip", "")
            att_entries[name] = ("rl", p)
    else:
        att_paths = discover_models(f"swarm_adv_attacker_{tag}*.zip", args.model_dir)
        for p in att_paths:
            name = os.path.basename(p).replace(".zip", "")
            att_entries[name] = ("rl", p)

    # Print header
    att_names = list(att_entries.keys())
    print(f"\n{'='*100}")
    print(f"  SWARM EVALUATION MATRIX — {tag} ({args.episodes} episodes per matchup)")
    rw = "ON" if args.realworld else "OFF"
    print(f"  Obstacles: {args.obstacles} | Worlds: {args.num_worlds} | Real-world: {rw}")
    print(f"{'='*100}\n")

    # Column widths
    col_w = max(16, max(len(n) for n in att_names) + 4)
    def_col_w = max(30, max(len(os.path.basename(p)) for p in def_paths) + 2)

    header = f"{'Defender':<{def_col_w}}" + "".join(f"{'| ' + n:<{col_w}}" for n in att_names)
    print(header)
    print("-" * len(header))

    all_results = {}

    for def_path in def_paths:
        def_name = os.path.basename(def_path).replace(".zip", "")
        row = f"{def_name:<{def_col_w}}"

        for att_name, att_info in att_entries.items():
            att_type, att_value = att_info
            if att_type == "rl" and not os.path.exists(att_value):
                row += f"{'| N/A':<{col_w}}"
                continue

            att_path = att_value if att_type == "rl" else None
            att_strategy = att_value if att_type == "scripted" else "direct"

            result = eval_swarm_matchup(
                defender_path=def_path,
                attacker_path=att_path,
                n_def=n_def,
                n_att=n_att,
                num_episodes=args.episodes,
                num_worlds=args.num_worlds,
                num_obstacles=args.obstacles,
                realworld=args.realworld,
                attacker_strategy=att_strategy,
            )

            i = result["intercepted"]
            t = result["target_reached"]
            pct = i / args.episodes * 100
            cell = f"| {i:2d}/{args.episodes} ({pct:3.0f}%)"
            if t > 0:
                cell += f" T:{t}"
            row += f"{cell:<{col_w}}"
            all_results[(def_name, att_name)] = result

        print(row)

    # Summary
    print(f"\n{'='*100}")
    print("  I=intercepted, T=target reached, pct=interception rate")
    print(f"{'='*100}\n")

    # Best generalist (highest minimum interception across all attackers)
    print("Best generalist (highest min interception across all attackers):")
    best_min = -1
    best_name = ""
    for def_path in def_paths:
        def_name = os.path.basename(def_path).replace(".zip", "")
        min_pct = 100
        for att_name in att_entries.keys():
            key = (def_name, att_name)
            if key in all_results:
                pct = all_results[key]["intercepted"] / args.episodes * 100
                min_pct = min(min_pct, pct)
        print(f"  {def_name}: min {min_pct:.0f}%")
        if min_pct > best_min:
            best_min = min_pct
            best_name = def_name

    if best_name:
        print(f"\n  >>> Best: {best_name} (min {best_min:.0f}%)")

    # Average performance
    print("\nAverage interception rate:")
    for def_path in def_paths:
        def_name = os.path.basename(def_path).replace(".zip", "")
        pcts = []
        for att_name in att_entries.keys():
            key = (def_name, att_name)
            if key in all_results:
                pcts.append(all_results[key]["intercepted"] / args.episodes * 100)
        if pcts:
            print(f"  {def_name}: avg {np.mean(pcts):.0f}%")


if __name__ == "__main__":
    main()
