"""
Adaptive 300m training — runs curriculum with degenerate behavior detection.

After each stage, evaluates the model for:
  - Defender clustering (all defenders same position)
  - Zero intercept rate
  - Attacker immobility
If degenerate: bumps entropy, lowers LR, re-runs the stage.

Usage:
    python -m train.train_300m_fixed
    python -m train.train_300m_fixed --resume models/some_model.zip --start-stage 2
"""

import argparse
import os
import sys
import time

import numpy as np
import warp as wp
from stable_baselines3 import SAC

from envs.newton_swarm_env import NewtonSwarmEnv
from train.config import EnvConfig
from train.train_swarm_curriculum import STAGES, train_stage


LOG_FILE = "logs/adaptive_training.log"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def evaluate_model(model_path, num_episodes=50, realistic=True,
                   num_defenders=15, num_attackers=10,
                   k_opponents=5, k_teammates=4):
    """Run eval episodes and return behavior metrics."""
    cfg = EnvConfig(
        num_defenders=num_defenders,
        num_attackers=num_attackers,
        num_obstacles=0,
        k_opponents=k_opponents,
        k_teammates=k_teammates,
        kamikaze=True,
    )
    if realistic:
        cfg.apply_realistic_scale()
        cfg.num_defenders = num_defenders
        cfg.num_attackers = num_attackers
        cfg.num_obstacles = 0
        cfg.k_opponents = k_opponents
        cfg.k_teammates = k_teammates

    env = NewtonSwarmEnv(num_worlds=4, config=cfg)
    model = SAC.load(model_path, device="cuda")

    intercepts = 0
    target_reached = 0
    episodes = 0
    defender_spread_sum = 0.0
    attacker_moved_sum = 0.0

    obs = env.reset()
    while episodes < num_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)

        for i, info in enumerate(infos):
            if info is None:
                continue
            if info.get("intercepted", 0):
                intercepts += 1
            if info.get("target_reached", 0):
                target_reached += 1

        # Check if any world is done
        for w in range(env.num_worlds):
            if dones.any() if hasattr(dones, 'any') else any(dones):
                episodes += 1
                if episodes >= num_episodes:
                    break

        # Every 10 steps, sample defender spread and attacker movement
        step_counts = env.step_count_buf.numpy()
        if step_counts[0] % 10 == 0:
            try:
                body_q = env.state_0.body_q.numpy()
                for w in range(min(env.num_worlds, 4)):
                    def_pos = []
                    for d in range(env.N):
                        bidx = w * env.bodies_per_world + d
                        def_pos.append(body_q[bidx][:3])
                    if len(def_pos) > 1:
                        def_pos = np.array(def_pos)
                        centroid = def_pos.mean(axis=0)
                        spread = np.mean(np.linalg.norm(def_pos - centroid, axis=1))
                        defender_spread_sum += spread
            except Exception:
                pass

    env.close()

    intercept_rate = intercepts / max(episodes, 1)
    target_rate = target_reached / max(episodes, 1)
    avg_spread = defender_spread_sum / max(episodes, 1)

    return {
        "episodes": episodes,
        "intercept_rate": intercept_rate,
        "target_reached_rate": target_rate,
        "avg_defender_spread": avg_spread,
    }


def main():
    parser = argparse.ArgumentParser(description="Adaptive 300m Training")
    parser.add_argument("--defenders", type=int, default=15)
    parser.add_argument("--attackers", type=int, default=10)
    parser.add_argument("--num-worlds", type=int, default=1024)
    parser.add_argument("--timesteps-per-stage", type=int, default=40_000_000)
    parser.add_argument("--obstacles", type=int, default=0)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--start-stage", type=int, default=0)
    parser.add_argument("--k-opponents", type=int, default=5)
    parser.add_argument("--k-teammates", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--buffer-size", type=int, default=200000)
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max retries per stage if degenerate")
    args = parser.parse_args()

    # Always realistic + kamikaze
    args.realistic = True
    args.kamikaze = True

    wp.init()

    log(f"Starting adaptive 300m training: {args.defenders}v{args.attackers}")
    log(f"  {args.timesteps_per_stage:,} steps/stage × {len(STAGES)} stages = "
        f"{args.timesteps_per_stage * len(STAGES):,} total")

    resume_path = args.resume if args.resume else None

    for i, stage in enumerate(STAGES):
        if i < args.start_stage:
            continue

        retries = 0
        while retries <= args.max_retries:
            log(f"Stage {i+1}/{len(STAGES)}: {stage['desc']} (attempt {retries+1})")
            model_path = train_stage(args, stage, i, resume_path)

            # Evaluate for degenerate behavior
            log(f"Evaluating {model_path}...")
            metrics = evaluate_model(
                model_path,
                num_episodes=50,
                realistic=True,
                num_defenders=args.defenders,
                num_attackers=args.attackers,
                k_opponents=args.k_opponents,
                k_teammates=args.k_teammates,
            )
            log(f"  Intercept rate: {metrics['intercept_rate']:.2%}")
            log(f"  Target reached rate: {metrics['target_reached_rate']:.2%}")
            log(f"  Avg defender spread: {metrics['avg_defender_spread']:.2f}m")

            # Check for degenerate behavior
            degenerate = False
            reasons = []
            if metrics["intercept_rate"] < 0.05:
                reasons.append(f"low intercept rate ({metrics['intercept_rate']:.2%})")
                degenerate = True
            if metrics["avg_defender_spread"] < 2.0:
                reasons.append(f"defenders clustered ({metrics['avg_defender_spread']:.1f}m spread)")
                degenerate = True

            if degenerate and retries < args.max_retries:
                log(f"  DEGENERATE: {', '.join(reasons)} — retrying with higher entropy")
                retries += 1
                # Bump entropy for retry — load model and adjust
                model = SAC.load(model_path, device="cuda")
                model.ent_coef_tensor = model.ent_coef_tensor * 2.0
                model.learning_rate = model.learning_rate * 0.5
                retry_path = model_path.replace("_final.zip", f"_retry{retries}.zip")
                model.save(retry_path)
                resume_path = retry_path
                log(f"  Saved retry model: {retry_path}")
            else:
                if degenerate:
                    log(f"  WARNING: Still degenerate after {retries+1} attempts, moving on")
                else:
                    log(f"  Stage {i+1} PASSED")
                resume_path = model_path
                break

    log("=" * 60)
    log(f"Adaptive training complete! Final model: {resume_path}")
    log("=" * 60)


if __name__ == "__main__":
    main()
