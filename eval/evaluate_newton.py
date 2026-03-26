"""
Evaluate a trained defender drone model using Newton GPU physics.

Usage:
    python -m eval.evaluate_newton --model models/defender_newton_final.zip --episodes 50
    python -m eval.evaluate_newton --model models/best_model.zip --gui
"""

import argparse
import os
import time

import numpy as np
import warp as wp
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize

from envs.newton_intercept_env import NewtonInterceptVecEnv
from train.config import EnvConfig


def evaluate(args):
    wp.init()
    env_cfg = EnvConfig()

    # Single env for evaluation (sequential episodes)
    env = NewtonInterceptVecEnv(num_envs=1, config=env_cfg)

    # Load VecNormalize stats if available
    vec_norm_path = os.path.join(
        os.path.dirname(args.model), "vec_normalize_newton.pkl"
    )
    if os.path.exists(vec_norm_path):
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

    # Load model
    model = SAC.load(args.model)

    # --- Run episodes ---
    results = {
        "intercepted": 0,
        "target_reached": 0,
        "truncated": 0,
        "total_rewards": [],
        "episode_lengths": [],
    }

    print(f"\nEvaluating {args.model} for {args.episodes} episodes (Newton GPU)...")

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            done = dones[0]
            info = infos[0]
            total_reward += reward[0]
            steps += 1

            if args.gui:
                time.sleep(1.0 / env_cfg.ctrl_freq)

        # Record results
        results["total_rewards"].append(total_reward)
        results["episode_lengths"].append(steps)

        if info.get("intercepted", False):
            results["intercepted"] += 1
            outcome = "INTERCEPTED"
        elif info.get("target_reached", False):
            results["target_reached"] += 1
            outcome = "TARGET REACHED (fail)"
        else:
            results["truncated"] += 1
            outcome = "TRUNCATED"

        print(
            f"  Episode {ep+1:3d}/{args.episodes}: {outcome:20s} | "
            f"reward={total_reward:7.1f} | steps={steps:4d}"
        )

    env.close()

    # --- Summary ---
    n = args.episodes
    print(f"\n{'='*60}")
    print(f"  Results ({n} episodes)")
    print(f"{'='*60}")
    print(
        f"  Interception rate: {results['intercepted']/n*100:.1f}% "
        f"({results['intercepted']}/{n})"
    )
    print(
        f"  Target reached:    {results['target_reached']/n*100:.1f}% "
        f"({results['target_reached']}/{n})"
    )
    print(
        f"  Truncated:         {results['truncated']/n*100:.1f}% "
        f"({results['truncated']}/{n})"
    )
    print(
        f"  Avg reward:        {np.mean(results['total_rewards']):.1f} "
        f"(+/- {np.std(results['total_rewards']):.1f})"
    )
    print(
        f"  Avg episode len:   {np.mean(results['episode_lengths']):.0f} steps"
    )
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate defender drone (Newton GPU)"
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Path to saved model (.zip)"
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--gui", action="store_true", help="Slow down for visualization"
    )
    args = parser.parse_args()
    evaluate(args)
