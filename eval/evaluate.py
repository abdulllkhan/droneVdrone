"""
Evaluate a trained defender drone model.

Usage:
    python -m eval.evaluate --model models/defender_sac_final.zip --gui
    python -m eval.evaluate --model models/best_model.zip --episodes 50
"""

import argparse
import os
import time

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

from envs.intercept_env import DroneInterceptEnv
from train.config import EnvConfig


def evaluate(args):
    env_cfg = EnvConfig()

    # Create environment
    raw_env = DroneInterceptEnv(gui=args.gui, config=env_cfg)

    # Load model
    model = SAC.load(args.model)

    # Load VecNormalize stats if available
    vec_norm_path = os.path.join(os.path.dirname(args.model), "vec_normalize.pkl")
    use_vec_norm = os.path.exists(vec_norm_path)

    if use_vec_norm:
        # Wrap in DummyVecEnv + VecNormalize for consistent normalization
        vec_env = DummyVecEnv([lambda: raw_env])
        vec_env = VecNormalize.load(vec_norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        env = vec_env
    else:
        env = raw_env

    # --- Run episodes ---
    results = {
        "intercepted": 0,
        "target_reached": 0,
        "truncated": 0,
        "total_rewards": [],
        "episode_lengths": [],
        "intercept_distances": [],
    }

    print(f"\nEvaluating {args.model} for {args.episodes} episodes...")
    if args.gui:
        print("(GUI mode — episodes run in real-time)\n")

    for ep in range(args.episodes):
        if use_vec_norm:
            obs = env.reset()
        else:
            obs, info = env.reset()

        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)

            if use_vec_norm:
                obs, reward, dones, infos = env.step(action)
                done = dones[0]
                info = infos[0]
                reward = reward[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            total_reward += reward
            steps += 1

            if args.gui:
                time.sleep(1.0 / env_cfg.ctrl_freq)

        # Record results
        results["total_rewards"].append(total_reward)
        results["episode_lengths"].append(steps)
        results["intercept_distances"].append(info.get("distance_to_attacker", -1))

        if info.get("intercepted", False):
            results["intercepted"] += 1
            outcome = "INTERCEPTED"
        elif info.get("target_reached", False):
            results["target_reached"] += 1
            outcome = "TARGET REACHED (fail)"
        else:
            results["truncated"] += 1
            outcome = "TRUNCATED"

        print(f"  Episode {ep+1:3d}/{args.episodes}: {outcome:20s} | "
              f"reward={total_reward:7.1f} | steps={steps:4d} | "
              f"dist={info.get('distance_to_attacker', -1):.3f}")

    env.close()

    # --- Summary ---
    n = args.episodes
    print(f"\n{'='*60}")
    print(f"  Results ({n} episodes)")
    print(f"{'='*60}")
    print(f"  Interception rate: {results['intercepted']/n*100:.1f}% "
          f"({results['intercepted']}/{n})")
    print(f"  Target reached:    {results['target_reached']/n*100:.1f}% "
          f"({results['target_reached']}/{n})")
    print(f"  Truncated:         {results['truncated']/n*100:.1f}% "
          f"({results['truncated']}/{n})")
    print(f"  Avg reward:        {np.mean(results['total_rewards']):.1f} "
          f"(+/- {np.std(results['total_rewards']):.1f})")
    print(f"  Avg episode len:   {np.mean(results['episode_lengths']):.0f} steps")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate defender drone")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to saved model (.zip)")
    parser.add_argument("--episodes", type=int, default=20,
                        help="Number of evaluation episodes")
    parser.add_argument("--gui", action="store_true",
                        help="Show PyBullet GUI visualization")
    args = parser.parse_args()
    evaluate(args)
