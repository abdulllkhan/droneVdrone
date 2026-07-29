"""
Ukraine scenario training with PPO: 7 FPV interceptors vs 5 Shahed-type attackers.

Switched from SAC to PPO after consistent Q-function divergence with sparse kamikaze rewards.
SAC's off-policy replay buffer caused catastrophic Q-updates when bad experiences dominated.
PPO avoids this: on-policy rollouts, GAE for sparse signals, built-in max_grad_norm.

Rewards: sparse outcome signals + Stage 1 closing-distance bootstrap shaping.
  Stage 1: +0.1/meter closing (bootstrap), +200 kamikaze intercept, +100 team win, -200 breach
  Stage 2+: outcome-only: +200 intercept, +100 win, -200 breach, -100 all-lost, -50 bad

Stages (progressive difficulty):
  1. 7v1, no hills, no wind — learn single intercept (guaranteed positive signal)
  2. 7v3, hills + light wind — scale up to partial swarm
  3. 7v5 full, hills + wind + obs noise — full swarm + sensor noise
  4. 7v5, flanking attackers — evasion warm-up
  5. 7v5, full realism: gusts + sensor noise + drone variance + mixed

Usage:
    python -m train.train_ukraine_ppo
    python -m train.train_ukraine_ppo --timesteps-per-stage 20000000 --num-worlds 512
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import warp as wp
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from envs.newton_swarm_env import NewtonSwarmEnv
from train.config import EnvConfig, TrainConfig


LOG_FILE = "logs/train_ukraine_7v5.log"


STAGES = [
    {
        "name": "stage1_baseline",
        "desc": "7v1, no hills, no wind — learn single intercept",
        "num_attackers": 1,
        "num_obstacles": 0,
        "wind_enabled": False,
        "obs_noise_std": 0.0,
        "attacker_speed_scale": 0.8,
        "attacker_strategy": "direct",
        "sensor_range": 300.0,
        "approach_shaping_coeff": 0.05,  # Small bootstrap for cold-start
    },
    {
        "name": "stage2_hills_wind",
        "desc": "7v3, hills + light wind — scale up to partial swarm",
        "num_attackers": 3,
        "num_obstacles": 5,
        "wind_enabled": True,
        "wind_max_force": 0.3,
        "obs_noise_std": 0.0,
        "attacker_speed_scale": 0.8,
        "attacker_strategy": "direct",
        "sensor_range": 300.0,
        "approach_shaping_coeff": 0.02,  # Tiny shaping for 7v3 cold-start
    },
    {
        "name": "stage3_noise",
        "desc": "7v5 full, hills + wind + sensor noise",
        "num_obstacles": 5,
        "wind_enabled": True,
        "wind_max_force": 0.5,
        "obs_noise_std": 0.02,
        "attacker_speed_scale": 0.9,
        "attacker_strategy": "direct",
        "sensor_range": 250.0,
    },
    {
        "name": "stage4_evasion",
        "desc": "7v5, hills + wind + noise + flanking attackers",
        "num_obstacles": 5,
        "wind_enabled": True,
        "wind_max_force": 0.5,
        "obs_noise_std": 0.02,
        "attacker_speed_scale": 1.0,
        "attacker_strategy": "flanking",
        "sensor_range": 250.0,
    },
    {
        "name": "stage5_realworld",
        "desc": "7v5, full realism: gusts + sensor noise + drone variance + mixed attackers",
        "num_obstacles": 5,
        "wind_enabled": True,
        "wind_max_force": 0.5,
        "wind_gust_prob": 0.01,
        "wind_gust_force": 1.5,
        "obs_noise_std": 0.03,
        "attacker_speed_scale": 1.0,
        "attacker_strategy": "mixed",
        "sensor_range": 200.0,
        "drone_mass_variance": 0.1,
        "attacker_speed_variance": 0.1,
    },
]


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def build_env_cfg(args, stage):
    cfg = EnvConfig()
    cfg.apply_ukraine_scale()
    cfg.num_defenders = args.defenders
    cfg.num_attackers = args.attackers
    cfg.k_opponents = args.k_opponents
    cfg.k_teammates = args.k_teammates
    cfg.kamikaze = True

    for key, val in stage.items():
        if key in ("name", "desc"):
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    cfg.num_obstacles = stage.get("num_obstacles", 0)

    if args.approach_shaping_coeff is not None:
        cfg.approach_shaping_coeff = args.approach_shaping_coeff

    return cfg


def train_stage(args, stage, stage_idx, resume_path=None):
    log(f"Stage {stage_idx+1}/5: {stage['desc']}")

    train_cfg = TrainConfig()
    env_cfg = build_env_cfg(args, stage)
    eval_cfg = build_env_cfg(args, stage)

    model_name = f"ukraine_ppo_{args.defenders}v{args.attackers}_{stage['name']}"

    print(f"\n{'='*60}")
    print(f"  Stage {stage_idx+1}/5: {stage['desc']}")
    print(f"  Config: {env_cfg.num_defenders}v{env_cfg.num_attackers} | Arena: {env_cfg.arena_bounds*2:.0f}m")
    print(f"  Detection range: {env_cfg.sensor_range:.0f}m | Hills: {env_cfg.num_obstacles}")
    print(f"  Timesteps: {args.timesteps_per_stage:,} | Worlds: {args.num_worlds}")
    print(f"  Batch: {args.batch_size} | n_steps: {args.n_steps} | n_epochs: {args.n_epochs}")
    print(f"  Shaping coeff: {env_cfg.approach_shaping_coeff}")
    print(f"{'='*60}\n", flush=True)

    env = NewtonSwarmEnv(num_worlds=args.num_worlds, config=env_cfg)
    eval_env = NewtonSwarmEnv(num_worlds=max(4, args.num_worlds // 16), config=eval_cfg)

    n_agent_envs = env.n_envs
    rollout_size = args.n_steps * n_agent_envs
    ckpt_freq_calls = max(1, args.timesteps_per_stage // 10 // n_agent_envs)
    eval_freq_calls = max(1, args.timesteps_per_stage // 20 // n_agent_envs)
    log(f"  n_envs={n_agent_envs} | rollout={rollout_size:,} steps | "
        f"eval every {eval_freq_calls} calls = {eval_freq_calls*n_agent_envs/1e6:.1f}M steps | "
        f"ckpt every {ckpt_freq_calls} calls = {ckpt_freq_calls*n_agent_envs/1e6:.1f}M steps")

    best_model_dir = f"models/{model_name}_best"
    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(f"models/{model_name}_ckpt", exist_ok=True)

    callbacks = [
        CheckpointCallback(
            save_freq=ckpt_freq_calls,
            save_path=f"models/{model_name}_ckpt/",
            name_prefix=model_name,
        ),
        EvalCallback(
            eval_env,
            eval_freq=eval_freq_calls,
            n_eval_episodes=20,
            best_model_save_path=best_model_dir,
            log_path=f"logs/eval_{model_name}/",
            deterministic=True,
        ),
    ]

    print(f"  Creating PPO model...", flush=True)
    if resume_path and os.path.exists(resume_path):
        log(f"  Resuming from {resume_path}")
        model = PPO.load(
            resume_path,
            env=env,
            device="cuda",
            verbose=1,
            tensorboard_log="logs/tensorboard",
        )
        model.set_env(env)
        model.batch_size = args.batch_size
        model.n_steps = args.n_steps
        model.n_epochs = args.n_epochs
        # Reset optimizer to prevent stale momentum from prior stage
        new_lr = args.learning_rate
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = new_lr
        model.policy.optimizer.state.clear()
        log(f"  Optimizer reset: lr={new_lr}, Adam state cleared")
    else:
        log(f"  Starting from scratch")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            max_grad_norm=0.5,
            policy_kwargs={"net_arch": [512, 512, 256]},
            device="cuda",
            verbose=1,
            tensorboard_log="logs/tensorboard",
        )
    print("  PPO model ready. Starting training...", flush=True)

    model.learn(
        total_timesteps=args.timesteps_per_stage,
        callback=callbacks,
        reset_num_timesteps=True,  # Each stage gets a full timesteps_per_stage budget
        tb_log_name=model_name,
        progress_bar=False,
    )

    best_model_path = os.path.join(best_model_dir, "best_model.zip")
    final_path = f"models/{model_name}_final.zip"
    model.save(final_path)
    log(f"Saved final: {final_path}")

    if os.path.exists(best_model_path):
        log(f"Best model at: {best_model_path} (used for next stage warm-start)")
        warm_start = best_model_path
    else:
        log(f"No best model saved; using final for next stage warm-start")
        warm_start = final_path

    env.close()
    eval_env.close()
    return warm_start


def main():
    parser = argparse.ArgumentParser(description="Ukraine Scenario Training (PPO)")
    parser.add_argument("--defenders", type=int, default=7)
    parser.add_argument("--attackers", type=int, default=5)
    parser.add_argument("--num-worlds", type=int, default=1024)
    parser.add_argument("--timesteps-per-stage", type=int, default=40_000_000)
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from model path (PPO checkpoint)")
    parser.add_argument("--start-stage", type=int, default=0, help="Start from stage index (0-4)")
    parser.add_argument("--k-opponents", type=int, default=3)
    parser.add_argument("--k-teammates", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-steps", type=int, default=512,
                        help="PPO rollout steps per env (512 × 7168 envs = 3.7M per rollout)")
    parser.add_argument("--n-epochs", type=int, default=10,
                        help="PPO gradient epochs per rollout")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="Entropy coefficient (exploration bonus)")
    parser.add_argument("--approach-shaping-coeff", type=float, default=None,
                        help="Override stage approach_shaping_coeff (0.0 = disable)")
    args = parser.parse_args()

    wp.init()
    log(f"Starting Ukraine PPO training: {args.defenders}v{args.attackers}")
    log(f"  {args.timesteps_per_stage:,} steps/stage × {len(STAGES)} stages = "
        f"{args.timesteps_per_stage * len(STAGES):,} total")
    log(f"  PPO: n_steps={args.n_steps}, batch={args.batch_size}, epochs={args.n_epochs}, "
        f"ent_coef={args.ent_coef}")

    resume = args.resume
    for i, stage in enumerate(STAGES[args.start_stage:], start=args.start_stage):
        log(f"Stage {i+1}/5: {stage['desc']}")
        resume = train_stage(args, stage, i, resume_path=resume if i == args.start_stage else resume)

    log("=" * 60)
    log(f"Ukraine PPO training complete! Final model: {resume}")
    log("=" * 60)


if __name__ == "__main__":
    main()
