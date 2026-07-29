"""
Ukraine scenario training: 7 FPV interceptors vs 5 Shahed-type attackers.

Progressive attacker curriculum — critical for stable training:
  Direct 7v5 training causes Q-function divergence (sparse positive reward flood).
  Starting from 7v1 then scaling to 7v5 avoids the coordination dead-lock.

Rewards: sparse outcome signals + Stage 1 closing-distance bootstrap shaping.
  Stage 1: +0.1/meter closing (bootstrap), +200 kamikaze intercept, +100 team win, -200 breach
  Stage 2+: outcome-only: +200 intercept, +100 win, -200 breach, -100 all-lost, -50 bad

Stages (progressive difficulty — attacker count AND environment complexity):
  1. 7v1, no hills, no wind — learn single intercept (guaranteed positive signal)
  2. 7v3, hills + light wind — scale up to partial swarm
  3. 7v5 full, hills + wind + obs noise — full swarm + sensor noise
  4. 7v5, flanking attackers — evasion warm-up
  5. 7v5, full realism: gusts + sensor noise + drone variance + mixed

Usage:
    python -m train.train_ukraine
    python -m train.train_ukraine --timesteps-per-stage 20000000 --num-worlds 512
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import warp as wp
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from envs.newton_swarm_env import NewtonSwarmEnv
from train.config import EnvConfig, TrainConfig


LOG_FILE = "logs/train_ukraine.log"


STAGES = [
    {
        "name": "stage1_baseline",
        "desc": "7v1, no hills, no wind — learn single intercept",
        "num_attackers": 1,            # Start simple: 7 defenders vs 1 attacker
        "num_obstacles": 0,
        "wind_enabled": False,
        "obs_noise_std": 0.0,
        "attacker_speed_scale": 0.8,
        "attacker_strategy": "direct",
        "sensor_range": 300.0,
        "approach_shaping_coeff": 0.1,  # Bootstrap: ~50/episode shaping vs 200 intercept
    },
    {
        "name": "stage2_hills_wind",
        "desc": "7v3, hills + light wind — scale up to partial swarm",
        "num_attackers": 3,            # Scale up: 7 defenders vs 3 attackers
        "num_obstacles": 5,
        "wind_enabled": True,
        "wind_max_force": 0.3,
        "obs_noise_std": 0.0,
        "attacker_speed_scale": 0.8,
        "attacker_strategy": "direct",
        "sensor_range": 300.0,
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

    # Apply stage overrides
    for key, val in stage.items():
        if key in ("name", "desc"):
            continue
        if hasattr(cfg, key):
            setattr(cfg, key, val)

    # Hills: place at ground level (obstacle_height is half-height of cylinder)
    # obstacle_height already set by apply_ukraine_scale; stages override num_obstacles
    cfg.num_obstacles = stage.get("num_obstacles", 0)

    # CLI override: --approach-shaping-coeff 0 disables shaping when resuming a trained model
    if args.approach_shaping_coeff is not None:
        cfg.approach_shaping_coeff = args.approach_shaping_coeff

    return cfg


def _add_grad_clipping(model, max_norm: float):
    """Monkey-patch actor/critic optimizer.step() to clip gradients first.

    SB3 2.7.1 has no built-in max_grad_norm for SAC. Extreme reward batches
    (e.g. -274 from shaping + breach) cause catastrophic Q-updates without this.
    """
    for name, opt, params in [
        ("actor",  model.actor.optimizer,  list(model.actor.parameters())),
        ("critic", model.critic.optimizer, list(model.critic.parameters())),
    ]:
        orig_step = opt.step
        def _make_step(orig, ps, nm=max_norm):
            def clipped_step(closure=None):
                torch.nn.utils.clip_grad_norm_(ps, nm)
                return orig(closure)
            return clipped_step
        opt.step = _make_step(orig_step, params)
    log(f"  Gradient clipping enabled: max_norm={max_norm}")


def train_stage(args, stage, stage_idx, resume_path=None):
    log(f"Stage {stage_idx+1}/5: {stage['desc']}")

    train_cfg = TrainConfig()
    train_cfg.batch_size = args.batch_size
    train_cfg.buffer_size = args.buffer_size

    env_cfg = build_env_cfg(args, stage)
    eval_cfg = build_env_cfg(args, stage)

    model_name = f"ukraine_{args.defenders}v{args.attackers}_{stage['name']}"

    print(f"\n{'='*60}")
    print(f"  Stage {stage_idx+1}/5: {stage['desc']}")
    print(f"  Config: {env_cfg.num_defenders}v{env_cfg.num_attackers} | Arena: {env_cfg.arena_bounds*2:.0f}m")
    print(f"  Detection range: {env_cfg.sensor_range:.0f}m | Hills: {env_cfg.num_obstacles}")
    print(f"  Timesteps: {args.timesteps_per_stage:,} | Worlds: {args.num_worlds}")
    print(f"  Batch: {train_cfg.batch_size} | Buffer: {train_cfg.buffer_size * args.defenders:,}")
    print(f"{'='*60}\n", flush=True)

    env = NewtonSwarmEnv(num_worlds=args.num_worlds, config=env_cfg)
    eval_env = NewtonSwarmEnv(num_worlds=max(4, args.num_worlds // 16), config=eval_cfg)

    # SB3 callbacks use n_calls (env steps, not timesteps).
    # With n_envs=num_worlds*N, divide timestep targets by n_envs.
    n_agent_envs = env.n_envs
    ckpt_freq_calls = max(1, args.timesteps_per_stage // 10 // n_agent_envs)
    eval_freq_calls = max(1, args.timesteps_per_stage // 20 // n_agent_envs)
    log(f"  n_envs={n_agent_envs} | eval every {eval_freq_calls} calls = {eval_freq_calls*n_agent_envs/1e6:.1f}M steps | "
        f"ckpt every {ckpt_freq_calls} calls = {ckpt_freq_calls*n_agent_envs/1e6:.1f}M steps")

    gradient_steps = 64
    train_freq = 4

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

    print(f"  Creating SAC model (buffer={train_cfg.buffer_size * args.defenders:,})...", flush=True)
    if resume_path and os.path.exists(resume_path):
        log(f"  Resuming from {resume_path}")
        model = SAC.load(
            resume_path,
            env=env,
            device="cuda",
            verbose=1,
            tensorboard_log=f"logs/tensorboard",
        )
        model.set_env(env)
        model.gradient_steps = gradient_steps
        # Do NOT override model.train_freq — SAC.load() restores it as a TrainFreq namedtuple;
        # overwriting with a plain int fails the isinstance assertion in learn().
        model.batch_size = train_cfg.batch_size
        model.learning_starts = max(1000, train_cfg.batch_size * 2)
        # Reset optimizer state and apply new learning rate to prevent divergence from stale momentum
        new_lr = args.learning_rate
        for opt in [model.actor.optimizer, model.critic.optimizer]:
            opt.state.clear()
            for pg in opt.param_groups:
                pg["lr"] = new_lr
        log(f"  Optimizer reset: lr={new_lr}, Adam state cleared")
        _add_grad_clipping(model, max_norm=1.0)
    else:
        log(f"  Starting from scratch (no resume path)")
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            batch_size=train_cfg.batch_size,
            buffer_size=train_cfg.buffer_size * args.defenders,
            learning_starts=max(1000, train_cfg.batch_size * 10),
            gamma=0.99,
            tau=0.005,
            ent_coef="0.2",            # Fixed entropy — auto collapsed to 0.003 previously
            policy_kwargs={"net_arch": [512, 512, 256]},
            gradient_steps=gradient_steps,
            train_freq=train_freq,
            device="cuda",
            verbose=1,
            tensorboard_log=f"logs/tensorboard",
        )
    print("  SAC model ready. Starting training...", flush=True)

    model.learn(
        total_timesteps=args.timesteps_per_stage,
        callback=callbacks,
        reset_num_timesteps=False,
        tb_log_name=model_name,
        progress_bar=False,
    )

    # Use best eval model for warm-starting next stage; fall back to final if missing
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
    parser = argparse.ArgumentParser(description="Ukraine Scenario Training")
    parser.add_argument("--defenders", type=int, default=7)
    parser.add_argument("--attackers", type=int, default=5)
    parser.add_argument("--num-worlds", type=int, default=1024)
    parser.add_argument("--timesteps-per-stage", type=int, default=40_000_000)
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from model path (must match obs_dim)")
    parser.add_argument("--start-stage", type=int, default=0, help="Start from stage index (0-4)")
    parser.add_argument("--k-opponents", type=int, default=3)
    parser.add_argument("--k-teammates", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="SAC learning rate (use 1e-4 when resuming to prevent divergence)")
    parser.add_argument("--approach-shaping-coeff", type=float, default=None,
                        help="Override stage approach_shaping_coeff (0.0 = disable when resuming trained model)")
    args = parser.parse_args()

    wp.init()
    log(f"Starting Ukraine training: {args.defenders}v{args.attackers}")
    log(f"  {args.timesteps_per_stage:,} steps/stage × {len(STAGES)} stages = "
        f"{args.timesteps_per_stage * len(STAGES):,} total")

    resume = args.resume
    for i, stage in enumerate(STAGES[args.start_stage:], start=args.start_stage):
        log(f"Stage {i+1}/5: {stage['desc']} (attempt 1)")
        resume = train_stage(args, stage, i, resume_path=resume if i == args.start_stage else resume)

    log("=" * 60)
    log(f"Ukraine training complete! Final model: {resume}")
    log("=" * 60)


if __name__ == "__main__":
    main()
