"""
Swarm Curriculum Training — Progressive difficulty scaling.

Trains swarm defenders through increasing difficulty:
  Stage 1: NvM with static obstacles, no wind
  Stage 2: NvM with wind perturbation
  Stage 3: NvM with dynamic obstacles + wind
  Stage 4: NvM with faster attackers + all randomization

Each stage loads weights from the previous stage.

Usage:
    python -m train.train_swarm_curriculum --defenders 2 --attackers 2 --num-worlds 128
    python -m train.train_swarm_curriculum --defenders 3 --attackers 2 --num-worlds 64 --timesteps-per-stage 1000000
"""

import argparse
import os

import warp as wp
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)

from envs.newton_swarm_env import NewtonSwarmEnv
from train.config import TrainConfig, EnvConfig


STAGES = [
    {
        "name": "stage1_static",
        "desc": "Static obstacles, no wind",
        "wind_enabled": False,
        "obstacle_dynamic": False,
        "obs_noise_std": 0.0,
        "attacker_speed_scale": 0.8,
    },
    {
        "name": "stage2_wind",
        "desc": "Static obstacles + wind",
        "wind_enabled": True,
        "wind_max_force": 0.3,
        "obstacle_dynamic": False,
        "obs_noise_std": 0.0,
        "attacker_speed_scale": 0.8,
    },
    {
        "name": "stage3_dynamic",
        "desc": "Dynamic obstacles + wind + noise + flanking",
        "wind_enabled": True,
        "wind_max_force": 0.3,
        "obstacle_dynamic": True,
        "obstacle_max_speed": 0.3,
        "obs_noise_std": 0.02,
        "attacker_speed_scale": 0.8,
        "attacker_strategy": "flanking",
    },
    {
        "name": "stage4_hard",
        "desc": "All randomization + faster attackers + mixed strategy",
        "wind_enabled": True,
        "wind_max_force": 0.5,
        "obstacle_dynamic": True,
        "obstacle_max_speed": 0.5,
        "obs_noise_std": 0.03,
        "attacker_speed_scale": 1.0,
        "attacker_strategy": "mixed",
    },
    {
        "name": "stage5_realworld",
        "desc": "Real-world: gusts + sensor limits + drone variance + speed variance",
        "wind_enabled": True,
        "wind_max_force": 0.5,
        "wind_gust_prob": 0.03,
        "wind_gust_force": 1.0,
        "obstacle_dynamic": True,
        "obstacle_max_speed": 0.5,
        "obs_noise_std": 0.03,
        "attacker_speed_scale": 1.0,
        "attacker_strategy": "mixed",
        "sensor_range": 5.0,
        "attacker_speed_variance": 0.15,
        "drone_mass_variance": 0.1,
    },
]


def train_stage(args, stage, stage_idx, resume_path=None):
    train_cfg = TrainConfig()
    if args.batch_size > 0:
        train_cfg.batch_size = args.batch_size
    if args.buffer_size > 0:
        train_cfg.buffer_size = args.buffer_size

    env_kwargs = dict(
        mode="intercept",
        num_defenders=args.defenders,
        num_attackers=args.attackers,
        num_obstacles=args.obstacles,
        k_opponents=args.k_opponents,
        k_teammates=args.k_teammates,
        kamikaze=args.kamikaze,
    )
    # Apply stage-specific settings
    for k, v in stage.items():
        if k not in ("name", "desc"):
            env_kwargs[k] = v

    env_cfg = EnvConfig(**env_kwargs)
    tag = f"swarm_{args.defenders}v{args.attackers}_{stage['name']}"

    print(f"\n{'='*60}")
    print(f"  Stage {stage_idx+1}/{len(STAGES)}: {stage['desc']}")
    print(f"  Config: {args.defenders}v{args.attackers} | Kamikaze: {'ON' if args.kamikaze else 'OFF'} | {tag}")
    if resume_path:
        print(f"  Resume: {resume_path}")
    print(f"  Timesteps: {args.timesteps_per_stage:,}")
    print(f"  Worlds: {args.num_worlds} | Batch: {train_cfg.batch_size} | Buffer: {train_cfg.buffer_size}")
    print(f"{'='*60}\n")

    env = NewtonSwarmEnv(num_worlds=args.num_worlds, config=env_cfg)

    # Eval env (clean — no randomization)
    eval_cfg = EnvConfig(
        num_defenders=args.defenders,
        num_attackers=args.attackers,
        num_obstacles=args.obstacles,
        k_opponents=args.k_opponents,
        k_teammates=args.k_teammates,
        kamikaze=args.kamikaze,
    )
    eval_env = NewtonSwarmEnv(num_worlds=max(4, args.num_worlds // 16), config=eval_cfg)

    # Maximize GPU utilization on DGX Spark (1536 CUDA cores, 128GB unified memory):
    # SB3's replay buffer is CPU-bound, so we push GPU work via:
    # 1) Large gradient_steps — many SGD updates per env.step() (GPU backprop)
    # 2) Large batch_size — saturate GPU cores per gradient step
    # 3) Collect multiple env steps before training (train_freq)
    gradient_steps = 64  # 64 gradient updates per training round → GPU stays busy
    train_freq = 4       # Collect 4 env steps (4*num_worlds transitions) between training rounds

    from stable_baselines3.common.type_aliases import TrainFreq, TrainFrequencyUnit

    if resume_path:
        model = SAC.load(resume_path, env=env, device="cuda")
        model.gradient_steps = gradient_steps
        model.train_freq = TrainFreq(train_freq, TrainFrequencyUnit.STEP)
    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=train_cfg.learning_rate,
            batch_size=train_cfg.batch_size,
            buffer_size=train_cfg.buffer_size * args.defenders,
            learning_starts=train_cfg.learning_starts,
            gamma=train_cfg.gamma,
            tau=train_cfg.tau,
            ent_coef=train_cfg.ent_coef,
            gradient_steps=gradient_steps,
            train_freq=(train_freq, "step"),
            policy_kwargs={"net_arch": train_cfg.net_arch},
            tensorboard_log=train_cfg.tensorboard_log,
            verbose=1,
            seed=train_cfg.seed,
            device="cuda",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=train_cfg.checkpoint_freq,
        save_path=train_cfg.model_dir,
        name_prefix=tag,
    )
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=80,
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=train_cfg.model_dir,
        log_path=train_cfg.log_dir,
        eval_freq=train_cfg.eval_freq,
        n_eval_episodes=train_cfg.eval_episodes,
        callback_after_eval=stop_cb,
        verbose=1,
    )

    model.learn(
        total_timesteps=args.timesteps_per_stage,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name=tag,
    )

    final_path = os.path.join(train_cfg.model_dir, f"{tag}_final")
    model.save(final_path)
    print(f"\nSaved: {final_path}.zip")

    env.close()
    eval_env.close()
    return f"{final_path}.zip"


def main():
    parser = argparse.ArgumentParser(description="Swarm Curriculum Training")
    parser.add_argument("--defenders", type=int, default=2)
    parser.add_argument("--attackers", type=int, default=2)
    parser.add_argument("--num-worlds", type=int, default=128)
    parser.add_argument("--timesteps-per-stage", type=int, default=1_000_000)
    parser.add_argument("--obstacles", type=int, default=3)
    parser.add_argument("--resume", type=str, default="", help="Resume from model")
    parser.add_argument("--start-stage", type=int, default=0, help="Start from stage (0-4)")
    parser.add_argument("--k-opponents", type=int, default=3, help="K nearest opponents to observe")
    parser.add_argument("--k-teammates", type=int, default=2, help="K nearest teammates to observe")
    parser.add_argument("--kamikaze", action="store_true", default=True, help="Defender destroyed on interception")
    parser.add_argument("--no-kamikaze", dest="kamikaze", action="store_false", help="Disable kamikaze")
    parser.add_argument("--batch-size", type=int, default=0, help="Override batch size (0 = use config default)")
    parser.add_argument("--buffer-size", type=int, default=0, help="Override buffer size (0 = use config default)")
    args = parser.parse_args()

    wp.init()

    resume_path = args.resume if args.resume else None

    for i, stage in enumerate(STAGES):
        if i < args.start_stage:
            continue
        resume_path = train_stage(args, stage, i, resume_path)

    print(f"\n{'='*60}")
    print(f"  Curriculum complete! Final model: {resume_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
