"""
Swarm Training — Train N defenders vs M attackers.

Uses CTDE (Centralized Training, Decentralized Execution) with parameter sharing.
One SAC model controls all defenders. Each defender gets its own observation.

Usage:
    # 2v2 against scripted attackers
    python -m train.train_swarm --defenders 2 --attackers 2 --num-worlds 128

    # 2v2 with domain randomization (wind + noise)
    python -m train.train_swarm --defenders 2 --attackers 2 --wind --noise 0.02

    # 3v3 with obstacles
    python -m train.train_swarm --defenders 3 --attackers 3 --obstacles 3

    # Resume from 1v1 model (weight transfer)
    python -m train.train_swarm --defenders 2 --attackers 2 --resume models/defender_best_v5.zip
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


def make_env(num_worlds, cfg):
    """Create swarm environment."""
    return NewtonSwarmEnv(num_worlds=num_worlds, config=cfg)


def train_swarm(args):
    wp.init()

    train_cfg = TrainConfig()

    env_cfg = EnvConfig(
        mode="intercept",
        num_defenders=args.defenders,
        num_attackers=args.attackers,
        num_obstacles=args.obstacles,
        wind_enabled=args.wind,
        wind_max_force=args.wind_force,
        obs_noise_std=args.noise,
        attacker_strategy=args.strategy,
    )

    print(f"\n{'='*60}")
    print(f"  Swarm Training: {args.defenders}v{args.attackers}")
    print(f"  Worlds: {args.num_worlds} | Obstacles: {args.obstacles}")
    print(f"  Wind: {'ON' if args.wind else 'OFF'} | Noise: {args.noise}")
    print(f"  Strategy: {args.strategy}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"{'='*60}\n")

    # Training env
    env = make_env(args.num_worlds, env_cfg)

    # Eval env (smaller)
    eval_cfg = EnvConfig(
        mode="intercept",
        num_defenders=args.defenders,
        num_attackers=args.attackers,
        num_obstacles=args.obstacles,
    )
    eval_env = make_env(max(4, args.num_worlds // 16), eval_cfg)

    # Model
    model_kwargs = dict(
        policy="MlpPolicy",
        env=env,
        learning_rate=train_cfg.learning_rate,
        batch_size=train_cfg.batch_size,
        buffer_size=train_cfg.buffer_size * args.defenders,  # scale with N
        learning_starts=train_cfg.learning_starts,
        gamma=train_cfg.gamma,
        tau=train_cfg.tau,
        ent_coef=train_cfg.ent_coef,
        policy_kwargs={"net_arch": train_cfg.net_arch},
        tensorboard_log=train_cfg.tensorboard_log,
        verbose=1,
        seed=train_cfg.seed,
    )

    if args.resume:
        print(f"Loading weights from {args.resume}")
        model = SAC.load(args.resume, env=env)
        # Note: weight transfer from 1v1 (35-dim) to swarm (57-dim) requires
        # architecture match. If dimensions differ, train from scratch.
    else:
        model = SAC(**model_kwargs)

    # Callbacks
    tag = f"swarm_{args.defenders}v{args.attackers}"
    checkpoint_cb = CheckpointCallback(
        save_freq=train_cfg.checkpoint_freq,
        save_path=train_cfg.model_dir,
        name_prefix=tag,
    )

    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=100,
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

    # Train
    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name=tag,
    )

    # Save final model
    final_path = os.path.join(train_cfg.model_dir, f"{tag}_final")
    model.save(final_path)
    print(f"\nSaved: {final_path}.zip")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Swarm Drone Training")
    parser.add_argument("--defenders", type=int, default=2, help="Number of defenders")
    parser.add_argument("--attackers", type=int, default=2, help="Number of attackers")
    parser.add_argument("--num-worlds", type=int, default=128, help="Parallel GPU worlds")
    parser.add_argument("--timesteps", type=int, default=2_000_000, help="Total timesteps")
    parser.add_argument("--obstacles", type=int, default=3, help="Obstacles per world")
    parser.add_argument("--wind", action="store_true", help="Enable wind randomization")
    parser.add_argument("--wind-force", type=float, default=0.3, help="Max wind force (N)")
    parser.add_argument("--noise", type=float, default=0.0, help="Observation noise std")
    parser.add_argument("--strategy", type=str, default="direct",
                        choices=["direct", "flanking", "mixed"],
                        help="Scripted attacker strategy")
    parser.add_argument("--resume", type=str, default="", help="Resume from model path")
    args = parser.parse_args()

    train_swarm(args)
