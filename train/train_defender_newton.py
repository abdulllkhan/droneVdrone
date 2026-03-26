"""
Train a defender drone using SAC with GPU-accelerated Newton physics.

Usage:
    python -m train.train_defender_newton
    python -m train.train_defender_newton --timesteps 5000000 --num-envs 1024
"""

import argparse
import os

import warp as wp
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import VecNormalize

from envs.newton_intercept_env import NewtonInterceptVecEnv
from train.config import TrainConfig, EnvConfig


def train(args):
    wp.init()

    cfg = TrainConfig()
    env_cfg = EnvConfig()

    if args.timesteps:
        cfg.total_timesteps = args.timesteps

    num_envs = args.num_envs or cfg.n_envs

    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    # --- GPU-accelerated training environment ---
    train_env = NewtonInterceptVecEnv(num_envs=num_envs, config=env_cfg)
    if cfg.normalize_obs or cfg.normalize_reward:
        train_env = VecNormalize(
            train_env,
            norm_obs=cfg.normalize_obs,
            norm_reward=cfg.normalize_reward,
        )

    # --- Evaluation environment (smaller, same GPU) ---
    eval_env = NewtonInterceptVecEnv(num_envs=8, config=env_cfg)
    if cfg.normalize_obs or cfg.normalize_reward:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=cfg.normalize_obs,
            norm_reward=False,
            training=False,
        )

    # --- Callbacks ---
    checkpoint_cb = CheckpointCallback(
        save_freq=max(cfg.checkpoint_freq // num_envs, 1),
        save_path=cfg.model_dir,
        name_prefix="defender_newton",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=cfg.model_dir,
        log_path=cfg.log_dir,
        eval_freq=max(cfg.eval_freq // num_envs, 1),
        n_eval_episodes=cfg.eval_episodes,
        deterministic=True,
    )

    # --- Model ---
    model = SAC(
        cfg.policy,
        train_env,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        gamma=cfg.gamma,
        tau=cfg.tau,
        policy_kwargs={"net_arch": cfg.net_arch},
        tensorboard_log=cfg.tensorboard_log,
        seed=cfg.seed,
        verbose=1,
    )

    print(f"\n{'='*60}")
    print(f"  Training Defender Drone (SAC + Newton GPU Physics)")
    print(f"  Timesteps: {cfg.total_timesteps:,}")
    print(f"  Parallel envs (GPU): {num_envs}")
    print(f"  Checkpoint every: {cfg.checkpoint_freq:,} steps")
    print(f"  Eval every: {cfg.eval_freq:,} steps")
    print(f"{'='*60}\n")

    # --- Train ---
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # --- Save final model ---
    final_path = os.path.join(cfg.model_dir, "defender_newton_final")
    model.save(final_path)
    train_env.save(os.path.join(cfg.model_dir, "vec_normalize_newton.pkl"))
    print(f"\nModel saved to {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train defender drone (Newton GPU)")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Override total training timesteps")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Number of parallel GPU environments (default: from config)")
    args = parser.parse_args()
    train(args)
