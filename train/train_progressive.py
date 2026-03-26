"""
Progressive Training Pipeline — trains a drone model through phases:

  Phase 1 (fly_to_target): Learn to fly to random static targets
  Phase 2 (chase):         Chase a moving target (transfers Phase 1 weights)
  Phase 3 (intercept):     Defend against Phase 2 model as attacker

Each phase saves a model that the next phase loads for weight transfer.

Usage:
    python -m train.train_progressive                          # all phases
    python -m train.train_progressive --phase 1                # phase 1 only
    python -m train.train_progressive --phase 2 --resume models/phase1_fly.zip
    python -m train.train_progressive --phase 3 --attacker models/phase2_chase.zip
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

from envs.newton_env import NewtonDroneEnv
from train.config import TrainConfig, EnvConfig


def make_env(mode, num_envs, attacker_model_path="", **env_overrides):
    """Create a NewtonDroneEnv with the given mode."""
    cfg = EnvConfig(mode=mode, attacker_model_path=attacker_model_path, **env_overrides)
    return NewtonDroneEnv(num_envs=num_envs, config=cfg)


def make_model(env, train_cfg, resume_path=None):
    """Create or load a SAC model."""
    if resume_path:
        print(f"  Loading weights from: {resume_path}")
        model = SAC.load(resume_path, env=env)
        # Reset learning rate and exploration (fresh training schedule)
        model.learning_rate = train_cfg.learning_rate
        model.learning_starts = train_cfg.learning_starts
        return model

    # Use fixed entropy coefficient to prevent collapse
    ent_coef = train_cfg.ent_coef
    if ent_coef != "auto":
        try:
            ent_coef = float(ent_coef)
        except (ValueError, TypeError):
            ent_coef = "auto"

    return SAC(
        train_cfg.policy,
        env,
        learning_rate=train_cfg.learning_rate,
        batch_size=train_cfg.batch_size,
        buffer_size=train_cfg.buffer_size,
        learning_starts=train_cfg.learning_starts,
        gamma=train_cfg.gamma,
        tau=train_cfg.tau,
        ent_coef=ent_coef,
        policy_kwargs={"net_arch": train_cfg.net_arch},
        tensorboard_log=train_cfg.tensorboard_log,
        seed=train_cfg.seed,
        verbose=1,
    )


def train_phase(phase_name, mode, timesteps, num_envs, train_cfg,
                resume_path=None, attacker_model_path="", **env_overrides):
    """Run a single training phase."""
    print(f"\n{'='*60}")
    print(f"  Phase: {phase_name}")
    print(f"  Mode: {mode}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Parallel envs: {num_envs}")
    if resume_path:
        print(f"  Resume from: {resume_path}")
    if attacker_model_path:
        print(f"  Attacker model: {attacker_model_path}")
    print(f"{'='*60}\n")

    os.makedirs(train_cfg.model_dir, exist_ok=True)
    os.makedirs(train_cfg.log_dir, exist_ok=True)

    # Training env
    train_env = make_env(mode, num_envs, attacker_model_path, **env_overrides)
    if train_cfg.normalize_obs or train_cfg.normalize_reward:
        train_env = VecNormalize(
            train_env,
            norm_obs=train_cfg.normalize_obs,
            norm_reward=train_cfg.normalize_reward,
        )

    # Eval env (smaller)
    eval_env = make_env(mode, min(8, num_envs), attacker_model_path, **env_overrides)
    if train_cfg.normalize_obs:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=train_cfg.normalize_obs,
            norm_reward=False,
            training=False,
        )

    # Callbacks
    prefix = phase_name.replace(" ", "_").lower()
    checkpoint_cb = CheckpointCallback(
        save_freq=max(train_cfg.checkpoint_freq // num_envs, 1),
        save_path=train_cfg.model_dir,
        name_prefix=prefix,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=train_cfg.model_dir,
        log_path=train_cfg.log_dir,
        eval_freq=max(train_cfg.eval_freq // num_envs, 1),
        n_eval_episodes=train_cfg.eval_episodes,
        deterministic=True,
    )

    # Model
    model = make_model(train_env, train_cfg, resume_path)

    # Train
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # Save
    save_path = os.path.join(train_cfg.model_dir, f"{prefix}_final")
    model.save(save_path)
    if hasattr(train_env, "save"):
        train_env.save(os.path.join(train_cfg.model_dir, f"{prefix}_vec_normalize.pkl"))

    print(f"\n  Saved: {save_path}.zip")

    train_env.close()
    eval_env.close()

    return f"{save_path}.zip"


def run(args):
    wp.init()
    train_cfg = TrainConfig()
    num_envs = args.num_envs

    phases_to_run = [args.phase] if args.phase else [1, 2, 3]

    phase1_model = args.resume
    phase2_model = args.attacker

    for phase in phases_to_run:
        if phase == 1:
            phase1_model = train_phase(
                phase_name="phase1_fly",
                mode="fly_to_target",
                timesteps=args.timesteps_p1,
                num_envs=num_envs,
                train_cfg=train_cfg,
                resume_path=args.resume,
            )

        elif phase == 2:
            if not phase1_model and not args.resume:
                phase1_model = os.path.join(train_cfg.model_dir, "phase1_fly_final.zip")
            resume = args.resume or phase1_model

            # Curriculum: increasing target speed
            speeds = [0.1, 0.3, 0.5]
            steps_per_speed = args.timesteps_p2 // len(speeds)

            for i, speed in enumerate(speeds):
                resume = train_phase(
                    phase_name=f"phase2_chase_speed{speed}",
                    mode="chase",
                    timesteps=steps_per_speed,
                    num_envs=num_envs,
                    train_cfg=train_cfg,
                    resume_path=resume,
                    moving_target_speed=speed,
                )
            phase2_model = resume

        elif phase == 3:
            if not phase2_model and not args.attacker:
                phase2_model = os.path.join(train_cfg.model_dir, "phase2_chase_speed0.5_final.zip")
            attacker = args.attacker or phase2_model

            train_phase(
                phase_name="phase3_defender",
                mode="intercept",
                timesteps=args.timesteps_p3,
                num_envs=num_envs,
                train_cfg=train_cfg,
                resume_path=None,  # Fresh defender
                attacker_model_path=attacker,
            )

    print("\n" + "="*60)
    print("  Progressive training complete!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive Drone Training")
    parser.add_argument("--phase", type=int, default=None, choices=[1, 2, 3],
                        help="Run only this phase (default: all)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to model to resume from")
    parser.add_argument("--attacker", type=str, default=None,
                        help="Path to attacker model (phase 3)")
    parser.add_argument("--num-envs", type=int, default=256,
                        help="Parallel GPU environments")
    parser.add_argument("--timesteps-p1", type=int, default=500_000,
                        help="Phase 1 timesteps")
    parser.add_argument("--timesteps-p2", type=int, default=1_500_000,
                        help="Phase 2 total timesteps (split across speed levels)")
    parser.add_argument("--timesteps-p3", type=int, default=2_000_000,
                        help="Phase 3 timesteps")
    args = parser.parse_args()
    run(args)
