"""
Adversarial Co-Training — alternates between training attacker and defender.

Round 1: Train evasive attacker vs frozen defender (Phase 3 best)
Round 2: Retrain defender vs the new evasive attacker
Round N: Repeat — both models improve through competition

Mixture mode: Train defender against a pool of all previous attackers simultaneously,
preventing overfitting to a single opponent.

Usage:
    python -m train.train_adversarial --defender models/defender_v2_99pct.zip --rounds 4
    python -m train.train_adversarial --defender models/best_model.zip --attacker models/phase2v2_best_attacker.zip --rounds 6
    python -m train.train_adversarial --defender models/defender_generalist.zip --mixture --rounds 4
"""

import argparse
import os
import shutil

import warp as wp
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)

from envs.newton_env import NewtonDroneEnv
from train.config import TrainConfig, EnvConfig


def linear_schedule(initial_value):
    """Linear decay from initial_value to 10% of initial_value."""
    def func(progress_remaining):
        return max(0.1, progress_remaining) * initial_value
    return func


def train_role(role, mode, timesteps, num_envs, train_cfg,
               frozen_model_path, resume_path=None):
    """Train one side of the adversarial pair.

    role: 'attacker' or 'defender'
    mode: 'evade' (attacker training) or 'intercept' (defender training)
    frozen_model_path: path to the opponent model (frozen)
    """
    print(f"\n{'='*60}")
    print(f"  Training: {role}")
    print(f"  Mode: {mode}")
    print(f"  Timesteps: {timesteps:,}")
    print(f"  Frozen opponent: {frozen_model_path}")
    if resume_path:
        print(f"  Resume from: {resume_path}")
    print(f"{'='*60}\n")

    os.makedirs(train_cfg.model_dir, exist_ok=True)
    os.makedirs(train_cfg.log_dir, exist_ok=True)

    # Build env config
    env_kwargs = {"mode": mode}
    if mode == "evade":
        env_kwargs["defender_model_path"] = frozen_model_path
    elif mode == "intercept":
        env_kwargs["attacker_model_path"] = frozen_model_path

    cfg = EnvConfig(**env_kwargs)
    train_env = NewtonDroneEnv(num_envs=num_envs, config=cfg)
    eval_env = NewtonDroneEnv(num_envs=min(8, num_envs), config=cfg)

    # Callbacks
    prefix = f"adv_{role}"
    checkpoint_cb = CheckpointCallback(
        save_freq=max(train_cfg.checkpoint_freq // num_envs, 1),
        save_path=train_cfg.model_dir,
        name_prefix=prefix,
    )
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=100,
        min_evals=40,
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=train_cfg.model_dir,
        log_path=train_cfg.log_dir,
        eval_freq=max(train_cfg.eval_freq // num_envs, 1),
        n_eval_episodes=train_cfg.eval_episodes,
        deterministic=True,
        callback_after_eval=stop_cb,
    )

    # Model
    ent_coef = train_cfg.ent_coef
    if ent_coef != "auto":
        try:
            ent_coef = float(ent_coef)
        except (ValueError, TypeError):
            ent_coef = "auto"

    if resume_path:
        print(f"  Loading weights from: {resume_path}")
        model = SAC.load(resume_path, env=train_env)
        model.learning_rate = train_cfg.learning_rate
        model.learning_starts = train_cfg.learning_starts
    else:
        model = SAC(
            train_cfg.policy,
            train_env,
            learning_rate=linear_schedule(train_cfg.learning_rate),
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

    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    save_path = os.path.join(train_cfg.model_dir, f"{prefix}_final")
    model.save(save_path)

    print(f"\n  Saved: {save_path}")

    train_env.close()
    eval_env.close()

    # Return path to best model
    best_path = os.path.join(train_cfg.model_dir, "best_model.zip")
    return best_path


def train_mixture_defender(attacker_paths, timesteps, num_envs, train_cfg,
                           resume_path=None, prefix="mix_defender"):
    """Train defender against a mixture of attacker models simultaneously.

    Each parallel env randomly gets assigned a different attacker on reset,
    preventing overfitting to any single opponent.
    """
    print(f"\n{'='*60}")
    print(f"  Mixture Training: defender vs {len(attacker_paths)} attackers + scripted")
    for p in attacker_paths:
        print(f"    - {p}")
    print(f"  Timesteps: {timesteps:,}")
    if resume_path:
        print(f"  Resume from: {resume_path}")
    print(f"{'='*60}\n")

    os.makedirs(train_cfg.model_dir, exist_ok=True)
    os.makedirs(train_cfg.log_dir, exist_ok=True)

    cfg = EnvConfig(mode="intercept", attacker_model_paths=tuple(attacker_paths))
    train_env = NewtonDroneEnv(num_envs=num_envs, config=cfg)
    eval_env = NewtonDroneEnv(num_envs=min(8, num_envs), config=cfg)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(train_cfg.checkpoint_freq // num_envs, 1),
        save_path=train_cfg.model_dir,
        name_prefix=prefix,
    )
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=100,
        min_evals=40,
        verbose=1,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=train_cfg.model_dir,
        log_path=train_cfg.log_dir,
        eval_freq=max(train_cfg.eval_freq // num_envs, 1),
        n_eval_episodes=train_cfg.eval_episodes,
        deterministic=True,
        callback_after_eval=stop_cb,
    )

    ent_coef = train_cfg.ent_coef
    if ent_coef != "auto":
        try:
            ent_coef = float(ent_coef)
        except (ValueError, TypeError):
            ent_coef = "auto"

    if resume_path:
        print(f"  Loading weights from: {resume_path}")
        model = SAC.load(resume_path, env=train_env)
        model.learning_rate = train_cfg.learning_rate
        model.learning_starts = train_cfg.learning_starts
    else:
        model = SAC(
            train_cfg.policy,
            train_env,
            learning_rate=linear_schedule(train_cfg.learning_rate),
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

    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    save_path = os.path.join(train_cfg.model_dir, f"{prefix}_final")
    model.save(save_path)
    print(f"\n  Saved: {save_path}")

    train_env.close()
    eval_env.close()

    best_path = os.path.join(train_cfg.model_dir, "best_model.zip")
    return best_path


def run(args):
    wp.init()
    train_cfg = TrainConfig()

    if args.mixture:
        # Mixture mode: collect all attacker models and train defender against all
        attacker_pool = list(args.attackers) if args.attackers else []

        defender_path = args.defender

        for round_num in range(1, args.rounds + 1):
            print(f"\n{'#'*60}")
            print(f"  MIXTURE ROUND {round_num}/{args.rounds}")
            print(f"  Attacker pool: {len(attacker_pool)} RL models + scripted")
            print(f"{'#'*60}")

            # Step 1: Train evasive attacker vs current best defender
            attacker_path = train_role(
                role="attacker",
                mode="evade",
                timesteps=args.timesteps,
                num_envs=args.num_envs,
                train_cfg=train_cfg,
                frozen_model_path=defender_path,
                resume_path=None,  # Fresh attacker each round
            )
            round_att_path = os.path.join(train_cfg.model_dir, f"mix_attacker_round{round_num}.zip")
            shutil.copy(attacker_path, round_att_path)
            attacker_pool.append(round_att_path)

            # Step 2: Train defender vs ALL attackers in pool
            defender_path = train_mixture_defender(
                attacker_paths=attacker_pool,
                timesteps=args.timesteps,
                num_envs=args.num_envs,
                train_cfg=train_cfg,
                resume_path=defender_path,
                prefix=f"mix_defender_round{round_num}",
            )
            round_def_path = os.path.join(train_cfg.model_dir, f"mix_defender_round{round_num}.zip")
            shutil.copy(defender_path, round_def_path)

            print(f"\n  Round {round_num} complete:")
            print(f"    New attacker: {round_att_path}")
            print(f"    Defender: {round_def_path}")
            print(f"    Pool size: {len(attacker_pool)} RL attackers + scripted")

        print(f"\n{'='*60}")
        print(f"  Mixture co-training complete! ({args.rounds} rounds)")
        print(f"{'='*60}")
        return

    # Standard adversarial mode
    defender_path = args.defender
    attacker_path = args.attackers[0] if args.attackers else None

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'#'*60}")
        print(f"  ADVERSARIAL ROUND {round_num}/{args.rounds}")
        print(f"{'#'*60}")

        # Step 1: Train evasive attacker vs frozen defender
        attacker_path = train_role(
            role="attacker",
            mode="evade",
            timesteps=args.timesteps,
            num_envs=args.num_envs,
            train_cfg=train_cfg,
            frozen_model_path=defender_path,
            resume_path=attacker_path,
        )
        # Save round checkpoint
        round_att_path = os.path.join(train_cfg.model_dir, f"adv_attacker_round{round_num}.zip")
        shutil.copy(attacker_path, round_att_path)

        # Step 2: Retrain defender vs new evasive attacker
        defender_path = train_role(
            role="defender",
            mode="intercept",
            timesteps=args.timesteps,
            num_envs=args.num_envs,
            train_cfg=train_cfg,
            frozen_model_path=round_att_path,
            resume_path=defender_path,
        )
        # Save round checkpoint
        round_def_path = os.path.join(train_cfg.model_dir, f"adv_defender_round{round_num}.zip")
        shutil.copy(defender_path, round_def_path)

        print(f"\n  Round {round_num} complete:")
        print(f"    Attacker: {round_att_path}")
        print(f"    Defender: {round_def_path}")

    print(f"\n{'='*60}")
    print(f"  Adversarial co-training complete! ({args.rounds} rounds)")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adversarial Co-Training")
    parser.add_argument("--defender", type=str, required=True,
                        help="Path to initial defender model")
    parser.add_argument("--attackers", type=str, nargs="*", default=[],
                        help="Path(s) to initial attacker model(s)")
    parser.add_argument("--rounds", type=int, default=4,
                        help="Number of adversarial rounds")
    parser.add_argument("--num-envs", type=int, default=256,
                        help="Parallel GPU environments")
    parser.add_argument("--timesteps", type=int, default=3_000_000,
                        help="Timesteps per training phase")
    parser.add_argument("--mixture", action="store_true",
                        help="Use mixture-of-opponents training (prevents overfitting)")
    args = parser.parse_args()
    run(args)
