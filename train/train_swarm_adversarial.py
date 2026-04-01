"""
Swarm Adversarial Co-Training — NvM attacker/defender arms race.

Alternates between:
  1. Train RL swarm attackers vs frozen defender model
  2. Retrain swarm defenders vs the new RL attacker model
  3. Repeat for N rounds

Each round produces stronger attackers and defenders through competition.
Supports mixture mode: train defender against pool of all previous attacker generations.

Usage:
    # Standard adversarial (2v2, 4 rounds)
    python -m train.train_swarm_adversarial --defenders 2 --attackers 2 --defender models/swarm_2v2_final.zip --rounds 4

    # Mixture mode (prevents overfitting to single attacker)
    python -m train.train_swarm_adversarial --defenders 2 --attackers 2 --defender models/swarm_2v2_final.zip --mixture --rounds 4

    # 3v2 asymmetric
    python -m train.train_swarm_adversarial --defenders 3 --attackers 2 --defender models/swarm_3v2_final.zip --rounds 3
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

from envs.newton_swarm_env import NewtonSwarmEnv
from train.config import TrainConfig, EnvConfig


def linear_schedule(initial_value):
    """Linear decay from initial_value to 10% of initial_value."""
    def func(progress_remaining):
        return max(0.1, progress_remaining) * initial_value
    return func


def _realworld_kwargs(args):
    """Common real-world domain randomization kwargs."""
    return dict(
        wind_enabled=args.wind,
        wind_max_force=args.wind_force,
        wind_gust_prob=args.gust_prob,
        wind_gust_force=args.gust_force,
        obs_noise_std=args.noise,
        sensor_range=args.sensor_range,
        attacker_speed_variance=args.speed_var,
        drone_mass_variance=args.mass_var,
        k_opponents=args.k_opponents,
        k_teammates=args.k_teammates,
        kamikaze=args.kamikaze,
    )


def train_swarm_role(role, args, train_cfg, frozen_model_path, resume_path=None, tag_suffix=""):
    """Train one side of the swarm adversarial pair.

    role: 'attacker' or 'defender'
    frozen_model_path: path to opponent model
    """
    rw = _realworld_kwargs(args)
    if role == "attacker":
        mode = "evade"
        env_kwargs = dict(
            mode=mode,
            num_defenders=args.defenders,
            num_attackers=args.attackers,
            num_obstacles=args.obstacles,
            defender_model_path=frozen_model_path,
            **rw,
        )
    else:
        mode = "intercept"
        env_kwargs = dict(
            mode=mode,
            num_defenders=args.defenders,
            num_attackers=args.attackers,
            num_obstacles=args.obstacles,
            attacker_model_path=frozen_model_path,
            **rw,
        )

    tag = f"swarm_adv_{role}_{args.defenders}v{args.attackers}{tag_suffix}"

    print(f"\n{'='*60}")
    print(f"  Training: {role} ({mode} mode)")
    print(f"  Config: {args.defenders}v{args.attackers} | {tag}")
    print(f"  Frozen opponent: {frozen_model_path}")
    if resume_path:
        print(f"  Resume: {resume_path}")
    print(f"  Timesteps: {args.timesteps:,}")
    print(f"{'='*60}\n")

    cfg = EnvConfig(**env_kwargs)
    env = NewtonSwarmEnv(num_worlds=args.num_worlds, config=cfg)

    # Eval env (no randomization, scripted opponent for cleaner eval)
    eval_kwargs = dict(
        mode=mode,
        num_defenders=args.defenders,
        num_attackers=args.attackers,
        num_obstacles=args.obstacles,
        k_opponents=args.k_opponents,
        k_teammates=args.k_teammates,
        kamikaze=args.kamikaze,
    )
    if mode == "evade":
        eval_kwargs["defender_model_path"] = frozen_model_path
    else:
        eval_kwargs["attacker_model_path"] = frozen_model_path
    eval_cfg = EnvConfig(**eval_kwargs)
    eval_env = NewtonSwarmEnv(num_worlds=max(4, args.num_worlds // 16), config=eval_cfg)

    # Model
    ent_coef = train_cfg.ent_coef
    if ent_coef != "auto":
        try:
            ent_coef = float(ent_coef)
        except (ValueError, TypeError):
            ent_coef = "auto"

    if resume_path:
        model = SAC.load(resume_path, env=env, device="cuda")
        model.learning_rate = train_cfg.learning_rate
    else:
        model = SAC(
            train_cfg.policy,
            env,
            learning_rate=linear_schedule(train_cfg.learning_rate),
            batch_size=train_cfg.batch_size,
            buffer_size=train_cfg.buffer_size * (args.attackers if role == "attacker" else args.defenders),
            learning_starts=train_cfg.learning_starts,
            gamma=train_cfg.gamma,
            tau=train_cfg.tau,
            ent_coef=ent_coef,
            policy_kwargs={"net_arch": train_cfg.net_arch},
            tensorboard_log=train_cfg.tensorboard_log,
            seed=train_cfg.seed,
            verbose=1,
            device="cuda",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=train_cfg.checkpoint_freq,
        save_path=train_cfg.model_dir,
        name_prefix=tag,
    )
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=80,
        min_evals=30,
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
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name=tag,
    )

    final_path = os.path.join(train_cfg.model_dir, f"{tag}_final")
    model.save(final_path)
    print(f"\n  Saved: {final_path}.zip")

    env.close()
    eval_env.close()

    best_path = os.path.join(train_cfg.model_dir, "best_model.zip")
    return best_path


def train_swarm_mixture_defender(args, train_cfg, attacker_pool, resume_path=None, tag_suffix=""):
    """Train swarm defender against mixture of attacker models + scripted."""
    tag = f"swarm_mix_def_{args.defenders}v{args.attackers}{tag_suffix}"

    print(f"\n{'='*60}")
    print(f"  Mixture Training: defender vs {len(attacker_pool)} RL attackers + scripted")
    for p in attacker_pool:
        print(f"    - {p}")
    print(f"  Timesteps: {args.timesteps:,}")
    if resume_path:
        print(f"  Resume: {resume_path}")
    print(f"{'='*60}\n")

    rw = _realworld_kwargs(args)
    cfg = EnvConfig(
        mode="intercept",
        num_defenders=args.defenders,
        num_attackers=args.attackers,
        num_obstacles=args.obstacles,
        attacker_model_paths=tuple(attacker_pool),
        **rw,
    )
    env = NewtonSwarmEnv(num_worlds=args.num_worlds, config=cfg)

    eval_cfg = EnvConfig(
        mode="intercept",
        num_defenders=args.defenders,
        num_attackers=args.attackers,
        num_obstacles=args.obstacles,
        attacker_model_paths=tuple(attacker_pool),
        k_opponents=args.k_opponents,
        k_teammates=args.k_teammates,
        kamikaze=args.kamikaze,
    )
    eval_env = NewtonSwarmEnv(num_worlds=max(4, args.num_worlds // 16), config=eval_cfg)

    ent_coef = train_cfg.ent_coef
    if ent_coef != "auto":
        try:
            ent_coef = float(ent_coef)
        except (ValueError, TypeError):
            ent_coef = "auto"

    if resume_path:
        model = SAC.load(resume_path, env=env, device="cuda")
        model.learning_rate = train_cfg.learning_rate
    else:
        model = SAC(
            train_cfg.policy,
            env,
            learning_rate=linear_schedule(train_cfg.learning_rate),
            batch_size=train_cfg.batch_size,
            buffer_size=train_cfg.buffer_size * args.defenders,
            learning_starts=train_cfg.learning_starts,
            gamma=train_cfg.gamma,
            tau=train_cfg.tau,
            ent_coef=ent_coef,
            policy_kwargs={"net_arch": train_cfg.net_arch},
            tensorboard_log=train_cfg.tensorboard_log,
            seed=train_cfg.seed,
            verbose=1,
            device="cuda",
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=train_cfg.checkpoint_freq,
        save_path=train_cfg.model_dir,
        name_prefix=tag,
    )
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=80,
        min_evals=30,
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
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name=tag,
    )

    final_path = os.path.join(train_cfg.model_dir, f"{tag}_final")
    model.save(final_path)
    print(f"\n  Saved: {final_path}.zip")

    env.close()
    eval_env.close()

    best_path = os.path.join(train_cfg.model_dir, "best_model.zip")
    return best_path


def main():
    parser = argparse.ArgumentParser(description="Swarm Adversarial Co-Training")
    parser.add_argument("--defenders", type=int, default=2)
    parser.add_argument("--attackers", type=int, default=2)
    parser.add_argument("--defender", type=str, required=True,
                        help="Path to initial defender model")
    parser.add_argument("--attacker", type=str, default="",
                        help="Path to initial attacker model (optional)")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--num-worlds", type=int, default=128)
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--obstacles", type=int, default=3)
    parser.add_argument("--wind", action="store_true")
    parser.add_argument("--wind-force", type=float, default=0.3)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--mixture", action="store_true",
                        help="Use mixture-of-opponents training")
    parser.add_argument("--sensor-range", type=float, default=0.0, help="Sensor range (0=unlimited)")
    parser.add_argument("--gust-prob", type=float, default=0.0, help="Wind gust probability per step")
    parser.add_argument("--gust-force", type=float, default=1.0, help="Max gust force (N)")
    parser.add_argument("--speed-var", type=float, default=0.0, help="Attacker speed variance ±fraction")
    parser.add_argument("--mass-var", type=float, default=0.0, help="Drone mass variance ±fraction")
    parser.add_argument("--k-opponents", type=int, default=3, help="K nearest opponents to observe")
    parser.add_argument("--k-teammates", type=int, default=2, help="K nearest teammates to observe")
    parser.add_argument("--kamikaze", action="store_true", default=True, help="Defender destroyed on interception")
    parser.add_argument("--no-kamikaze", dest="kamikaze", action="store_false", help="Disable kamikaze")
    args = parser.parse_args()

    wp.init()
    train_cfg = TrainConfig()

    defender_path = args.defender
    attacker_path = args.attacker if args.attacker else None
    attacker_pool = []

    for round_num in range(1, args.rounds + 1):
        print(f"\n{'#'*60}")
        print(f"  SWARM ADVERSARIAL ROUND {round_num}/{args.rounds}")
        print(f"  Config: {args.defenders}v{args.attackers}")
        if args.mixture:
            print(f"  Mode: Mixture ({len(attacker_pool)} RL attackers in pool)")
        print(f"{'#'*60}")

        # Step 1: Train RL swarm attackers vs frozen defender
        attacker_best = train_swarm_role(
            role="attacker",
            args=args,
            train_cfg=train_cfg,
            frozen_model_path=defender_path,
            resume_path=attacker_path,
            tag_suffix=f"_r{round_num}",
        )
        round_att_path = os.path.join(
            train_cfg.model_dir,
            f"swarm_adv_attacker_{args.defenders}v{args.attackers}_round{round_num}.zip"
        )
        shutil.copy(attacker_best, round_att_path)
        attacker_pool.append(round_att_path)

        # Step 2: Train/retrain defender
        if args.mixture and len(attacker_pool) > 1:
            # Mixture: train against all attacker generations
            defender_best = train_swarm_mixture_defender(
                args=args,
                train_cfg=train_cfg,
                attacker_pool=attacker_pool,
                resume_path=defender_path,
                tag_suffix=f"_r{round_num}",
            )
        else:
            # Standard: train against latest attacker only
            defender_best = train_swarm_role(
                role="defender",
                args=args,
                train_cfg=train_cfg,
                frozen_model_path=round_att_path,
                resume_path=defender_path,
                tag_suffix=f"_r{round_num}",
            )
        round_def_path = os.path.join(
            train_cfg.model_dir,
            f"swarm_adv_defender_{args.defenders}v{args.attackers}_round{round_num}.zip"
        )
        shutil.copy(defender_best, round_def_path)
        defender_path = round_def_path

        # Reset attacker resume for next round (fresh attacker each round)
        attacker_path = None

        print(f"\n  Round {round_num} complete:")
        print(f"    Attacker: {round_att_path}")
        print(f"    Defender: {round_def_path}")
        if args.mixture:
            print(f"    Pool size: {len(attacker_pool)} RL attackers")

    print(f"\n{'='*60}")
    print(f"  Swarm adversarial co-training complete! ({args.rounds} rounds)")
    print(f"  Final defender: {defender_path}")
    print(f"  Attacker pool: {len(attacker_pool)} models")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
