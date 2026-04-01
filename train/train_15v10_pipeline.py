"""
15v10 Swarm Training Pipeline — Full autonomous training.

Runs the complete training pipeline:
  1. Train 15v10 defenders vs scripted mixed attackers (10M steps)
  2. Train 15v10 RL attackers vs frozen defender (10M steps)
  3. Retrain defenders vs RL attackers (10M steps)
  4. Repeat adversarial rounds (4 rounds total)

Total training time: ~12+ hours on DGX Spark.
Kamikaze mode ON — defenders are destroyed on interception.
K-nearest observation (K=3 opponents, K=2 teammates) for NvM generalization.

Usage:
    python3 -m train.train_15v10_pipeline
"""

import os
import shutil
import time

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
    def func(progress_remaining):
        return max(0.1, progress_remaining) * initial_value
    return func


COMMON_KWARGS = dict(
    k_opponents=3,
    k_teammates=2,
    kamikaze=True,
    num_obstacles=3,
    wind_enabled=True,
    wind_max_force=0.3,
    wind_gust_prob=0.02,
    wind_gust_force=1.0,
    obs_noise_std=0.02,
    sensor_range=5.0,
    attacker_speed_variance=0.15,
    drone_mass_variance=0.1,
)


def train_role(role, defenders, attackers, num_worlds, timesteps, train_cfg,
               frozen_model_path=None, resume_path=None, tag="", strategy="mixed"):
    """Train one side of the swarm."""
    if role == "attacker":
        mode = "evade"
        opponent_kwargs = dict(defender_model_path=frozen_model_path) if frozen_model_path else {}
    else:
        mode = "intercept"
        opponent_kwargs = dict(attacker_model_path=frozen_model_path) if frozen_model_path else {}

    cfg = EnvConfig(
        mode=mode,
        num_defenders=defenders,
        num_attackers=attackers,
        attacker_strategy=strategy,
        **COMMON_KWARGS,
        **opponent_kwargs,
    )
    env = NewtonSwarmEnv(num_worlds=num_worlds, config=cfg)

    eval_cfg = EnvConfig(
        mode=mode,
        num_defenders=defenders,
        num_attackers=attackers,
        k_opponents=3,
        k_teammates=2,
        kamikaze=True,
        num_obstacles=3,
        **opponent_kwargs,
    )
    eval_env = NewtonSwarmEnv(num_worlds=max(4, num_worlds // 16), config=eval_cfg)

    ent_coef = train_cfg.ent_coef
    if ent_coef != "auto":
        try:
            ent_coef = float(ent_coef)
        except (ValueError, TypeError):
            ent_coef = "auto"

    n_agents = attackers if role == "attacker" else defenders

    if resume_path and os.path.exists(resume_path):
        print(f"  Loading weights from {resume_path}")
        model = SAC.load(resume_path, env=env, device="cuda")
        model.learning_rate = linear_schedule(train_cfg.learning_rate)
    else:
        model = SAC(
            train_cfg.policy,
            env,
            learning_rate=linear_schedule(train_cfg.learning_rate),
            batch_size=train_cfg.batch_size,
            buffer_size=train_cfg.buffer_size * n_agents,
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
        max_no_improvement_evals=100,
        min_evals=40,
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

    start = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb],
        tb_log_name=tag,
    )
    elapsed = time.time() - start

    final_path = os.path.join(train_cfg.model_dir, f"{tag}_final")
    model.save(final_path)
    print(f"\n  Saved: {final_path}.zip ({elapsed/60:.1f} min)")

    env.close()
    eval_env.close()

    best_path = os.path.join(train_cfg.model_dir, "best_model.zip")
    return best_path


def main():
    wp.init()
    train_cfg = TrainConfig()

    N_DEF = 15
    N_ATT = 10
    NUM_WORLDS = 128
    STEPS_SCRIPTED = 10_000_000
    STEPS_ADVERSARIAL = 10_000_000
    NUM_ROUNDS = 4

    print(f"\n{'#'*60}")
    print(f"  15v10 SWARM TRAINING PIPELINE")
    print(f"  Kamikaze: ON | K-nearest: 3 opp, 2 tm")
    print(f"  Worlds: {NUM_WORLDS} | Scripted steps: {STEPS_SCRIPTED:,}")
    print(f"  Adversarial rounds: {NUM_ROUNDS} x {STEPS_ADVERSARIAL:,} steps")
    print(f"{'#'*60}\n")

    # ============================================================
    # Phase 1: Train defenders vs scripted mixed attackers
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  PHASE 1: 15v10 Defenders vs Scripted Attackers")
    print(f"{'='*60}")

    # Try warm-start from curriculum-trained 2v2 model
    resume = "models/swarm_2v2_stage5_realworld_final.zip"
    if not os.path.exists(resume):
        resume = None

    defender_best = train_role(
        role="defender",
        defenders=N_DEF,
        attackers=N_ATT,
        num_worlds=NUM_WORLDS,
        timesteps=STEPS_SCRIPTED,
        train_cfg=train_cfg,
        frozen_model_path=None,
        resume_path=resume,
        tag="swarm_15v10_scripted",
        strategy="mixed",
    )
    defender_path = os.path.join(train_cfg.model_dir, "swarm_15v10_scripted_best.zip")
    shutil.copy(defender_best, defender_path)
    print(f"\n  Phase 1 defender: {defender_path}")

    # ============================================================
    # Phase 2: Adversarial co-training rounds
    # ============================================================
    attacker_path = None
    attacker_pool = []

    for round_num in range(1, NUM_ROUNDS + 1):
        print(f"\n{'#'*60}")
        print(f"  ADVERSARIAL ROUND {round_num}/{NUM_ROUNDS}")
        print(f"{'#'*60}")

        # Step A: Train RL attackers vs frozen defender
        print(f"\n  --- Training RL Attackers (round {round_num}) ---")
        attacker_best = train_role(
            role="attacker",
            defenders=N_DEF,
            attackers=N_ATT,
            num_worlds=NUM_WORLDS,
            timesteps=STEPS_ADVERSARIAL,
            train_cfg=train_cfg,
            frozen_model_path=defender_path,
            resume_path=attacker_path,
            tag=f"swarm_15v10_att_r{round_num}",
        )
        round_att_path = os.path.join(
            train_cfg.model_dir,
            f"swarm_15v10_attacker_round{round_num}.zip"
        )
        shutil.copy(attacker_best, round_att_path)
        attacker_pool.append(round_att_path)
        print(f"  Attacker round {round_num}: {round_att_path}")

        # Step B: Retrain defenders vs RL attackers
        print(f"\n  --- Retraining Defenders (round {round_num}) ---")
        defender_best = train_role(
            role="defender",
            defenders=N_DEF,
            attackers=N_ATT,
            num_worlds=NUM_WORLDS,
            timesteps=STEPS_ADVERSARIAL,
            train_cfg=train_cfg,
            frozen_model_path=round_att_path,
            resume_path=defender_path,
            tag=f"swarm_15v10_def_r{round_num}",
        )
        round_def_path = os.path.join(
            train_cfg.model_dir,
            f"swarm_15v10_defender_round{round_num}.zip"
        )
        shutil.copy(defender_best, round_def_path)
        defender_path = round_def_path
        print(f"  Defender round {round_num}: {round_def_path}")

        # Reset attacker resume for next round
        attacker_path = None

        print(f"\n  Round {round_num} complete!")

    # ============================================================
    # Phase 3: Generalization validation — test on different swarm sizes
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  GENERALIZATION TEST")
    print(f"{'='*60}")

    final_model = SAC.load(defender_path, device="cpu")
    for n_d, n_a in [(8, 5), (15, 10), (20, 15), (5, 3)]:
        cfg = EnvConfig(
            mode="intercept",
            num_defenders=n_d,
            num_attackers=n_a,
            k_opponents=3,
            k_teammates=2,
            kamikaze=True,
            num_obstacles=3,
        )
        env = NewtonSwarmEnv(num_worlds=16, config=cfg)
        obs = env.reset()
        total_reward = 0.0
        total_intercepts = 0
        total_episodes = 0
        for _ in range(500):
            actions, _ = final_model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(actions)
            total_reward += rewards.sum()
            for info in infos:
                if info.get("intercepted"):
                    total_intercepts += 1
            total_episodes += dones.sum()
        env.close()
        print(f"  {n_d}v{n_a}: avg_reward={total_reward/(total_episodes+1):.1f}, "
              f"intercepts={total_intercepts}, episodes={int(total_episodes)}")

    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE!")
    print(f"  Final defender: {defender_path}")
    print(f"  Attacker pool: {len(attacker_pool)} models")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
