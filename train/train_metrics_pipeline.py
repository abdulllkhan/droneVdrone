"""
Metrics-Driven Swarm Training Pipeline.

Trains defenders in phases, collecting diagnostics after each phase and
adjusting strategy based on what the metrics reveal.

Pipeline:
  Phase 1: 2v2 basics — learn to pursue and intercept (5M steps)
  Phase 2: 5v3 scaling — learn multi-target coordination (5M steps)
  Phase 3: 10v7 scale — stress-test at medium scale (5M steps)
  Phase 4: 15v10 full scale — target configuration (5M steps)
  Phase 5: Adversarial — train vs RL attackers (5M steps)

After each phase:
  - Collect metrics (intercept rate, clustering, closing speed, death types...)
  - Run diagnose() to identify issues
  - Log everything to logs/metrics_pipeline.jsonl
  - Print actionable report

Usage:
    python3 -m train.train_metrics_pipeline
"""

import os
import shutil
import time
import json

import numpy as np
import warp as wp
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

from envs.newton_swarm_env import NewtonSwarmEnv
from train.config import TrainConfig, EnvConfig
from analysis.metrics import MetricsTracker, diagnose


METRICS_LOG = "logs/metrics_pipeline.jsonl"


def linear_schedule(initial_value):
    def func(progress_remaining):
        return max(0.1, progress_remaining) * initial_value
    return func


class MetricsCallback(BaseCallback):
    """SB3 callback that periodically summarizes and logs metrics."""

    def __init__(self, env, report_freq=500, log_path=METRICS_LOG,
                 phase_name="", verbose=1):
        super().__init__(verbose)
        self.env = env
        self.report_freq = report_freq
        self.log_path = log_path
        self.phase_name = phase_name
        self._step = 0

    def _on_step(self) -> bool:
        self._step += 1
        if self._step % self.report_freq == 0 and self.env.metrics is not None:
            report = self.env.metrics.summarize(reset=True)
            if report["n_episodes"] > 0:
                report["timestep"] = self.num_timesteps
                if self.verbose:
                    print(MetricsTracker.format_report(report, self.phase_name))
                    issues = diagnose(report)
                    for issue, severity, msg in issues:
                        print(f"  [{severity.upper()}] {issue}: {msg}")
                    print()
                # Save to log file
                entry = {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "phase": self.phase_name,
                    **report,
                }
                os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
                with open(self.log_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
        return True


COMMON_KWARGS = dict(
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


def train_phase(phase_name, defenders, attackers, num_worlds, timesteps,
                train_cfg, resume_path=None, strategy="mixed",
                frozen_attacker_path=None, k_opponents=3, k_teammates=2):
    """Train one phase and return metrics + model path."""

    print(f"\n{'='*60}")
    print(f"  PHASE: {phase_name}")
    print(f"  {defenders}v{attackers} | {num_worlds} worlds | {timesteps:,} steps")
    print(f"  Resume: {resume_path or 'None'}")
    print(f"{'='*60}\n")

    opponent_kwargs = {}
    if frozen_attacker_path:
        opponent_kwargs["attacker_model_path"] = frozen_attacker_path

    cfg = EnvConfig(
        mode="intercept",
        num_defenders=defenders,
        num_attackers=attackers,
        k_opponents=k_opponents,
        k_teammates=k_teammates,
        attacker_strategy=strategy,
        **COMMON_KWARGS,
        **opponent_kwargs,
    )

    env = NewtonSwarmEnv(num_worlds=num_worlds, config=cfg)
    metrics = env.enable_metrics()

    # Eval env (smaller, no metrics needed)
    eval_cfg = EnvConfig(
        mode="intercept",
        num_defenders=defenders,
        num_attackers=attackers,
        k_opponents=k_opponents,
        k_teammates=k_teammates,
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

    tag = phase_name.replace(" ", "_").lower()

    # GPU-optimized hyperparams
    batch_size = 4096
    gradient_steps = 64

    if resume_path and os.path.exists(resume_path):
        print(f"  Loading weights from {resume_path}")
        model = SAC.load(resume_path, env=env, device="cuda")
        model.learning_rate = linear_schedule(train_cfg.learning_rate)
        model.batch_size = batch_size
        model.gradient_steps = gradient_steps
    else:
        model = SAC(
            train_cfg.policy,
            env,
            learning_rate=linear_schedule(train_cfg.learning_rate),
            batch_size=batch_size,
            buffer_size=train_cfg.buffer_size * defenders,
            learning_starts=train_cfg.learning_starts,
            gamma=train_cfg.gamma,
            tau=train_cfg.tau,
            ent_coef=ent_coef,
            gradient_steps=gradient_steps,
            policy_kwargs={"net_arch": train_cfg.net_arch},
            tensorboard_log=train_cfg.tensorboard_log,
            seed=train_cfg.seed,
            verbose=1,
            device="cuda",
        )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=train_cfg.checkpoint_freq,
        save_path=train_cfg.model_dir,
        name_prefix=tag,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=train_cfg.model_dir,
        log_path=train_cfg.log_dir,
        eval_freq=train_cfg.eval_freq,
        n_eval_episodes=train_cfg.eval_episodes,
        verbose=1,
    )
    metrics_cb = MetricsCallback(
        env, report_freq=200, phase_name=phase_name,
    )

    start = time.time()
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_cb, eval_cb, metrics_cb],
        tb_log_name=tag,
    )
    elapsed = time.time() - start

    # Final model save
    final_path = os.path.join(train_cfg.model_dir, f"{tag}_final")
    model.save(final_path)
    print(f"\n  Saved: {final_path}.zip ({elapsed/60:.1f} min)")

    # Final metrics collection
    final_report = metrics.summarize(reset=True)
    if final_report["n_episodes"] > 0:
        print(MetricsTracker.format_report(final_report, f"{phase_name} FINAL"))
        issues = diagnose(final_report)
        for issue, severity, msg in issues:
            print(f"  [{severity.upper()}] {issue}: {msg}")

    # Save final report
    entry = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "phase": f"{phase_name}_FINAL",
        "elapsed_min": elapsed / 60,
        **final_report,
    }
    os.makedirs("logs", exist_ok=True)
    with open(METRICS_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")

    env.close()
    eval_env.close()

    best_path = os.path.join(train_cfg.model_dir, "best_model.zip")
    if os.path.exists(best_path):
        tagged_best = os.path.join(train_cfg.model_dir, f"{tag}_best.zip")
        shutil.copy(best_path, tagged_best)
        return tagged_best, final_report

    return f"{final_path}.zip", final_report


def run_evaluation(model_path, defenders, attackers, num_worlds=16,
                   num_steps=500, k_opponents=3, k_teammates=2):
    """Run evaluation episodes and return metrics report."""
    cfg = EnvConfig(
        mode="intercept",
        num_defenders=defenders,
        num_attackers=attackers,
        k_opponents=k_opponents,
        k_teammates=k_teammates,
        kamikaze=True,
        num_obstacles=3,
    )
    env = NewtonSwarmEnv(num_worlds=num_worlds, config=cfg)
    metrics = env.enable_metrics()

    model = SAC.load(model_path, device="cpu")
    obs = env.reset()
    for _ in range(num_steps):
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

    report = metrics.summarize(reset=True)
    env.close()
    return report


def main():
    wp.init()
    train_cfg = TrainConfig()

    NUM_WORLDS = 1024
    STEPS_PER_PHASE = 5_000_000

    print(f"\n{'#'*60}")
    print(f"  METRICS-DRIVEN SWARM TRAINING PIPELINE")
    print(f"  Simplified rewards | Death differentiation | Multi-metric")
    print(f"  Worlds: {NUM_WORLDS} | Steps/phase: {STEPS_PER_PHASE:,}")
    print(f"{'#'*60}\n")

    # ==========================================================
    # Phase 1: 2v2 — learn basic pursuit + interception
    # ==========================================================
    model_path, report_1 = train_phase(
        "phase1_2v2_basics",
        defenders=2, attackers=2,
        num_worlds=NUM_WORLDS,
        timesteps=STEPS_PER_PHASE,
        train_cfg=train_cfg,
        strategy="direct",
    )

    # Evaluate on different sizes to check generalization
    for n_d, n_a in [(2, 2), (3, 2), (5, 3)]:
        eval_report = run_evaluation(model_path, n_d, n_a)
        print(MetricsTracker.format_report(eval_report, f"Eval {n_d}v{n_a}"))

    # ==========================================================
    # Phase 2: 5v3 — learn multi-target coordination
    # ==========================================================
    model_path, report_2 = train_phase(
        "phase2_5v3_coordination",
        defenders=5, attackers=3,
        num_worlds=NUM_WORLDS,
        timesteps=STEPS_PER_PHASE,
        train_cfg=train_cfg,
        resume_path=model_path,
        strategy="mixed",
    )

    for n_d, n_a in [(5, 3), (3, 2), (8, 5)]:
        eval_report = run_evaluation(model_path, n_d, n_a)
        print(MetricsTracker.format_report(eval_report, f"Eval {n_d}v{n_a}"))

    # ==========================================================
    # Phase 3: 10v7 — medium scale stress test
    # ==========================================================
    model_path, report_3 = train_phase(
        "phase3_10v7_scaling",
        defenders=10, attackers=7,
        num_worlds=NUM_WORLDS,
        timesteps=STEPS_PER_PHASE,
        train_cfg=train_cfg,
        resume_path=model_path,
        strategy="mixed",
    )

    for n_d, n_a in [(10, 7), (5, 3), (15, 10)]:
        eval_report = run_evaluation(model_path, n_d, n_a)
        print(MetricsTracker.format_report(eval_report, f"Eval {n_d}v{n_a}"))

    # ==========================================================
    # Phase 4: 15v10 — full target scale
    # ==========================================================
    model_path, report_4 = train_phase(
        "phase4_15v10_fullscale",
        defenders=15, attackers=10,
        num_worlds=NUM_WORLDS,
        timesteps=STEPS_PER_PHASE,
        train_cfg=train_cfg,
        resume_path=model_path,
        strategy="mixed",
    )

    for n_d, n_a in [(15, 10), (8, 5), (20, 15), (5, 3)]:
        eval_report = run_evaluation(model_path, n_d, n_a)
        print(MetricsTracker.format_report(eval_report, f"Eval {n_d}v{n_a}"))

    # ==========================================================
    # Phase 5: Adversarial — train RL attacker, then retrain defender
    # ==========================================================
    defender_path = model_path

    # Train RL attackers vs frozen defender
    print(f"\n{'='*60}")
    print(f"  ADVERSARIAL: Training RL Attackers")
    print(f"{'='*60}")

    att_cfg = EnvConfig(
        mode="evade",
        num_defenders=15,
        num_attackers=10,
        k_opponents=3,
        k_teammates=2,
        defender_model_path=defender_path,
        **COMMON_KWARGS,
    )
    att_env = NewtonSwarmEnv(num_worlds=NUM_WORLDS, config=att_cfg)
    att_model = SAC(
        train_cfg.policy,
        att_env,
        learning_rate=linear_schedule(train_cfg.learning_rate),
        batch_size=4096,
        buffer_size=train_cfg.buffer_size * 10,
        learning_starts=train_cfg.learning_starts,
        gamma=train_cfg.gamma,
        tau=train_cfg.tau,
        ent_coef="auto",
        gradient_steps=64,
        policy_kwargs={"net_arch": train_cfg.net_arch},
        tensorboard_log=train_cfg.tensorboard_log,
        seed=train_cfg.seed,
        verbose=1,
        device="cuda",
    )
    att_model.learn(total_timesteps=STEPS_PER_PHASE, tb_log_name="phase5_attacker")
    attacker_path = os.path.join(train_cfg.model_dir, "phase5_attacker_final")
    att_model.save(attacker_path)
    att_env.close()

    # Retrain defenders vs RL attackers
    model_path, report_5 = train_phase(
        "phase5_15v10_adversarial",
        defenders=15, attackers=10,
        num_worlds=NUM_WORLDS,
        timesteps=STEPS_PER_PHASE,
        train_cfg=train_cfg,
        resume_path=defender_path,
        frozen_attacker_path=f"{attacker_path}.zip",
    )

    # Final generalization eval
    print(f"\n{'='*60}")
    print(f"  FINAL GENERALIZATION EVALUATION")
    print(f"{'='*60}")
    for n_d, n_a in [(5, 3), (8, 5), (10, 7), (15, 10), (20, 15)]:
        eval_report = run_evaluation(model_path, n_d, n_a)
        print(MetricsTracker.format_report(eval_report, f"Final Eval {n_d}v{n_a}"))
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": f"final_eval_{n_d}v{n_a}",
            **eval_report,
        }
        with open(METRICS_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")

    print(f"\n{'#'*60}")
    print(f"  PIPELINE COMPLETE")
    print(f"  Final model: {model_path}")
    print(f"  Metrics log: {METRICS_LOG}")
    print(f"  View in launcher: python -m eval.launcher")
    print(f"{'#'*60}")


if __name__ == "__main__":
    main()
