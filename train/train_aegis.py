"""AEGIS staged training pipeline (docs/AEGIS_PLAN.md §3.6).

One shared PPO policy over per-drone streams; env-side allocation handles
coordination. Stages 0-3 with hard gates: fail a gate -> extend once by +20M ->
fail again -> HALT (never blind warm-start). Checkpoint selection is by LOWEST
BREACH RATE on fixed seeds, never mean reward.

Usage:
    python -m train.train_aegis
    python -m train.train_aegis --start-stage 1 --resume models/aegis_stage0_best/best_model.zip
"""

import argparse
import os
import sys
import time
from dataclasses import replace

import numpy as np


LOG_FILE = "logs/train_aegis.log"


def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    os.makedirs("logs", exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def stage_configs():
    """Stage table from the plan. Returns list of (name, env_kwargs, num_worlds, steps, gate)."""
    return [
        dict(
            name="stage0_pursuit",
            desc="1v1 pursuit sanity — no hills/wind, direct 0.8x",
            env=dict(n_max=1, m_max=1, n_range=(1, 1), m_range=(1, 1),
                     num_obstacles=0, wind_enabled=False, sensor_range=350.0,
                     attacker_speed_scale=0.8, attacker_strategies=("direct",),
                     cull_prob=0.0),
            num_worlds=2048, steps=20_000_000,
            gate=dict(win=0.90, bad_death_rate=0.02),
        ),
        dict(
            name="stage1_core_nvm",
            desc="core NvM — N~U[2,4] vs M~U[1,3], direct 1.0x",
            env=dict(n_max=4, m_max=3, n_range=(2, 4), m_range=(1, 3),
                     num_obstacles=0, wind_enabled=False, sensor_range=350.0,
                     attacker_speed_scale=1.0, attacker_strategies=("direct",),
                     cull_prob=0.0),
            num_worlds=1024, steps=40_000_000,
            gate=dict(win=0.80, mean_leakers=0.15, bad_death_rate=0.03),
        ),
        dict(
            name="stage2_fullrange",
            desc="full range + terrain + dropout — N~U[3,8] vs M~U[1,8], hills, wind, culling",
            env=dict(n_max=8, m_max=8, n_range=(3, 8), m_range=(1, 8),
                     num_obstacles=5, wind_enabled=True, wind_max_force=0.5,
                     sensor_range=300.0, attacker_speed_scale=1.0,
                     attacker_strategies=("direct", "flanking"), cull_prob=0.3),
            num_worlds=512, steps=60_000_000,
            gate=dict(win=0.70),
        ),
        dict(
            name="stage3_realism",
            desc="realism + self-healing — mixed/flanking/jink, gusts, noise, variance",
            env=dict(n_max=8, m_max=8, n_range=(3, 8), m_range=(1, 8),
                     num_obstacles=5, wind_enabled=True, wind_max_force=0.5,
                     wind_gust_prob=0.01, wind_gust_force=1.5,
                     sensor_range=300.0, attacker_speed_scale=1.0,
                     attacker_strategies=("mixed", "flanking", "jink"),
                     obs_noise_std=0.02, drone_mass_variance=0.1,
                     attacker_speed_variance=0.1, cull_prob=0.3),
            num_worlds=512, steps=60_000_000,
            gate=dict(win=0.60, mean_leakers=0.5),
        ),
    ]


def parse_overrides(pairs):
    """--set key=val overrides for AegisConfig; values parsed as python literals."""
    import ast
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        try:
            out[k] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k] = v
    return out


def make_env(env_kwargs, num_worlds, seed, overrides=None):
    from envs.aegis_env import AegisSwarmEnv, AegisConfig
    cfg = AegisConfig(seed=seed, **{**env_kwargs, **(overrides or {})})
    return AegisSwarmEnv(num_worlds=num_worlds, config=cfg)


def collect_outcomes(model, env, n_episodes, deterministic=True):
    """Run policy on env until n_episodes outcomes collected. Returns list of dicts."""
    env.pop_outcomes()  # clear stale
    obs = env.reset()
    outcomes = []
    guard = 0
    while len(outcomes) < n_episodes and guard < 200_000:
        actions, _ = model.predict(obs, deterministic=deterministic)
        env.step_async(actions)
        obs, _, _, _ = env.step_wait()
        outcomes.extend(env.pop_outcomes())
        guard += 1
    return outcomes[:n_episodes]


def outcome_metrics(outcomes):
    if not outcomes:
        return dict(episodes=0, win=0.0, breach_rate=1.0, mean_leakers=99.0,
                    bad_death_rate=1.0, mean_intercepts=0.0, mean_first_intercept_s=-1.0)
    n = len(outcomes)
    tot_att = sum(o["m"] for o in outcomes)
    tot_def = sum(o["n"] for o in outcomes)
    tot_breach = sum(o["breaches"] for o in outcomes)
    fi = [o["first_intercept"] / 12.0 for o in outcomes if o["first_intercept"] >= 0]
    tot_stranded = sum(o.get("stranded", 0) for o in outcomes)
    return dict(
        episodes=n,
        win=sum(o["win"] for o in outcomes) / n,
        breach_rate=(tot_breach + tot_stranded) / max(tot_att, 1),
        true_breach_rate=tot_breach / max(tot_att, 1),
        mean_leakers=(tot_breach + tot_stranded) / n,
        bad_death_rate=sum(o["bad_deaths"] for o in outcomes) / max(tot_def, 1),
        mean_intercepts=sum(o["intercepts"] for o in outcomes) / n,
        mean_first_intercept_s=float(np.mean(fi)) if fi else -1.0,
        timeout_rate=sum(o["timeout"] for o in outcomes) / n,
        terrain_kill_rate=sum(o["terrain_kills"] for o in outcomes) / max(tot_att, 1),
    )


def fmt(m):
    return (f"win={m['win']:.2f} breach_rate={m['breach_rate']:.3f} "
            f"true_breach={m.get('true_breach_rate', 0):.3f} "
            f"leakers={m['mean_leakers']:.2f} bad_death={m['bad_death_rate']:.3f} "
            f"intercepts={m['mean_intercepts']:.2f} t_first={m['mean_first_intercept_s']:.1f}s "
            f"timeout={m.get('timeout_rate', 0):.2f} (n={m['episodes']})")


class BreachRateEvalCallback:
    """Periodic fixed-seed eval; saves best model by LOWEST breach rate."""

    def __new__(cls, *args, **kwargs):
        from stable_baselines3.common.callbacks import BaseCallback

        class _Impl(BaseCallback):
            def __init__(self, eval_env, eval_every_calls, best_dir, train_env,
                         n_episodes=96, verbose=0):
                super().__init__(verbose)
                self.eval_env = eval_env
                self.eval_every = max(1, eval_every_calls)
                self.best_dir = best_dir
                self.train_env = train_env
                self.n_episodes = n_episodes
                self.best_breach = np.inf

            def _on_step(self):
                if self.n_calls % self.eval_every != 0:
                    return True
                self.eval_env.seed(12345)
                m = outcome_metrics(collect_outcomes(self.model, self.eval_env,
                                                     self.n_episodes))
                for k, v in m.items():
                    self.logger.record(f"aegis_eval/{k}", v)
                # training-env telemetry (pop clears; telemetry-only)
                tm = outcome_metrics(self.train_env.pop_outcomes())
                if tm["episodes"] > 0:
                    self.logger.record("aegis_train/win", tm["win"])
                    self.logger.record("aegis_train/breach_rate", tm["breach_rate"])
                    self.logger.record("aegis_train/mean_intercepts", tm["mean_intercepts"])
                self.logger.record("aegis_train/effective_sample_fraction",
                                   self.train_env.effective_sample_fraction())
                log(f"  [eval @ {self.num_timesteps:,}] {fmt(m)} | "
                    f"train: {fmt(tm) if tm['episodes'] else 'n/a'} | "
                    f"esf={self.train_env.effective_sample_fraction():.2f}")
                if m["breach_rate"] < self.best_breach and m["episodes"] >= self.n_episodes // 2:
                    self.best_breach = m["breach_rate"]
                    os.makedirs(self.best_dir, exist_ok=True)
                    self.model.save(os.path.join(self.best_dir, "best_model.zip"))
                    log(f"  new best (breach_rate={m['breach_rate']:.3f}) -> {self.best_dir}/best_model.zip")
                return True

        return _Impl(*args, **kwargs)


def lr_schedule(progress_remaining):
    return 1e-4 + (3e-4 - 1e-4) * progress_remaining


def run_stage(args, stage, stage_idx, resume_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    steps = int(stage["steps"] * args.timesteps_scale)
    num_worlds = max(8, int(stage["num_worlds"] * args.num_worlds_scale))
    log(f"=== Stage {stage_idx}: {stage['desc']} | {steps:,} steps | {num_worlds} worlds ===")

    ov = parse_overrides(args.set)
    env = make_env(stage["env"], num_worlds, seed=1000 + stage_idx, overrides=ov)
    eval_env = make_env({**stage["env"], "cull_prob": 0.0}, 32, seed=12345, overrides=ov)
    n_envs = env.n_envs
    log(f"  n_envs={n_envs} (streams) | rollout={128 * n_envs:,} transitions")

    if resume_path and os.path.exists(resume_path):
        log(f"  warm-start from {resume_path}")
        model = PPO.load(resume_path, env=env, device="cuda",
                         tensorboard_log="logs/tensorboard", verbose=1)
        model.learning_rate = lr_schedule
        model._setup_lr_schedule()
        for pg in model.policy.optimizer.param_groups:
            pg["lr"] = 3e-4
        model.policy.optimizer.state.clear()
        log("  optimizer reset (fresh Adam state, lr schedule restarted)")
    else:
        log("  from scratch")
        model = PPO(
            "MlpPolicy", env,
            learning_rate=lr_schedule,
            n_steps=128, batch_size=16384, n_epochs=4,
            gamma=0.995, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            device="cuda", verbose=1, tensorboard_log="logs/tensorboard",
        )

    name = f"aegis{args.tag}_{stage['name']}"
    best_dir = f"models/{name}_best"
    callbacks = [
        CheckpointCallback(save_freq=max(1, steps // 10 // n_envs),
                           save_path=f"models/{name}_ckpt/", name_prefix=name),
        BreachRateEvalCallback(eval_env, eval_every_calls=max(1, steps // 12 // n_envs),
                               best_dir=best_dir, train_env=env),
    ]

    model.learn(total_timesteps=steps, callback=callbacks,
                reset_num_timesteps=True, tb_log_name=name, progress_bar=False)
    final = f"models/{name}_final.zip"
    model.save(final)
    log(f"  saved {final}")

    best = os.path.join(best_dir, "best_model.zip")
    gate_model_path = best if os.path.exists(best) else final

    # ---- GATE ----
    passed = True
    if not args.skip_gates:
        gate_model = PPO.load(gate_model_path, env=eval_env, device="cuda")
        eval_env.seed(99)
        m = outcome_metrics(collect_outcomes(gate_model, eval_env, 256))
        log(f"  GATE eval: {fmt(m)} vs {stage['gate']}")
        passed = _gate_ok(m, stage["gate"])
        if not passed:
            log(f"  GATE FAILED — extending stage by +20M steps (one retry)")
            model.learn(total_timesteps=20_000_000, callback=callbacks,
                        reset_num_timesteps=False, tb_log_name=name + "_ext",
                        progress_bar=False)
            model.save(final)
            gate_model_path = best if os.path.exists(best) else final
            gate_model = PPO.load(gate_model_path, env=eval_env, device="cuda")
            eval_env.seed(99)
            m = outcome_metrics(collect_outcomes(gate_model, eval_env, 256))
            log(f"  GATE re-eval: {fmt(m)}")
            passed = _gate_ok(m, stage["gate"])
            if not passed:
                log(f"  GATE FAILED TWICE — HALTING. Do not warm-start from this. "
                    f"Diagnose with eval/eval_nvm_grid.py and analysis/metrics.py.")
    env.close()
    eval_env.close()
    return gate_model_path, passed


def _gate_ok(m, gate):
    if "win" in gate and m["win"] < gate["win"]:
        return False
    if "mean_leakers" in gate and m["mean_leakers"] > gate["mean_leakers"]:
        return False
    if "bad_death_rate" in gate and m["bad_death_rate"] > gate["bad_death_rate"]:
        return False
    return True


def main():
    p = argparse.ArgumentParser(description="AEGIS staged training")
    p.add_argument("--start-stage", type=int, default=0)
    p.add_argument("--resume", type=str, default="")
    p.add_argument("--num-worlds-scale", type=float, default=1.0)
    p.add_argument("--timesteps-scale", type=float, default=1.0)
    p.add_argument("--skip-gates", action="store_true")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                   help="AegisConfig override applied to every stage (repeatable), "
                        "e.g. --set r_bad_death=-30 --set kamikaze=False")
    p.add_argument("--tag", type=str, default="",
                   help="suffix for model/tb names, e.g. _patch -> models/aegis_patch_stage0_*")
    args = p.parse_args()
    if args.tag and not args.tag.startswith("_"):
        args.tag = "_" + args.tag

    import warp as wp
    wp.init()

    stages = stage_configs()
    log(f"AEGIS training start: stages {args.start_stage}..{len(stages)-1} "
        f"(scales: worlds x{args.num_worlds_scale}, steps x{args.timesteps_scale})")

    resume = args.resume
    for i in range(args.start_stage, len(stages)):
        resume, passed = run_stage(args, stages[i], i, resume)
        if not passed and not args.skip_gates:
            log(f"HALTED at stage {i}. Best-so-far model: {resume}")
            sys.exit(1)

    log("=" * 60)
    log(f"AEGIS training complete. Final model: {resume}")
    log("View in launcher: python -m eval.launcher | Grid eval: "
        f"python -m eval.eval_nvm_grid --model {resume}")
    log("=" * 60)


if __name__ == "__main__":
    main()
