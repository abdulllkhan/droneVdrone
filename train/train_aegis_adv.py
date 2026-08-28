"""Phase-3 adversarial co-training (docs/AEGIS_PLAN.md Phase 3).

Round A — RL attackers (evade mode) learn to penetrate the frozen AEGIS defender.
Round B — the defender retrains against a MIXTURE of scripted + frozen RL attackers
(mixture pools prevent catastrophic forgetting — established in the pre-AEGIS era).

Usage:
    python -m train.train_aegis_adv \
        --defender models/aegis_patch_stage3_realism_best/best_model.zip
"""

import argparse
import os

import numpy as np

from train.train_aegis import (log, make_env, collect_outcomes, outcome_metrics,
                               fmt, lr_schedule, parse_overrides)

REALISM = dict(n_max=8, m_max=8, n_range=(3, 8), m_range=(1, 8),
               num_obstacles=5, wind_enabled=True, wind_max_force=0.5,
               wind_gust_prob=0.01, wind_gust_force=1.5, obs_noise_std=0.02,
               sensor_range=300.0, attacker_speed_scale=1.0,
               drone_mass_variance=0.1, attacker_speed_variance=0.1)

PPO_KW = dict(n_steps=128, batch_size=16384, n_epochs=4, gamma=0.995,
              gae_lambda=0.95, clip_range=0.2, ent_coef=0.005, vf_coef=0.5,
              max_grad_norm=0.5, device="cuda", verbose=1,
              tensorboard_log="logs/tensorboard",
              policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))


def eval_callback(eval_env, every_calls, best_dir, train_env, maximize_breach):
    """Fixed-seed eval; saves best by breach_rate (min for defender, max for attacker)."""
    from stable_baselines3.common.callbacks import BaseCallback

    class _Cb(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.best = -np.inf if maximize_breach else np.inf

        def _on_step(self):
            if self.n_calls % every_calls != 0:
                return True
            eval_env.seed(12345)
            m = outcome_metrics(collect_outcomes(self.model, eval_env, 96))
            for k, v in m.items():
                self.logger.record(f"adv_eval/{k}", v)
            log(f"  [eval @ {self.num_timesteps:,}] {fmt(m)} | "
                f"esf={train_env.effective_sample_fraction():.2f}")
            # attackers are judged on TRUE penetration (stalling at timeout counts
            # as stranded in breach_rate and must not select checkpoints)
            score = m["true_breach_rate"] if maximize_breach else m["breach_rate"]
            better = score > self.best if maximize_breach else score < self.best
            if better and m["episodes"] >= 48:
                self.best = score
                os.makedirs(best_dir, exist_ok=True)
                self.model.save(os.path.join(best_dir, "best_model.zip"))
                log(f"  new best (breach_rate={score:.3f}) -> {best_dir}/best_model.zip")
            return True

    return _Cb()


def base_env_kw(args):
    kw = dict(REALISM)
    if args.kamikaze_false:
        kw["kamikaze"] = False
    kw.update(parse_overrides(args.env_set))
    return kw


def train_round_a(args):
    """RL attackers vs the frozen defender."""
    from stable_baselines3 import PPO
    tag = args.round_tag
    log(f"=== ROUND A{tag}: RL attackers vs frozen defender ({args.defender}) ===")
    kw = {**base_env_kw(args), "mode": "evade", "defender_model_path": args.defender,
          "cull_prob": 0.0}
    env = make_env(kw, args.num_worlds, seed=7000 + sum(map(ord, tag)))
    eval_env = make_env(kw, 32, seed=71234)
    log(f"  n_envs={env.n_envs} attacker streams")

    model = PPO("MlpPolicy", env, learning_rate=lr_schedule, **PPO_KW)
    best_dir = f"models/aegis_adv_attacker{tag}_best"
    cb = eval_callback(eval_env, max(1, args.steps_a // 12 // env.n_envs),
                       best_dir, env, maximize_breach=True)
    model.learn(total_timesteps=args.steps_a, callback=cb,
                reset_num_timesteps=True, tb_log_name=f"aegis_adv_attacker{tag}")
    final = f"models/aegis_adv_attacker{tag}_final.zip"
    model.save(final)

    best = os.path.join(best_dir, "best_model.zip")
    a_path = best if os.path.exists(best) else final
    eval_env.seed(99)
    gate_model = PPO.load(a_path, device="cuda")
    m = outcome_metrics(collect_outcomes(gate_model, eval_env, 256))
    log(f"  ROUND A gate: {fmt(m)} (attacker success = breach_rate; "
        f"scripted attackers manage ~0.05 vs this defender)")
    env.close(); eval_env.close()
    if m["true_breach_rate"] < 0.08:
        log("  ROUND A verdict: RL attackers did NOT meaningfully beat the defender "
            "(<0.10). Round B unnecessary — the defender is already robust. Stopping.")
        return a_path, False
    log(f"  ROUND A verdict: adversarial gap found (breach_rate={m['breach_rate']:.2f}).")
    return a_path, True


def train_round_b(args, attacker_path):
    """Defender retrains vs mixture of scripted + the whole RL attacker POOL."""
    from stable_baselines3 import PPO
    tag = args.round_tag
    pool = tuple(args.pool_list + [attacker_path])
    log(f"=== ROUND B{tag}: defender vs mixture (scripted + pool of {len(pool)}) ===")
    for p in pool:
        log(f"    pool member: {p}")
    kw = {**base_env_kw(args), "mode": "defend", "cull_prob": 0.3,
          "attacker_model_paths": pool,
          "attacker_strategies": ("mixed", "flanking", "jink", "rl", "rl")}
    env = make_env(kw, args.num_worlds, seed=8000 + sum(map(ord, tag)))
    eval_env = make_env({**kw, "cull_prob": 0.0}, 32, seed=81234)

    model = PPO.load(args.defender, env=env, device="cuda",
                     tensorboard_log="logs/tensorboard", verbose=1)
    model.learning_rate = lr_schedule
    model._setup_lr_schedule()
    for pg in model.policy.optimizer.param_groups:
        pg["lr"] = 3e-4
    model.policy.optimizer.state.clear()

    best_dir = f"models/aegis_adv_defender{tag}_best"
    cb = eval_callback(eval_env, max(1, args.steps_b // 12 // env.n_envs),
                       best_dir, env, maximize_breach=False)
    model.learn(total_timesteps=args.steps_b, callback=cb,
                reset_num_timesteps=True, tb_log_name=f"aegis_adv_defender{tag}")
    final = f"models/aegis_adv_defender{tag}_final.zip"
    model.save(final)

    best = os.path.join(best_dir, "best_model.zip")
    b_path = best if os.path.exists(best) else final
    eval_env.seed(99)
    gate_model = PPO.load(b_path, device="cuda")
    m = outcome_metrics(collect_outcomes(gate_model, eval_env, 256))
    log(f"  ROUND B gate vs mixture: {fmt(m)} (target win>=0.70)")
    env.close(); eval_env.close()
    return b_path


def main():
    p = argparse.ArgumentParser(description="AEGIS Phase-3 adversarial co-training")
    p.add_argument("--defender", type=str,
                   default="models/aegis_patch_stage3_realism_best/best_model.zip")
    p.add_argument("--num-worlds", type=int, default=512)
    p.add_argument("--steps-a", type=int, default=40_000_000)
    p.add_argument("--steps-b", type=int, default=60_000_000)
    p.add_argument("--skip-a", type=str, default="",
                   help="skip round A, use this attacker model path")
    p.add_argument("--pool", type=str, default="",
                   help="comma-separated prior attacker models for the round-B mixture")
    p.add_argument("--round-tag", type=str, default="",
                   help="suffix for model names, e.g. _r2")
    p.add_argument("--kamikaze-false", action="store_true",
                   help="reuse-defender track: defenders survive intercepts")
    p.add_argument("--env-set", action="append", default=[], metavar="KEY=VAL",
                   help="AegisConfig override for both rounds (repeatable), "
                        "e.g. --env-set \"terrains=('hills','urban','maze')\"")
    args = p.parse_args()
    if args.round_tag and not args.round_tag.startswith("_"):
        args.round_tag = "_" + args.round_tag
    args.pool_list = [s for s in args.pool.split(",") if s]

    import warp as wp
    wp.init()

    if args.skip_a:
        a_path, gap = args.skip_a, True
    else:
        a_path, gap = train_round_a(args)
    if gap:
        b_path = train_round_b(args, a_path)
        log(f"ADVERSARIAL CO-TRAINING COMPLETE. Attacker: {a_path} | Defender: {b_path}")
    else:
        log(f"ADVERSARIAL ROUND A COMPLETE (no gap). Attacker: {a_path}")


if __name__ == "__main__":
    main()
