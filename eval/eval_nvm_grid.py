"""NvM grid evaluation for AEGIS (docs/AEGIS_PLAN.md §3.8).

Evaluates one policy (or the scripted allocator+pursuit baseline) across a grid of
(N defenders, M attackers) cells and writes CSV + JSON + a win-rate heatmap.
The trained policy must beat the baseline's breach rate in >=80% of cells or the
learning is not paying for itself.

Usage:
    python -m eval.eval_nvm_grid --baseline
    python -m eval.eval_nvm_grid --model models/aegis_stage3_realism_best/best_model.zip
    python -m eval.eval_nvm_grid --model M.zip --cells 3x2,7x5 --episodes 20
"""

import argparse
import csv
import json
import os

import numpy as np

DEFAULT_N = [1, 3, 5, 7, 10, 15]
DEFAULT_M = [1, 2, 3, 5, 8, 10]
HOLDOUT = [(6, 4), (9, 6)]


def build_cells(spec):
    if spec and spec != "auto":
        return [tuple(map(int, c.split("x"))) for c in spec.split(",")]
    cells = [(n, m) for n in DEFAULT_N for m in DEFAULT_M]
    return cells + HOLDOUT


def make_env(n, m, strategy, realism, culled, seed=4242):
    from envs.aegis_env import AegisSwarmEnv, AegisConfig
    kw = dict(n_max=n, m_max=m, fixed_nm=(n, m), n_range=(n, n), m_range=(m, m),
              attacker_strategies=(strategy,), seed=seed,
              cull_prob=1.0 if culled else 0.0)
    if realism:
        kw.update(num_obstacles=5, wind_enabled=True, wind_max_force=0.5,
                  wind_gust_prob=0.01, wind_gust_force=1.5, obs_noise_std=0.02,
                  sensor_range=300.0)
    return AegisSwarmEnv(num_worlds=16, config=AegisConfig(**kw))


def baseline_actions(env):
    """Scripted allocator + pursuit-to-PIP baseline (no learning)."""
    from envs.allocator import assigned_tgo
    goal = env.stations.copy()
    has_t = env.assign >= 0
    a_idx = np.clip(env.assign, 0, None)[..., None]
    apos = np.take_along_axis(env.att_pos, a_idx, axis=1)
    avel = np.take_along_axis(env.att_vel, a_idx, axis=1)
    tgo = assigned_tgo(env.def_pos, env.cfg.defender_speed_limit,
                       env.att_pos, env.att_vel, env.assign)
    pip = apos + avel * tgo[..., None]
    goal[has_t] = pip[has_t]
    d = goal - env.def_pos
    norm = np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), 1e-6)
    return (d / norm).reshape(env.n_envs, 3).astype(np.float32)


def run_cell(model, n, m, strategy, episodes, realism, culled):
    env = make_env(n, m, strategy, realism, culled)
    env.pop_outcomes()
    obs = env.reset()
    outs, guard = [], 0
    while len(outs) < episodes and guard < 100_000:
        if model is None:
            acts = baseline_actions(env)
        else:
            acts, _ = model.predict(obs, deterministic=True)
        env.step_async(acts)
        obs, _, _, _ = env.step_wait()
        outs.extend(env.pop_outcomes())
        guard += 1
    env.close()
    outs = outs[:episodes]
    ne = max(len(outs), 1)
    tot_att = sum(o["m"] for o in outs) or 1
    tot_def = sum(o["n"] for o in outs) or 1
    def_deaths = sum(o["n"] - o["defenders_left"] for o in outs)
    fi = [o["first_intercept"] / 12.0 for o in outs if o["first_intercept"] >= 0]
    return dict(
        n=n, m=m, strategy=strategy, episodes=len(outs),
        win_rate=sum(o["win"] for o in outs) / ne,
        breach_rate=sum(o["breaches"] + o.get("stranded", 0) for o in outs) / tot_att,
        mean_leakers=sum(o["breaches"] + o.get("stranded", 0) for o in outs) / ne,
        exchange_ratio=sum(o["intercepts"] for o in outs) / max(def_deaths, 1),
        bad_death_rate=sum(o["bad_deaths"] for o in outs) / tot_def,
        terrain_kill_rate=sum(o["terrain_kills"] for o in outs) / tot_att,
        time_to_first_intercept_s=float(np.mean(fi)) if fi else -1.0,
        mean_length_s=sum(o["length"] for o in outs) / ne / 12.0,
    )


def heatmap(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = sorted({r["n"] for r in rows})
    ms = sorted({r["m"] for r in rows})
    grid = np.full((len(ns), len(ms)), np.nan)
    for r in rows:
        i, j = ns.index(r["n"]), ms.index(r["m"])
        v = grid[i, j]
        grid[i, j] = r["win_rate"] if np.isnan(v) else (v + r["win_rate"]) / 2
    fig, ax = plt.subplots(figsize=(1.2 * len(ms) + 2, 1.0 * len(ns) + 2))
    im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(ms)), [f"M={m}" for m in ms])
    ax.set_yticks(range(len(ns)), [f"N={n}" for n in ns])
    for i in range(len(ns)):
        for j in range(len(ms)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.0%}", ha="center", va="center", fontsize=9)
    ax.set_title("Zero-leaker win rate")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="AEGIS NvM grid evaluation")
    p.add_argument("--model", type=str, default="")
    p.add_argument("--baseline", action="store_true")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--cells", type=str, default="auto")
    p.add_argument("--strategies", type=str, default="direct,flanking,mixed")
    p.add_argument("--realism", action="store_true")
    p.add_argument("--culled", action="store_true")
    p.add_argument("--out", type=str, default="models/eval_output")
    args = p.parse_args()

    import warp as wp
    wp.init()

    model = None
    tag = "baseline"
    if not args.baseline:
        if not args.model:
            p.error("--model required unless --baseline")
        from stable_baselines3 import PPO
        model = PPO.load(args.model, device="cuda")
        tag = os.path.splitext(os.path.basename(args.model))[0]
        if tag == "best_model":
            tag = os.path.basename(os.path.dirname(args.model))

    cells = build_cells(args.cells)
    strategies = args.strategies.split(",")
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for (n, m) in cells:
        for s in strategies:
            r = run_cell(model, n, m, s, args.episodes, args.realism, args.culled)
            rows.append(r)
            extra = f" xr={r['exchange_ratio']:.2f}" if n < m else ""
            print(f"  {n}v{m} [{s:9s}] win={r['win_rate']:.0%} "
                  f"breach={r['breach_rate']:.3f} leakers={r['mean_leakers']:.2f}"
                  f"{extra} bad={r['bad_death_rate']:.2%}", flush=True)

    csv_path = os.path.join(args.out, f"nvm_grid_{tag}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, f"nvm_grid_{tag}.json"), "w") as f:
        json.dump(rows, f, indent=2)
    heatmap(rows, os.path.join(args.out, f"nvm_grid_{tag}_winrate.png"))
    print(f"\nWrote {csv_path} (+ .json, heatmap PNG)")

    # Compare vs baseline if available
    base_csv = os.path.join(args.out, "nvm_grid_baseline.csv")
    if model is not None and os.path.exists(base_csv):
        with open(base_csv) as f:
            base = {(int(r["n"]), int(r["m"]), r["strategy"]): float(r["breach_rate"])
                    for r in csv.DictReader(f)}
        wins = tot = 0
        for r in rows:
            k = (r["n"], r["m"], r["strategy"])
            if k in base:
                tot += 1
                wins += r["breach_rate"] < base[k] + 1e-9
        if tot:
            print(f"Model beats baseline breach rate in {wins}/{tot} cells "
                  f"({wins/tot:.0%}; plan target >=80%)")


if __name__ == "__main__":
    main()
