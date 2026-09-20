"""Render an AEGIS episode to an animated GIF (top-down tactical view).

Usage:
    python -m eval.render_aegis --model models/aegis_adv_defender_reuse_best/best_model.zip \
        --nm 5x8 --out models/eval_output/aegis_reuse_5v8.gif
    python -m eval.render_aegis --model ... --nm 7x5 --kamikaze --strategy mixed
"""

import argparse
import os

import numpy as np


def rollout(model_path, n, m, kamikaze, strategy, realism, seed, terrain="hills",
            target_speed=0.0, num_targets=1):
    import warp as wp
    wp.init()
    from stable_baselines3 import PPO
    from envs.aegis_env import AegisSwarmEnv, AegisConfig

    kw = dict(n_max=n, m_max=m, fixed_nm=(n, m), n_range=(n, n), m_range=(m, m),
              kamikaze=kamikaze, attacker_strategies=(strategy,), seed=seed,
              terrain=terrain, num_targets=num_targets,
              target_speed_range=(target_speed, target_speed))
    if realism:
        kw.update(num_obstacles=5, wind_enabled=True, wind_max_force=0.5,
                  wind_gust_prob=0.01, wind_gust_force=1.5, sensor_range=300.0)
    env = AegisSwarmEnv(num_worlds=1, config=kw and AegisConfig(**kw))
    model = PPO.load(model_path, device="cuda")

    obs = env.reset()
    # capture terrain meta BEFORE stepping: the loop ends after the world has
    # already auto-reset, so a post-loop capture reads the NEXT episode's world
    meta = dict(target=env.target_pos[0].copy(),
                hills=(env.obstacle_pos[0][:env.hill_count[0]].copy()
                       if env.hill_count[0] else None),
                boxes=(env.box_obs[0][:env.box_count[0]].copy()
                       if env.box_count[0] else None),
                hill_r=env.cfg.obstacle_radius, bounds=env.cfg.arena_bounds,
                targets=env.targets[0].copy())
    frames = []
    for _ in range(env.max_steps + 1):
        frames.append(dict(
            dp=env.def_pos[0].copy(), ap=env.att_pos[0].copy(),
            da=env.def_alive[0].copy(), aa=env.att_active[0].copy(),
            ab=env.att_breached[0].copy(), at=env.att_terrain[0].copy(),
            assign=env.assign[0].copy(), t=int(env.step_count[0]),
            ic=int(env.ep_intercepts[0]), tp=env.target_pos[0].copy(),
        ))
        acts, _ = model.predict(obs, deterministic=True)
        env.step_async(acts)
        obs, _, _, _ = env.step_wait()
        if env.step_count[0] < frames[-1]["t"]:   # world reset -> episode over
            break
    meta["outcome"] = env.pop_outcomes() or [{}]
    env.close()
    return frames, meta


def render(frames, meta, out_path, title, stride=2, fps=18, trail=18):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    b = meta["bounds"]
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    def draw(fi):
        ax.clear()
        ax.set_facecolor("#0d1117")
        ax.set_xlim(-b, b); ax.set_ylim(-b, b)
        ax.set_aspect("equal")
        ax.tick_params(colors="#666")
        for s in ax.spines.values():
            s.set_color("#333")
        f = frames[fi * stride]
        if meta["hills"] is not None:
            for h in meta["hills"]:
                ax.add_patch(plt.Circle((h[0], h[1]), meta["hill_r"],
                                        color="#3b3b2f", alpha=0.9, zorder=1))
        if meta.get("boxes") is not None:
            from matplotlib.patches import Rectangle
            for bx in meta["boxes"]:
                ax.add_patch(Rectangle((bx[0] - bx[2], bx[1] - bx[3]),
                                       2 * bx[2], 2 * bx[3],
                                       color="#44475e", alpha=0.95, zorder=1))
        tgts = meta.get("targets")
        if tgts is not None and len(tgts) > 1:
            ax.scatter(tgts[:, 0], tgts[:, 1], marker="*", s=420, c="#ffd700",
                       edgecolors="#fff", zorder=5, label="targets")
        else:
            tg = f.get("tp", meta["target"])
            tps = [frames[k * stride].get("tp", meta["target"]) for k in range(0, fi + 1)]
            ax.plot([p[0] for p in tps], [p[1] for p in tps],
                    c="#ffd700", alpha=0.35, lw=1.4, zorder=4)
            ax.scatter([tg[0]], [tg[1]], marker="*", s=420, c="#ffd700",
                       edgecolors="#fff", zorder=5, label="target")
        # trails
        lo = max(0, fi - trail)
        for d in range(len(f["dp"])):
            xs = [frames[k * stride]["dp"][d, 0] for k in range(lo, fi + 1)]
            ys = [frames[k * stride]["dp"][d, 1] for k in range(lo, fi + 1)]
            ax.plot(xs, ys, c="#2f81f7", alpha=0.25, lw=1, zorder=2)
        for a in range(len(f["ap"])):
            xs = [frames[k * stride]["ap"][a, 0] for k in range(lo, fi + 1)]
            ys = [frames[k * stride]["ap"][a, 1] for k in range(lo, fi + 1)]
            ax.plot(xs, ys, c="#f85149", alpha=0.25, lw=1, zorder=2)
        # assignment lines
        for d in range(len(f["dp"])):
            if f["da"][d] and f["assign"][d] >= 0 and f["aa"][f["assign"][d]]:
                a = f["assign"][d]
                ax.plot([f["dp"][d, 0], f["ap"][a, 0]],
                        [f["dp"][d, 1], f["ap"][a, 1]],
                        c="#8b949e", alpha=0.35, lw=0.7, ls=":", zorder=3)
        # drones
        alive_d = f["da"]
        ax.scatter(f["dp"][alive_d, 0], f["dp"][alive_d, 1], s=90, c="#2f81f7",
                   marker="^", edgecolors="#a5c8ff", zorder=6, label="defenders")
        dead_d = ~alive_d
        ax.scatter(f["dp"][dead_d, 0], f["dp"][dead_d, 1], s=50, c="#30363d",
                   marker="x", zorder=4)
        act = f["aa"]
        ax.scatter(f["ap"][act, 0], f["ap"][act, 1], s=90, c="#f85149",
                   marker="v", edgecolors="#ffb3ad", zorder=6, label="attackers")
        killed = ~act & ~f["ab"]
        ax.scatter(f["ap"][killed, 0], f["ap"][killed, 1], s=60, c="#7d3a3a",
                   marker="x", zorder=4)
        breached = f["ab"]
        ax.scatter(f["ap"][breached, 0], f["ap"][breached, 1], s=120, c="#ff6b00",
                   marker="P", zorder=5)
        ax.set_title(
            f"{title}   t={f['t']/12.0:5.1f}s   "
            f"defenders {alive_d.sum()}/{len(alive_d)}   "
            f"attackers {act.sum()}/{len(act)}   kills {f['ic']}   "
            f"breaches {int(f['ab'].sum())}",
            color="#e6edf3", fontsize=10)
        ax.legend(loc="upper right", fontsize=8, facecolor="#161b22",
                  labelcolor="#e6edf3", edgecolor="#333")

    n_frames = (len(frames) + stride - 1) // stride
    anim = FuncAnimation(fig, draw, frames=n_frames, interval=1000 / fps)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return n_frames


def main():
    p = argparse.ArgumentParser(description="Render AEGIS episode GIF")
    p.add_argument("--model", required=True)
    p.add_argument("--nm", default="5x8", help="NxM, e.g. 5x8")
    p.add_argument("--kamikaze", action="store_true",
                   help="kamikaze rules (default: reusable interceptor)")
    p.add_argument("--strategy", default="mixed",
                   choices=["direct", "flanking", "mixed", "jink",
                            "saturation", "waves", "decoy"])
    p.add_argument("--terrain", default="hills",
                   choices=["hills", "urban", "maze", "none"])
    p.add_argument("--target-speed", type=float, default=0.0,
                   help="moving-target (convoy) speed in m/s")
    p.add_argument("--num-targets", type=int, default=1,
                   help="split defense: number of protected points")
    p.add_argument("--plain", action="store_true", help="no hills/wind")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="")
    p.add_argument("--fps", type=int, default=18)
    p.add_argument("--stride", type=int, default=2)
    args = p.parse_args()

    n, m = map(int, args.nm.split("x"))
    tslug = "" if args.terrain == "hills" else f"_{args.terrain}"
    out = args.out or (f"models/eval_output/aegis_"
                       f"{'kam' if args.kamikaze else 'reuse'}_{n}v{m}_{args.strategy}{tslug}.gif")
    frames, meta = rollout(args.model, n, m, args.kamikaze, args.strategy,
                           not args.plain, args.seed, terrain=args.terrain,
                           target_speed=args.target_speed, num_targets=args.num_targets)
    o = (meta["outcome"] or [{}])[0]
    title = f"AEGIS {n}v{m} [{args.strategy}]" + (" kamikaze" if args.kamikaze else " reusable")
    nf = render(frames, meta, out, title, stride=args.stride, fps=args.fps)
    print(f"episode: {len(frames)} steps ({len(frames)/12.0:.1f}s sim) | "
          f"outcome: {'WIN' if o.get('win') else 'loss'} "
          f"kills={o.get('intercepts')} breaches={o.get('breaches')}")
    print(f"wrote {out} ({nf} frames)")


if __name__ == "__main__":
    main()
