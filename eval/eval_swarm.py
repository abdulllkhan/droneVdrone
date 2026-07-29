"""
Swarm evaluation — Run NvM episodes and generate trajectory plots.

Usage:
    # Defender vs scripted attackers (small arena)
    python -m eval.eval_swarm --model models/swarm_2v2_final.zip --defenders 2 --attackers 2

    # Ukraine 7v5 scenario (1km arena, hills, detection)
    python3 -m eval.eval_swarm --model models/ukraine_7v5_stage1_baseline_final.zip --ukraine --gif

    # Ukraine with flanking attackers
    python3 -m eval.eval_swarm --model models/ukraine_7v5_stage4_evasion_final.zip --ukraine --strategy flanking --gif

    # Defender vs RL attacker model
    python -m eval.eval_swarm --model models/swarm_2v2_final.zip --defenders 2 --attackers 2 --attacker-model models/swarm_adv_attacker_2v2_round1.zip

    # With GIF and more episodes
    python -m eval.eval_swarm --model models/swarm_3v2_final.zip --defenders 3 --attackers 2 --gif --episodes 20
"""

import argparse
import os

import numpy as np
import warp as wp
from stable_baselines3 import SAC, PPO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter


def run_swarm_episode(model, env, max_steps=480):
    """Run one episode, return trajectory data and outcome."""
    N = env.N  # defenders
    M = env.M  # attackers

    obs = env.reset()
    target_pos = env.target_pos_np.copy()

    def_trajs = [[] for _ in range(N)]
    att_trajs = [[] for _ in range(M)]
    reward_history = []

    outcome = "timeout"
    end_step = max_steps

    for step in range(max_steps):
        actions, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(actions)

        body_q = env.state_0.body_q.numpy()
        bpw = env.bodies_per_world
        for d in range(N):
            def_trajs[d].append(body_q[d][:3].copy())
        for a in range(M):
            att_trajs[a].append(body_q[N + a][:3].copy())

        reward_history.append(float(np.mean(rewards)))

        if np.any(dones):
            info = infos[0]
            if info.get("all_eliminated"):
                outcome = "all_intercepted"
            elif info.get("target_reached"):
                outcome = "target_reached"
            else:
                outcome = "done"
            end_step = step + 1
            break

    return {
        "def_trajs": [np.array(t) for t in def_trajs],
        "att_trajs": [np.array(t) for t in att_trajs],
        "target_pos": target_pos,
        "obstacle_pos": env.obstacle_pos_buf.numpy()[0] if env._num_obstacles > 0 else None,
        "outcome": outcome,
        "steps": end_step,
        "rewards": reward_history,
        "N": N,
        "M": M,
    }


def plot_episode_grid(episodes, save_path, title="Swarm Episodes"):
    """Plot a grid of episode 3D trajectories."""
    n = len(episodes)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6 * cols, 5 * rows))

    blue_shades = ["#2196F3", "#1565C0", "#0D47A1"]
    red_shades = ["#F44336", "#C62828", "#B71C1C"]

    for idx, ep in enumerate(episodes):
        ax = fig.add_subplot(rows, cols, idx + 1, projection="3d")

        for d in range(ep["N"]):
            t = ep["def_trajs"][d]
            c = blue_shades[d % len(blue_shades)]
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=c, linewidth=1.5,
                    label=f"Def {d}" if idx == 0 else None)
            ax.scatter(*t[0], color=c, marker="o", s=60)
            ax.scatter(*t[-1], color=c, marker="X", s=80)

        for a in range(ep["M"]):
            t = ep["att_trajs"][a]
            c = red_shades[a % len(red_shades)]
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=c, linewidth=1.5, linestyle="--",
                    label=f"Att {a}" if idx == 0 else None)
            ax.scatter(*t[0], color=c, marker="o", s=60)
            ax.scatter(*t[-1], color=c, marker="*", s=80)

        ax.scatter(*ep["target_pos"], color="#4CAF50", marker="D", s=120,
                   label="Target" if idx == 0 else None)

        if ep["obstacle_pos"] is not None:
            for o in range(len(ep["obstacle_pos"])):
                p = ep["obstacle_pos"][o]
                hill_h = p[2] * 2  # obstacle z = half-height, total = 2x
                ax.plot([p[0], p[0]], [p[1], p[1]], [0, hill_h],
                        color="gray", alpha=0.4, linewidth=5)

        ax.set_title(f"Ep {idx+1}: {ep['outcome']} ({ep['steps']} steps)", fontsize=9)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

    if len(episodes) > 0:
        fig.legend(loc="upper center", ncol=episodes[0]["N"] + episodes[0]["M"] + 1,
                   fontsize=8, bbox_to_anchor=(0.5, 1.0))
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")


def plot_reward_curves(episodes, save_path):
    """Plot reward curves for all episodes."""
    fig, ax = plt.subplots(figsize=(10, 4))
    for idx, ep in enumerate(episodes):
        color = "#4CAF50" if ep["outcome"] == "all_intercepted" else \
                "#F44336" if ep["outcome"] == "target_reached" else "#FFC107"
        ax.plot(ep["rewards"], color=color, alpha=0.6, linewidth=1,
                label=ep["outcome"] if idx < 3 else None)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Reward per Step")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  Saved: {save_path}")


def make_gif(episode, save_path, fps=15):
    """Create animated GIF of a single episode."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    blue_shades = ["#2196F3", "#1565C0", "#0D47A1"]
    red_shades = ["#F44336", "#C62828", "#B71C1C"]

    total_frames = episode["steps"]

    def update(frame):
        ax.clear()
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Set consistent axis limits
        all_pos = np.concatenate(
            [t[:frame+1] for t in episode["def_trajs"] + episode["att_trajs"]]
        )
        margin = 0.5
        for setter, dim in [(ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)]:
            lo = min(all_pos[:, dim].min(), episode["target_pos"][dim]) - margin
            hi = max(all_pos[:, dim].max(), episode["target_pos"][dim]) + margin
            setter(lo, hi)

        # Target
        ax.scatter(*episode["target_pos"], color="#4CAF50", marker="D", s=150)

        # Obstacles (hills)
        if episode["obstacle_pos"] is not None:
            for o in range(len(episode["obstacle_pos"])):
                p = episode["obstacle_pos"][o]
                hill_h = p[2] * 2  # obstacle z = half-height
                ax.plot([p[0], p[0]], [p[1], p[1]], [0, hill_h],
                        color="sienna", alpha=0.5, linewidth=6)

        # Defenders
        for d in range(episode["N"]):
            t = episode["def_trajs"][d][:frame+1]
            c = blue_shades[d % len(blue_shades)]
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=c, linewidth=2)
            ax.scatter(*t[-1], color=c, marker="o", s=80)

        # Attackers
        for a in range(episode["M"]):
            t = episode["att_trajs"][a][:frame+1]
            c = red_shades[a % len(red_shades)]
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=c, linewidth=2, linestyle="--")
            ax.scatter(*t[-1], color=c, marker="*", s=80)

        ax.set_title(f"Step {frame}/{total_frames} | {episode['outcome']}")

    anim = FuncAnimation(fig, update, frames=total_frames, interval=1000//fps)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Swarm Evaluation")
    parser.add_argument("--model", type=str, required=True, help="Path to swarm model")
    parser.add_argument("--defenders", type=int, default=2)
    parser.add_argument("--attackers", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--obstacles", type=int, default=3)
    parser.add_argument("--attacker-model", type=str, default="",
                        help="Path to RL attacker model (default: scripted)")
    parser.add_argument("--gif", action="store_true", help="Generate animated GIF")
    parser.add_argument("--strategy", type=str, default="direct",
                        choices=["direct", "flanking", "mixed"],
                        help="Scripted attacker strategy (ignored if --attacker-model set)")
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--k-opponents", type=int, default=3)
    parser.add_argument("--k-teammates", type=int, default=2)
    parser.add_argument("--ukraine", action="store_true",
                        help="Use Ukraine scenario scale (1km arena, Shahed vs FPV). "
                             "Sets defenders=7, attackers=5, k_opponents=3, k_teammates=3 unless overridden.")
    args = parser.parse_args()

    wp.init()

    from envs.newton_swarm_env import NewtonSwarmEnv
    from train.config import EnvConfig

    model_path = args.model
    try:
        model = PPO.load(model_path)
    except Exception:
        model = SAC.load(model_path)

    # Auto-detect K config from model obs dim
    model_obs_dim = model.observation_space.shape[0]
    k_opp, k_tm = args.k_opponents, args.k_teammates
    from envs.newton_swarm_env import compute_obs_dim
    if compute_obs_dim(k_opp, k_tm) != model_obs_dim:
        # Find K combo that matches obs_dim
        best = None
        for ko in range(1, 10):
            for kt in range(1, 10):
                if compute_obs_dim(ko, kt) == model_obs_dim:
                    if best is None or abs(ko - kt) < abs(best[0] - best[1]):
                        best = (ko, kt)
        if best:
            k_opp, k_tm = best
            print(f"  Auto-detected K: k_opponents={k_opp}, k_teammates={k_tm} (obs_dim={model_obs_dim})")

    if args.ukraine:
        # Ukraine scenario: apply realistic scale then override with args
        cfg = EnvConfig()
        cfg.apply_ukraine_scale()
        cfg.num_defenders = args.defenders if args.defenders != 2 else 7
        cfg.num_attackers = args.attackers if args.attackers != 2 else 5
        cfg.k_opponents = k_opp
        cfg.k_teammates = k_tm if k_tm != 2 else 3
        cfg.kamikaze = True
        cfg.attacker_strategy = args.strategy
        cfg.num_obstacles = args.obstacles if args.obstacles != 3 else 5
        if args.attacker_model:
            cfg.attacker_model_path = args.attacker_model
        print(f"  Ukraine scenario: {cfg.num_defenders}v{cfg.num_attackers}, "
              f"arena={cfg.arena_bounds*2:.0f}m, sensor_range={cfg.sensor_range:.0f}m, "
              f"hills={cfg.num_obstacles}")
    else:
        env_kwargs = dict(
            num_defenders=args.defenders,
            num_attackers=args.attackers,
            num_obstacles=args.obstacles,
            attacker_strategy=args.strategy,
            k_opponents=k_opp,
            k_teammates=k_tm,
            kamikaze=True,
        )
        if args.attacker_model:
            env_kwargs["attacker_model_path"] = args.attacker_model
        cfg = EnvConfig(**env_kwargs)

    env = NewtonSwarmEnv(num_worlds=1, config=cfg)

    os.makedirs(args.output_dir, exist_ok=True)
    att_label = "rl_att" if args.attacker_model else f"scripted_{args.strategy}"
    tag = f"swarm_{cfg.num_defenders}v{cfg.num_attackers}_{att_label}"
    print(f"\nRunning {args.episodes} episodes: {cfg.num_defenders}v{cfg.num_attackers}")

    episodes = []
    for i in range(args.episodes):
        ep = run_swarm_episode(model, env)
        episodes.append(ep)
        print(f"  Ep {i+1}: {ep['outcome']} ({ep['steps']} steps)")

    # Summary
    outcomes = [ep["outcome"] for ep in episodes]
    n_intercept = outcomes.count("all_intercepted")
    n_target = outcomes.count("target_reached")
    n_timeout = outcomes.count("timeout")
    avg_steps = np.mean([ep["steps"] for ep in episodes])

    print(f"\n{'='*60}")
    print(f"  {tag} Evaluation Summary ({args.episodes} episodes)")
    print(f"{'='*60}")
    print(f"  All intercepted: {n_intercept}/{args.episodes} ({100*n_intercept/args.episodes:.0f}%)")
    print(f"  Target reached:  {n_target}/{args.episodes} ({100*n_target/args.episodes:.0f}%)")
    print(f"  Timeout:         {n_timeout}/{args.episodes} ({100*n_timeout/args.episodes:.0f}%)")
    print(f"  Avg steps:       {avg_steps:.0f} ({avg_steps/48:.1f}s)")
    print(f"{'='*60}")

    # Plots
    grid_path = os.path.join(args.output_dir, f"eval_{tag}_grid.png")
    plot_episode_grid(episodes[:6], grid_path, title=f"{tag} — {n_intercept}/{args.episodes} intercepted")

    reward_path = os.path.join(args.output_dir, f"eval_{tag}_rewards.png")
    plot_reward_curves(episodes, reward_path)

    if args.gif and len(episodes) > 0:
        # Pick a good episode for GIF
        best = next((e for e in episodes if e["outcome"] == "all_intercepted"), episodes[0])
        gif_path = os.path.join(args.output_dir, f"eval_{tag}_replay.gif")
        make_gif(best, gif_path)

    env.close()


if __name__ == "__main__":
    main()
