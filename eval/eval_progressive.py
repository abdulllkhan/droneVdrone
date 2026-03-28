"""
Comprehensive evaluation and visualization for progressive training.

Generates 3D trajectory plots, distance/speed timeseries, animated GIFs,
and summary statistics for any mode (fly_to_target, chase, intercept).

Usage:
    python -m eval.eval_progressive --model models/best_model.zip --mode fly_to_target
    python -m eval.eval_progressive --model models/phase2_chase_speed0.5_final --mode chase --speed 0.5
    python -m eval.eval_progressive --model models/phase3_defender_final --mode intercept --attacker models/phase2_chase_speed0.5_final
"""

import argparse
import os

import numpy as np
import warp as wp
from stable_baselines3 import SAC

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter


def run_episodes(model, env, num_episodes, max_steps=480):
    """Run episodes and record trajectories."""
    obs = env.reset()
    n_envs = env.n_envs
    batch_size = min(n_envs, num_episodes)

    all_episodes = []
    completed = 0

    while completed < num_episodes:
        # Record trajectories for this batch
        trajectories = [{"pos": [], "vel": [], "target": [], "target_vel": []}
                        for _ in range(batch_size)]
        done_mask = np.zeros(batch_size, dtype=bool)
        outcomes = ["timeout"] * batch_size

        for step in range(max_steps):
            for i in range(batch_size):
                if not done_mask[i]:
                    trajectories[i]["pos"].append(obs[i, :3].copy())
                    trajectories[i]["vel"].append(obs[i, 3:6].copy())
                    trajectories[i]["target"].append(obs[i, 6:9].copy())
                    if obs.shape[1] > 9:
                        trajectories[i]["target_vel"].append(obs[i, 9:12].copy())

            actions, _ = model.predict(obs[:batch_size], deterministic=True)
            # Pad actions if env has more envs than batch
            if n_envs > batch_size:
                full_actions = np.zeros((n_envs, actions.shape[1]), dtype=np.float32)
                full_actions[:batch_size] = actions
                obs, rewards, dones, infos = env.step(full_actions)
            else:
                obs, rewards, dones, infos = env.step(actions)

            for i in range(batch_size):
                if dones[i] and not done_mask[i]:
                    done_mask[i] = True
                    if infos[i].get("arrived"):
                        outcomes[i] = "arrived"
                    elif infos[i].get("intercepted"):
                        outcomes[i] = "intercepted"
                    elif infos[i].get("target_reached"):
                        outcomes[i] = "target_reached"

            if all(done_mask):
                break

        for i in range(batch_size):
            ep = {
                "pos": np.array(trajectories[i]["pos"]),
                "vel": np.array(trajectories[i]["vel"]),
                "target": np.array(trajectories[i]["target"]),
                "target_vel": np.array(trajectories[i]["target_vel"]) if trajectories[i]["target_vel"] else None,
                "outcome": outcomes[i],
                "steps": len(trajectories[i]["pos"]),
            }
            all_episodes.append(ep)
            completed += 1
            if completed >= num_episodes:
                break

        # Reset for next batch
        if completed < num_episodes:
            obs = env.reset()

    return all_episodes


def plot_3d_trajectories(episodes, title, output_path, mode="fly_to_target"):
    """Plot 3D trajectories with start/end/target markers."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(episodes), 10)))

    for i, ep in enumerate(episodes[:10]):
        c = colors[i % 10]
        pos = ep["pos"]
        tgt = ep["target"]

        # Drone trajectory
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=c, alpha=0.8, linewidth=1.5,
                label=f"Ep {i+1}: {ep['outcome']} ({ep['steps']} steps)")
        ax.scatter(*pos[0], color=c, marker="o", s=50, zorder=5)  # start
        ax.scatter(*pos[-1], color=c, marker="x", s=80, zorder=5)  # end

        # Target (moving or static)
        if mode == "chase" and len(tgt) > 1:
            ax.plot(tgt[:, 0], tgt[:, 1], tgt[:, 2], color=c, alpha=0.3,
                    linewidth=1, linestyle="--")
        ax.scatter(*tgt[-1], color=c, marker="*", s=150, edgecolors="black",
                   zorder=5)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_timeseries(episodes, title, output_path, mode="fly_to_target"):
    """Plot distance + speed over time for each episode."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 1, min(len(episodes), 10)))

    for i, ep in enumerate(episodes[:10]):
        c = colors[i % 10]
        pos = ep["pos"]
        tgt = ep["target"]
        vel = ep["vel"]
        t = np.arange(len(pos)) / 48.0  # seconds

        # Distance to target
        dists = np.linalg.norm(pos - tgt[:len(pos)], axis=1)
        axes[0].plot(t, dists, color=c, alpha=0.7, linewidth=1.5,
                     label=f"Ep {i+1}: {ep['outcome']}")

        # Speed
        speeds = np.linalg.norm(vel, axis=1)
        axes[1].plot(t, speeds, color=c, alpha=0.7, linewidth=1.5)

    axes[0].axhline(y=0.3, color="green", linestyle="--", alpha=0.5, label="Arrival threshold")
    axes[0].set_ylabel("Distance to target (m)")
    axes[0].set_title(title)
    axes[0].legend(fontsize=7)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved: {output_path}")


def plot_animated_gif(episodes, output_path, mode="fly_to_target", fps=10):
    """Create animated GIF of trajectories."""
    ep = episodes[0]  # animate first episode
    pos = ep["pos"]
    tgt = ep["target"]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Set fixed limits
    all_pts = np.vstack([pos, tgt[:len(pos)]])
    margin = 0.5
    ax.set_xlim(all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ax.set_ylim(all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    ax.set_zlim(max(0, all_pts[:, 2].min() - margin), all_pts[:, 2].max() + margin)

    drone_line, = ax.plot([], [], [], "b-", alpha=0.5, linewidth=1)
    drone_dot, = ax.plot([], [], [], "bo", markersize=8)
    target_dot, = ax.plot([], [], [], "r*", markersize=15)
    title = ax.set_title("")

    # Subsample for GIF speed
    step_size = max(1, len(pos) // (fps * 10))  # ~10s of animation

    def update(frame):
        idx = frame * step_size
        if idx >= len(pos):
            idx = len(pos) - 1

        drone_line.set_data(pos[:idx+1, 0], pos[:idx+1, 1])
        drone_line.set_3d_properties(pos[:idx+1, 2])
        drone_dot.set_data([pos[idx, 0]], [pos[idx, 1]])
        drone_dot.set_3d_properties([pos[idx, 2]])

        tgt_idx = min(idx, len(tgt) - 1)
        target_dot.set_data([tgt[tgt_idx, 0]], [tgt[tgt_idx, 1]])
        target_dot.set_3d_properties([tgt[tgt_idx, 2]])

        dist = np.linalg.norm(pos[idx] - tgt[tgt_idx])
        title.set_text(f"Step {idx}/{len(pos)-1} | t={idx/48:.1f}s | dist={dist:.2f}m | {ep['outcome']}")
        return drone_line, drone_dot, target_dot, title

    n_frames = len(pos) // step_size + 1
    anim = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close()
    print(f"  Saved: {output_path}")


def print_summary(episodes, mode):
    """Print evaluation summary statistics."""
    n = len(episodes)
    outcomes = [ep["outcome"] for ep in episodes]
    steps = [ep["steps"] for ep in episodes]

    # Final distances
    final_dists = []
    min_dists = []
    for ep in episodes:
        pos = ep["pos"]
        tgt = ep["target"]
        dists = np.linalg.norm(pos - tgt[:len(pos)], axis=1)
        final_dists.append(dists[-1])
        min_dists.append(dists.min())

    print(f"\n{'='*60}")
    print(f"  Evaluation Summary ({n} episodes)")
    print(f"{'='*60}")

    if mode in ("fly_to_target", "chase"):
        arrived = outcomes.count("arrived")
        print(f"  Arrival rate:      {arrived}/{n} ({arrived/n*100:.0f}%)")
    elif mode == "intercept":
        intercepted = outcomes.count("intercepted")
        target_reached = outcomes.count("target_reached")
        print(f"  Interception rate: {intercepted}/{n} ({intercepted/n*100:.0f}%)")
        print(f"  Target reached:    {target_reached}/{n} ({target_reached/n*100:.0f}%)")

    timeout = outcomes.count("timeout")
    print(f"  Timeout:           {timeout}/{n} ({timeout/n*100:.0f}%)")
    print(f"  Avg steps:         {np.mean(steps):.0f} ({np.mean(steps)/48:.1f}s)")
    print(f"  Avg final dist:    {np.mean(final_dists):.3f}m")
    print(f"  Avg min dist:      {np.mean(min_dists):.3f}m")
    print(f"  Min dist ever:     {np.min(min_dists):.3f}m")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate progressive training models")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--mode", type=str, default="fly_to_target",
                        choices=["fly_to_target", "chase", "intercept"])
    parser.add_argument("--speed", type=float, default=0.5, help="Target speed (chase mode)")
    parser.add_argument("--attacker", type=str, default="", help="Attacker model (intercept mode)")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--prefix", type=str, default="eval")
    parser.add_argument("--gif", action="store_true", help="Generate animated GIF")
    parser.add_argument("--vec-normalize", type=str, default=None,
                        help="Path to VecNormalize stats (.pkl). Use 'none' to disable.")
    args = parser.parse_args()

    wp.init()

    from stable_baselines3.common.vec_env import VecNormalize
    from envs.newton_env import NewtonDroneEnv
    from train.config import EnvConfig

    # Build env config
    env_kwargs = {"mode": args.mode}
    if args.mode == "chase":
        env_kwargs["moving_target_speed"] = args.speed
    if args.mode == "intercept" and args.attacker:
        env_kwargs["attacker_model_path"] = args.attacker

    cfg = EnvConfig(**env_kwargs)
    env = NewtonDroneEnv(num_envs=args.num_envs, config=cfg)

    # Wrap with VecNormalize only if explicitly provided
    vec_norm_path = args.vec_normalize
    if vec_norm_path and vec_norm_path.lower() != "none" and os.path.exists(vec_norm_path):
        print(f"Loading VecNormalize from: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
        env.training = False
        env.norm_reward = False

    model = SAC.load(args.model)

    print(f"\nRunning {args.episodes} episodes: mode={args.mode}")
    episodes = run_episodes(model, env, args.episodes)
    env.close()

    # Summary
    print_summary(episodes, args.mode)

    # Plots
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = args.prefix

    plot_3d_trajectories(
        episodes,
        f"{prefix} — 3D Trajectories ({args.mode})",
        os.path.join(args.output_dir, f"{prefix}_3d.png"),
        mode=args.mode,
    )

    plot_timeseries(
        episodes,
        f"{prefix} — Distance & Speed ({args.mode})",
        os.path.join(args.output_dir, f"{prefix}_timeseries.png"),
        mode=args.mode,
    )

    if args.gif:
        plot_animated_gif(
            episodes,
            os.path.join(args.output_dir, f"{prefix}_replay.gif"),
            mode=args.mode,
        )


if __name__ == "__main__":
    main()
