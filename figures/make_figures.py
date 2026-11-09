"""Regenerate the figures of paper.tex (vector PDF, plus a PNG preview of each) into this directory.

    python figures/make_figures.py          # from the repository root

Needs matplotlib, numpy, pandas and Pillow. Every input is tracked in git:
    models/eval_output/nvm_grid_*.csv    fixed-seed grid evaluations (eval/eval_nvm_grid.py)
    docs/media/*.gif                     top-down replays (eval/render_aegis.py)
    models/eval_output/eval_swarm_7v5_scripted_direct_grid.png
                                         a pre-AEGIS swarm evaluation (eval/eval_swarm.py)
Figure 2 (timeline) and Figure 3f (adversarial gates) use numbers that exist only in the dated
experiment records; each is listed next to the file it comes from.

Outputs: replays.pdf and pipeline.pdf (Fig. 1), timeline.pdf (Fig. 2), results.pdf (Fig. 3),
crazyflie.pdf (Fig. 4). The script also prints every plotted number.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402
from PIL import Image, ImageSequence  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVAL = os.path.join(ROOT, "models", "eval_output")
MEDIA = os.path.join(ROOT, "docs", "media")

# Colour-blind-safe categorical slots (validated all-pairs: CVD dE >= 9.2, normal-vision dE >= 24)
# plus neutral inks; the grey reference series was validated against ORANGE and BLUE (CVD dE 9.8).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FILL_DATA, FILL_MODEL, EDGE_MODEL = "#f0efec", "#dbe9fb", "#86b6ef"
FILL_HW, EDGE_HW = "#e3f5ee", "#7fcfae"
SHADE = "#f3f2ee"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom", "mathtext.rm": "sans", "mathtext.it": "sans:italic", "mathtext.bf": "sans:bold",
    "font.size": 7.0, "axes.labelsize": 7.0, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.titlesize": 7.2, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.titlepad": 3.0,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.2, "ytick.major.size": 2.2, "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
})
COL_W, TEXT_W = 3.55, 7.30  # inches: one column and the full text width of paper.tex


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".pdf"), metadata={"CreationDate": None})
    fig.savefig(os.path.join(HERE, name + ".png"), dpi=200)
    plt.close(fig)


def grid(name):
    return pd.read_csv(os.path.join(EVAL, f"nvm_grid_{name}.csv"))


def style(ax, ygrid=True):
    if ygrid:
        ax.yaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(pad=1.5)


# ----------------------------------------------------------------------------- Fig. 1 (top): replays
def gif_frame(name, index):
    with Image.open(os.path.join(MEDIA, name)) as im:
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            if i == index:
                return np.asarray(frame.convert("RGB")).copy()
    raise SystemExit(f"{name}: no frame {index}")


def arena_crop(img, half=450.0):
    """Crop a 700 px replay frame to the square |x|, |y| <= half (metres) and blank its legend.

    The renderer draws the +-500 m arena between pixels 89 and 628 (both axes); its legend sits in
    the top-right corner of the axes and is unreadable at print size, so the caption replaces it."""
    img = img.copy()
    face = img[300, 100].copy()                       # axes face colour
    img[84:146, 515:628] = face                       # legend box
    px = 539.0 / 1000.0
    lo, hi = int(round(89 + (500 - half) * px)), int(round(89 + (500 + half) * px))
    return img[lo:hi, lo:hi]


def fig_replays(path):
    swarm = np.asarray(Image.open(os.path.join(EVAL, "eval_swarm_7v5_scripted_direct_grid.png")).convert("RGB"))
    before = swarm[925:1530, 1665:2415]               # episode 6 of the pre-AEGIS 7v5 evaluation grid
    panels = [
        (before, "(a) before AEGIS: 7v5"),
        (arena_crop(gif_frame("aegis_big24_30v30_hills.gif", 0)), "(b) 30v30, hills: $t$ = 0 s"),
        (arena_crop(gif_frame("aegis_big24_30v30_hills.gif", 25)), "(c) $t$ = 2.1 s"),
        (arena_crop(gif_frame("aegis_big24_30v30_hills.gif", 50)), "(d) $t$ = 4.2 s"),
        (arena_crop(gif_frame("aegis_vark_15v15_3targets.gif", 31)), "(e) 15v15, three points"),
        (arena_crop(gif_frame("aegis_large_50v40_open.gif", 19)), "(f) 50v40, open field"),
    ]
    gap, lab = 0.05, 0.15
    aspects = [p[0].shape[1] / p[0].shape[0] for p in panels]   # width / height of each image
    side_in = (TEXT_W - gap * (len(panels) - 1)) / sum(aspects)  # common panel height
    H = side_in + lab
    fig = plt.figure(figsize=(TEXT_W, H))
    x0 = 0.0
    for k, ((img, text), asp) in enumerate(zip(panels, aspects)):
        w_in = side_in * asp
        ax = fig.add_axes([x0 / TEXT_W, lab / H, w_in / TEXT_W, side_in / H])
        ax.imshow(img, interpolation="lanczos", aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(k == 0)
            s.set_color(AXIS)
            s.set_linewidth(0.6)
        fig.text((x0 + 0.01) / TEXT_W, 0.035 / H, text, fontsize=6.9, color=INK, ha="left", va="bottom")
        if k == 1:                                     # 200 m scale bar (crop spans 900 m)
            n = img.shape[0]
            bar = 200.0 / 900.0 * n
            ax.plot([0.06 * n, 0.06 * n + bar], [0.94 * n, 0.94 * n], color="white", lw=1.2,
                    solid_capstyle="butt")
            ax.text(0.06 * n + bar / 2, 0.91 * n, "200 m", color="white", fontsize=6.0, ha="center", va="bottom")
        x0 += w_in + gap
    save(fig, path)


# ----------------------------------------------------------------------------- Fig. 1 (bottom): pipeline
def _box(ax, x, y, w, h, text, fill=FILL_DATA, edge=AXIS, size=6.6, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.04", fc=fill, ec=edge,
                                lw=0.6, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.12)


def _arrow(ax, x0, y0, x1, y1, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=6.5, lw=0.8, color=INK2,
                                 shrinkA=0, shrinkB=0, ls=ls))


def fig_pipeline(path):
    H = 0.60
    fig = plt.figure(figsize=(TEXT_W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, TEXT_W)
    ax.set_ylim(0, H)
    ax.axis("off")
    y, h, gap = 0.18, 0.41, 0.13
    boxes = [
        (1.08, "512 parallel GPU worlds\n(Newton/Warp, 240 Hz)\n$N$ v $M$, terrain, wind", FILL_DATA, AXIS),
        (1.10, "allocator (classical):\nintercept-time costs,\nHungarian every 0.5 s", FILL_DATA, AXIS),
        (1.14, "56-d ego-centric obs\nper defender, the same\nsize for any $N$, $M$", FILL_DATA, AXIS),
        (1.06, "one shared PPO\npolicy, MLP 2$\\times$256,\nrun by every defender", FILL_MODEL, EDGE_MODEL),
        (0.98, "velocity command\n$\\leq$ 60 m/s per axis,\n12 Hz decisions", FILL_DATA, AXIS),
    ]
    x = 0.0
    centres = []
    for i, (w, txt, fc, ec) in enumerate(boxes):
        _box(ax, x, y, w, h, txt, fc, ec)
        centres.append(x + w / 2)
        if i < len(boxes) - 1:
            _arrow(ax, x + w + 0.012, y + h / 2, x + w + gap - 0.012, y + h / 2)
        x += w + gap
    _arrow(ax, x - gap + 0.012, y + h / 2, x - 0.012, y + h / 2, ls=(0, (2, 1.5)))
    _box(ax, x, y, TEXT_W - x - 0.005, h, "or a CrazyFlie 2.1:\nmocap in, cflib out\n(not yet flown)",
         FILL_HW, EDGE_HW, ls=(0, (3, 1.5)))
    # feedback loop under the row: command -> PD force tracking -> physics -> next state
    yl = 0.065
    xs, xe = centres[4], centres[0]
    ax.plot([xs, xs, xe], [y - 0.01, yl, yl], color=INK2, lw=0.8, solid_capstyle="butt")
    _arrow(ax, xe, yl, xe, y - 0.012)
    ax.text((xs + xe) / 2, yl + 0.012, "PD force tracking, physics step, per-defender owner-based reward",
            ha="center", va="bottom", fontsize=6.3, color=INK2)
    save(fig, path)


# ----------------------------------------------------------------------------- Fig. 2: timeline
# (date label, event, phase, largest engagement won as (drones, label) or None)
# Sources: git log (commit dates), README "Project history", docs/AEGIS_PLAN.md sections 1, 4.3 and 5.
TIMELINE = [
    ("22 Mar", "SAC defender learns 1v1 interception", "pre", (2, "1v1")),
    ("24-30 Mar", "GPU physics (Newton/Warp); shared-policy swarms", "pre", None),
    ("Apr-May", "swarm training stalls: 7 defenders, 3% intercepts", "fail", (12, "7v5")),
    ("12-23 Jul", "AEGIS design; all 4 curriculum gates in 28 min", "aegis", (25, "15v10")),
    ("19 Aug", "reusable interceptors; 40v30 without training", "aegis", (70, "40v30")),
    ("20-21 Aug", "adversarial co-training loop converges", "aegis", None),
    ("27 Aug", "urban and maze terrain, coordinated raids", "aegis", (90, "50v40")),
    ("28 Aug", "CrazyFlie lab policy and mocap bridge", "hw", None),
    ("29-30 Aug", "24v24 training; one policy for 1-4 points", "aegis", (90, "50v40")),
    ("Sep-Nov", "cflib link, safety layer; number audit; release", "hw", None),
]
PHASE = {"pre": (MUTED, "o", "end-to-end RL"), "fail": (ORANGE, "X", "failed"),
         "aegis": (BLUE, "o", "AEGIS, simulation"), "hw": (AQUA, "D", "hardware path")}


def fig_timeline(path):
    n = len(TIMELINE)
    row = 0.120
    top, bottom = 0.18, 0.15
    H = top + n * row + bottom
    fig = plt.figure(figsize=(COL_W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, COL_W)
    ax.set_ylim(0, H)
    ax.axis("off")
    x_date, x_spine, x_text = 0.50, 0.58, 0.67
    x_bar, bar_max, drones_max = 2.88, 0.36, 90.0
    ys = [H - top - (i + 0.5) * row for i in range(n)]
    ax.plot([x_spine, x_spine], [ys[-1], ys[0]], color=AXIS, lw=0.8, zorder=1)
    ax.text(x_text, H - 0.065, "milestone", fontsize=6.5, color=INK2, va="center", style="italic")
    ax.text(x_bar, H - 0.065, "largest win", fontsize=6.5, color=INK2, va="center", style="italic")
    for (date, text, phase, best), yv in zip(TIMELINE, ys):
        col, mk, _ = PHASE[phase]
        ax.text(x_date, yv, date, ha="right", va="center", fontsize=6.4, color=INK2)
        ax.scatter([x_spine], [yv], s=16 if mk != "X" else 22, marker=mk, color=col, edgecolor="white",
                   linewidth=0.6, zorder=3)
        ax.text(x_text, yv, text, ha="left", va="center", fontsize=6.6, color=INK)
        if best:
            d, lab = best
            w = max(bar_max * d / drones_max, 0.012)
            fail = phase == "fail"
            ax.add_patch(Rectangle((x_bar, yv - 0.045), w, 0.09, fc="white" if fail else BLUE,
                                   ec=ORANGE if fail else BLUE, lw=0.8 if fail else 0, hatch="//////" if fail else None))
            ax.text(x_bar + w + 0.03, yv, lab + (" (lost)" if fail else ""), ha="left", va="center", fontsize=6.2,
                    color=INK2)
    # legend
    xl = 0.08
    for key in ["pre", "fail", "aegis", "hw"]:
        col, mk, lab = PHASE[key]
        ax.scatter([xl], [0.065], s=14 if mk != "X" else 20, marker=mk, color=col, edgecolor="white", linewidth=0.5)
        ax.text(xl + 0.06, 0.065, lab, fontsize=6.3, color=INK2, va="center")
        xl += 0.13 + 0.047 * len(lab)
    plt.setp(ax.texts, clip_on=False)
    save(fig, path)


# ----------------------------------------------------------------------------- Fig. 3: results
def realism_summary():
    """Scripted baseline vs the July learned policy (kamikaze) vs the reusable-interceptor policy,
    realism grid: 38 N-v-M cells x 3 raid types, 32 episodes each."""
    out = {}
    for key, name in [("scripted", "baseline_realism"), ("kamikaze", "model_realism"),
                      ("reusable", "model_reuse_realism")]:
        d = grid(name)
        out[key] = {"win, all cells": 100 * d.win_rate.mean(),
                    "win, N = M": 100 * d[d.n == d.m].win_rate.mean(),
                    "win, N < M": 100 * d[d.n < d.m].win_rate.mean(),
                    "defenders lost": 100 * d.bad_death_rate.mean(),
                    "xr_outnumbered": d[d.n < d.m].exchange_ratio.mean()}
    b, m = grid("baseline_realism"), grid("model_realism")
    j = m.merge(b, on=["n", "m", "strategy"], suffixes=("_m", "_b"))
    better = int((j.breach_rate_m < j.breach_rate_b - 1e-12).sum())
    worse = int((j.breach_rate_m > j.breach_rate_b + 1e-12).sum())
    print(f"[3a] breach: learned better {better}, ties {len(j) - better - worse}, worse {worse} of {len(j)}")
    for k, v in out.items():
        print(f"[3a] {k:9s} " + "  ".join(f"{m_}={x:.1f}" for m_, x in v.items()))
    return out


def panel_learning(ax):
    s = realism_summary()
    metrics = ["win, all cells", "win, N = M", "win, N < M", "defenders lost"]
    series = [("scripted", "scripted", MUTED), ("kamikaze", "kamikaze", ORANGE), ("reusable", "reusable", BLUE)]
    x = np.arange(len(metrics))
    bw = 0.26
    for i, (key, lab, col) in enumerate(series):
        ax.bar(x + (i - 1) * bw, [s[key][m] for m in metrics], width=bw * 0.92, color=col, label=lab, lw=0, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["win\n(all)", "win\n($N$=$M$)", "win\n($N$<$M$)", "defenders\nlost"], fontsize=6.4)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.5, 3.5)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("%", labelpad=1.5)
    ax.legend(loc="upper right", frameon=False, handlelength=0.8, borderaxespad=0.0, labelspacing=0.15,
              handletextpad=0.35, bbox_to_anchor=(1.0, 1.0))
    ax.set_title("(a) scripted vs learned flight")
    style(ax)


def panel_scale(ax):
    single, scale = grid("sc_single"), grid("sc_scale")
    rows = []
    for d, setting in [(single, "realism"), (scale, "open")]:
        for (n, m), g in d.groupby(["n", "m"]):
            rows.append((int(n), int(m), 100 * g.win_rate.mean(), 100 * g.bad_death_rate.mean(), setting))
    order = ["1v1", "3v3", "5v5", "7v5", "10v10", "15v15", "20v20", "10v20", "20v40", "30v30", "40v40", "50v40"]
    by = {f"{n}v{m}": (w, b, s) for n, m, w, b, s in rows}
    x = np.arange(len(order))
    win = [by[c][0] for c in order]
    for c in order:
        print(f"[3b] {c:6s} win {by[c][0]:.1f}%  defenders lost {by[c][1]:.1f}%  ({by[c][2]})")
    ax.axvspan(7.5, 11.6, color=SHADE, lw=0, zorder=0)
    ax.axvline(6.5, color=AXIS, lw=0.6, zorder=1)
    ax.text(3.0, 80, "hills, wind, noise", ha="center", va="center", fontsize=6.2, color=INK2)
    ax.text(9.05, 80, "open field", ha="center", va="center", fontsize=6.2, color=INK2)
    ax.text(9.55, 62, "beyond the\ntraining range", ha="center", va="center", fontsize=6.2, color=INK2)
    ax.plot(x, win, color=BLUE, lw=1.0, marker="o", ms=3.2, mec="white", mew=0.5, zorder=3,
            label="reusable (aegis_split)")
    # the kamikaze policy at 10v20 (overnight_20260819 q1: 0% win, exactly 10 kills then overrun)
    ax.plot([7], [0], ls="none", marker="X", ms=4.5, color=ORANGE, mec="white", mew=0.5, zorder=4,
            label="kamikaze (July)")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, fontsize=6.2)
    ax.set_ylim(-6, 106)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.6, 11.6)
    ax.set_ylabel("win rate (%)", labelpad=1.5)
    ax.legend(loc="lower left", frameon=False, handlelength=1.2, borderaxespad=0.1, labelspacing=0.2,
              bbox_to_anchor=(0.0, 0.1))
    ax.set_title("(b) one policy from 1v1 to 50v40")
    style(ax)


def panel_targets(ax):
    ks = [1, 2, 3, 4]
    rows = []
    for k in ks:
        d = grid(f"vark_k{k}")
        rows.append((100 * d[d.strategy == "direct"].win_rate.mean(), 100 * d[d.strategy == "saturation"].win_rate.mean(),
                     100 * d.win_rate.mean(), 100 * d.bad_death_rate.mean()))
        print(f"[3c] K={k} direct {rows[-1][0]:.1f} saturation {rows[-1][1]:.1f} overall {rows[-1][2]:.1f} "
              f"defenders lost {rows[-1][3]:.1f}%")
    ax.plot(ks, [r[1] for r in rows], color=BLUE, lw=1.0, marker="o", ms=3.4, mec="white", mew=0.5,
            label="saturation raids", zorder=3)
    ax.plot(ks, [r[0] for r in rows], color=ORANGE, lw=1.0, marker="s", ms=3.2, mec="white", mew=0.5,
            label="direct raids", zorder=3)
    ax.set_xticks(ks)
    ax.set_xlim(0.7, 4.3)
    ax.set_ylim(85, 101.5)
    ax.set_yticks([85, 90, 95, 100])
    ax.set_xlabel("protected points $K$ (one policy)", labelpad=1.5)
    ax.set_ylabel("win rate (%)", labelpad=1.5)
    ax.legend(loc="lower left", frameon=False, handlelength=1.4, borderaxespad=0.1, labelspacing=0.2)
    ax.set_title("(c) defending 1-4 points (aegis_vark)")
    style(ax)


def panel_direct(ax):
    # K = 3, direct simultaneous raids, 10v10 / 15v15 / 20v20 (min, mean, max over the three cells)
    steps = [("split (29 Aug)", "sc_split3"),
             ("+ direct training", "splitdirect_k3direct"),
             ("+ spread spawns", "splitspawn_k3b"),
             ("aegis_vark", "vark_k3")]
    vals = []
    for lab, name in steps:
        d = grid(name)
        w = 100 * d[d.strategy == "direct"].win_rate
        vals.append((w.min(), w.mean(), w.max()))
        print(f"[3d] {name:22s} direct win {w.min():.1f}-{w.max():.1f} (mean {w.mean():.1f}), n={d.episodes.iloc[0]}")
    y = np.arange(len(steps))[::-1]
    for yi, (lo, mu, hi), i in zip(y, vals, range(len(steps))):
        col = BLUE if i >= 2 else MUTED
        ax.plot([lo, hi], [yi, yi], color=col, lw=1.6, solid_capstyle="round", zorder=3)
        ax.plot([mu], [yi], marker="o", ms=4.0, color=col, mec="white", mew=0.6, zorder=4)
        ax.text(hi + 1.0, yi, f"{mu:.0f}%", va="center", ha="left", fontsize=6.3, color=INK2)
    ax.set_yticks(y)
    ax.set_yticklabels([s[0] for s in steps], fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(70, 100)
    ax.set_xticks([70, 80, 90, 100])
    ax.set_ylim(-0.6, len(steps) - 0.4)
    ax.xaxis.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("win rate on direct raids, 3 points (%)", labelpad=1.5)
    ax.set_title("(d) closing the hardest case")


def panel_sensing(ax):
    ranges = [300, 150, 75]
    series = [("1 point", ["vark_base_k1", "vark_sensor150_k1", "vark_sensor75_k1"], MUTED, "o"),
              ("3 points", ["vark_base_k3", "vark_sensor150_k3", "vark_sensor75_k3"], BLUE, "o"),
              ("3 points, range-\nrandomized training",
               ["sensrobust2_base_k3", "sensrobust2_s150_k3", "sensrobust2_s75_k3"], ORANGE, "s")]
    x = np.arange(len(ranges))
    for lab, names, col, mk in series:
        w = [100 * grid(nm).win_rate.mean() for nm in names]
        print(f"[3e] {lab.replace(chr(10), ' '):48s} " + "  ".join(f"{r} m: {v:.1f}" for r, v in zip(ranges, w)))
        ax.plot(x, w, color=col, lw=1.0, marker=mk, ms=3.3, mec="white", mew=0.5, label=lab, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r} m" for r in ranges])
    ax.set_xlim(-0.25, 2.25)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("detection range (points 230-300 m apart)", labelpad=1.5)
    ax.set_ylabel("win rate (%)", labelpad=1.5)
    ax.legend(loc="lower left", frameon=False, handlelength=1.3, borderaxespad=0.1, labelspacing=0.25)
    ax.set_title("(e) sensing must span the points")
    style(ax)


# Round-A gates: true breach rate a freshly trained RL attacker (40M steps) reaches against each frozen
# defender, and the Round-B breach after co-training where one was run. Sources:
# experiments/adv_pop_20260821/results.md (rounds 1-3 and the reusable defender, reuse.log),
# overnight_20260827 (terrain generalist), cfadv_20260828, convoyadv_20260829, varkadv_20260831.
ADV = [  # (label, [(round-A breach, gap?) per round], breach after co-training or None)
    ("kamikaze, 3 rounds", [(0.75, True), (0.46, True), (0.094, False)], None),
    ("reusable", [(0.301, True)], 0.017),
    ("terrain generalist", [(0.012, False)], None),
    ("CrazyFlie lab", [(0.19, True)], 0.020),
    ("convoy escort", [(0.077, False)], None),
    ("1-4 points", [(0.005, False)], None),
]


def panel_adversarial(ax):
    y = np.arange(len(ADV))[::-1]
    arrow = dict(arrowstyle="-|>", color=INK2, lw=0.7, mutation_scale=6, shrinkA=3, shrinkB=3)
    for yi, (lab, pts, b) in zip(y, ADV):
        for (v0, _), (v1, _) in zip(pts[:-1], pts[1:]):
            ax.annotate("", xy=(v1, yi), xytext=(v0, yi), arrowprops=arrow)
        if b is not None:
            ax.annotate("", xy=(b, yi), xytext=(pts[-1][0], yi), arrowprops=arrow)
            ax.plot([b], [yi], marker="o", ms=3.8, mfc="white", mec=BLUE, mew=1.0, zorder=4)
        for v, gap in pts:
            ax.plot([v], [yi], marker="o", ms=4.0, color=ORANGE if gap else BLUE, mec="white", mew=0.5, zorder=4)
    ax.axvline(0.08, color=INK2, lw=0.7, zorder=2)
    ax.text(0.092, 0.0, "gap\nthreshold", fontsize=6.0, color=INK2, va="center", ha="left", linespacing=1.0)
    ax.set_xscale("log")
    ax.set_xlim(0.0006, 1.2)
    ax.set_xticks([0.001, 0.01, 0.1, 1.0])
    ax.set_xticklabels(["0.001", "0.01", "0.1", "1"])
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in ADV], fontsize=6.2)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, len(ADV) - 0.4)
    ax.xaxis.grid(True, color=GRID, lw=0.5, which="major")
    ax.set_axisbelow(True)
    ax.set_xlabel("breach rate of a learned attacker", labelpad=1.5)
    h = [plt.Line2D([], [], ls="none", marker="o", ms=4, color=ORANGE, mec="white", label="gap: co-train"),
         plt.Line2D([], [], ls="none", marker="o", ms=4, color=BLUE, mec="white", label="no gap"),
         plt.Line2D([], [], ls="none", marker="o", ms=3.8, mfc="white", mec=BLUE, mew=1.0, label="co-trained")]
    ax.legend(handles=h, loc="upper left", frameon=False, handlelength=0.6, borderaxespad=0.0, labelspacing=0.15,
              handletextpad=0.3, bbox_to_anchor=(-0.01, 1.02), fontsize=6.0)
    ax.set_title("(f) learned attackers")


def fig_results(path):
    H = 2.18
    fig = plt.figure(figsize=(TEXT_W, H))

    def at(x0, w, y0, h):                              # axes from inches: left, width, bottom, height
        return fig.add_axes([x0 / TEXT_W, y0 / H, w / TEXT_W, h / H])

    r1, h1, r2, h2 = 1.36, 0.68, 0.30, 0.64            # row 1 leaves room for (b)'s rotated tick labels
    axs = [at(0.40, 1.88, r1, h1), at(2.86, 1.88, r1, h1), at(5.33, 1.92, r1, h1),
           at(0.91, 1.33, r2, h2), at(2.86, 1.57, r2, h2), at(5.58, 1.67, r2, h2)]
    panel_learning(axs[0])
    panel_scale(axs[1])
    panel_targets(axs[2])
    panel_direct(axs[3])
    panel_sensing(axs[4])
    panel_adversarial(axs[5])
    save(fig, path)


# ----------------------------------------------------------------------------- Fig. 4: CrazyFlie
def fig_crazyflie(path):
    sets = [("large-team policy,\nlab-like", "cf_lab_control"), ("CrazyFlie policy,\nlab-like", "cfadv_scripted"),
            ("CrazyFlie policy,\nhills + wind", "sc_cflab")]
    fig = plt.figure(figsize=(COL_W, 1.14))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("blues", ["#f4f8fd", "#86b6ef", BLUE, "#174a8c"])
    w, gap, x0, b, hgt = 0.245, 0.045, 0.115, 0.19, 0.55
    for k, (title, name) in enumerate(sets):
        d = grid(name)
        g = d.groupby(["n", "m"]).win_rate.mean()
        M = np.full((3, 3), np.nan)
        for (n, m), v in g.items():
            M[int(n) - 1, int(m) - 1] = 100 * v
        print(f"[4] {name:16s} " + "  ".join(f"{int(n)}v{int(m)}={100 * v:.1f}" for (n, m), v in g.items()))
        ax = fig.add_axes([x0 + k * (w + gap), b, w, hgt])
        ax.imshow(np.ma.masked_invalid(M), cmap=cmap, vmin=40, vmax=100, origin="upper", aspect="auto")
        for i in range(3):
            for j in range(3):
                if np.isnan(M[i, j]):
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="white", ec=GRID, lw=0.5, hatch="////"))
                    continue
                v = M[i, j]
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.4,
                        color="white" if v >= 80 else INK)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["1", "2", "3"])
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["1", "2", "3"] if k == 0 else [])
        ax.tick_params(length=0, pad=1.5)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xlabel("attackers $M$", labelpad=1.0)
        if k == 0:
            ax.set_ylabel("defenders $N$", labelpad=1.0)
        ax.set_title(title, fontsize=6.6, fontweight="normal", loc="center", pad=2.5, linespacing=1.05)
    save(fig, path)


def main():
    fig_replays("replays")
    fig_pipeline("pipeline")
    fig_timeline("timeline")
    fig_results("results")
    fig_crazyflie("crazyflie")
    print("[done] wrote replays, pipeline, timeline, results, crazyflie (.pdf + .png) to", HERE)


if __name__ == "__main__":
    main()
