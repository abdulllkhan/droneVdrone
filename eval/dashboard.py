"""AEGIS results dashboard generator.

Scans models/eval_output/nvm_grid_*.csv, experiments/*/results.md, GIFs and git
history, and writes a self-contained HTML dashboard to
models/eval_output/dashboard.html (open it in any browser). Re-run any time:

    python -m eval.dashboard
    python -m eval.dashboard --inline-gifs --out /tmp/dash.html   # portable copy
"""

import argparse
import base64
import csv
import glob
import html
import os
import subprocess
import time

OUT_DEFAULT = "models/eval_output/dashboard.html"

CSS = """
:root { color-scheme: light;
  --s1:#fcfcfb; --s2:#f1f0ee; --ink:#0b0b0b; --ink2:#52514e; --line:#d8d6d2;
  --c1:#2a78d6; --c2:#eb6834; --c3:#1baf7a; --c4:#eda100; --good:#008300; --bad:#c92a2a; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  color-scheme: dark;
  --s1:#1a1a19; --s2:#242423; --ink:#ffffff; --ink2:#c3c2b7; --line:#3a3936;
  --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500; --good:#31c48d; --bad:#f27979; } }
:root[data-theme="dark"] {
  color-scheme: dark;
  --s1:#1a1a19; --s2:#242423; --ink:#ffffff; --ink2:#c3c2b7; --line:#3a3936;
  --c1:#3987e5; --c2:#d95926; --c3:#199e70; --c4:#c98500; --good:#31c48d; --bad:#f27979; }
* { box-sizing: border-box; }
body { margin:0; background:var(--s1); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; padding:24px; }
h1 { font-size:26px; margin:0 0 2px; } h2 { font-size:18px; margin:32px 0 10px; }
.sub { color:var(--ink2); font-size:13px; margin-bottom:22px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; }
.tile { background:var(--s2); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }
.tile .v { font-size:26px; font-weight:700; } .tile .k { color:var(--ink2); font-size:12.5px; }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th,td { text-align:left; padding:5px 10px; border-bottom:1px solid var(--line); }
th { color:var(--ink2); font-weight:600; } td.num { font-variant-numeric:tabular-nums; }
.bar { height:14px; border-radius:0 4px 4px 0; background:var(--c1); min-width:2px; }
.barrow { display:grid; grid-template-columns:220px 1fr 64px; gap:10px; align-items:center;
  padding:3px 0; } .barrow:hover { background:var(--s2); }
.barlab { font-size:13px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.barval { font-size:13px; color:var(--ink2); font-variant-numeric:tabular-nums; }
details { border:1px solid var(--line); border-radius:10px; margin:10px 0; background:var(--s2); }
summary { cursor:pointer; padding:10px 14px; font-weight:600; }
details pre { white-space:pre-wrap; padding:0 16px 14px; color:var(--ink2); font-size:12.5px; }
.gifs { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px; }
.gifs figure { margin:0; background:var(--s2); border:1px solid var(--line);
  border-radius:10px; padding:8px; }
.gifs img { width:100%; border-radius:6px; } .gifs figcaption { font-size:12.5px;
  color:var(--ink2); padding:6px 4px 2px; }
.legend { display:flex; gap:16px; font-size:12.5px; color:var(--ink2); margin:6px 0; }
.sw { display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:5px;
  vertical-align:-1px; }
.commits li { font-size:13px; color:var(--ink2); margin:3px 0; }
.commits b { color:var(--ink); font-weight:600; }
"""


def load_grids():
    grids = {}
    for p in sorted(glob.glob("models/eval_output/nvm_grid_*.csv")):
        tag = os.path.basename(p)[len("nvm_grid_"):-len(".csv")]
        with open(p) as f:
            rows = list(csv.DictReader(f))
        if rows:
            grids[tag] = rows
    return grids


def agg(rows):
    n = len(rows) or 1
    g = lambda k: sum(float(r.get(k, 0) or 0) for r in rows) / n
    return dict(win=g("win_rate"), leak=g("mean_leakers"), bad=g("bad_death_rate"),
                xr=min(g("exchange_ratio"), 99.0), cells=len(rows))


def bar_rows(items, color_var="--c1", fmt="{:.0%}"):
    out = []
    for label, v in items:
        pct = max(0.0, min(1.0, v)) * 100
        out.append(
            f'<div class="barrow" title="{html.escape(label)}: {fmt.format(v)}">'
            f'<div class="barlab">{html.escape(label)}</div>'
            f'<div><div class="bar" style="width:{pct:.1f}%;background:var({color_var})"></div></div>'
            f'<div class="barval">{fmt.format(v)}</div></div>')
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--inline-gifs", action="store_true",
                    help="base64-embed GIFs (portable single file; bigger)")
    ap.add_argument("--max-inline-mb", type=float, default=9.0)
    args = ap.parse_args()

    grids = load_grids()
    now = time.strftime("%Y-%m-%d %H:%M")
    try:
        head = subprocess.run(["git", "log", "--oneline", "-14"], capture_output=True,
                              text=True).stdout.strip().splitlines()
    except Exception:
        head = []

    # ---- hero tiles ----
    def grid_win(tag, pred=lambda r: True):
        rows = [r for r in grids.get(tag, []) if pred(r)]
        return agg(rows)["win"] if rows else None

    tiles = []
    champ = ("aegis_adv_defender_reuse_best"
             if "model_reuseadv_vs_ar" in grids else "aegis_stage3_realism_best")
    if os.path.exists("models/aegis_adv_defender_large_best/best_model.zip"):
        champ = "aegis_adv_defender_large_best"
    elif os.path.exists("models/aegis_large_stage3_realism_best/best_model.zip"):
        champ = "aegis_large_stage3_realism_best"
    tiles.append(("Current champion", champ.replace("aegis_", "").replace("_", " ")))
    v = grid_win("model_reuse_nlm")
    if v is not None:
        tiles.append(("Outnumbered (3v5…10v20) win", f"{v:.0%}"))
    v = grid_win("model_large_scale") or grid_win("model_scale")
    if v is not None:
        tiles.append(("Scale-extrapolation win", f"{v:.0%}"))
    v = grid_win("model_reuseadv_vs_ar") or grid_win("model_advdef_vs_rl")
    if v is not None:
        tiles.append(("vs adaptive RL attackers", f"{v:.0%}"))
    for t in ("hills", "urban", "maze"):
        v = grid_win(f"model_large_{t}")
        if v is not None:
            tiles.append((f"{t.title()} terrain win", f"{v:.0%}"))
    tiles.append(("Eval runs on file", str(len(grids))))
    tile_html = "\n".join(
        f'<div class="tile"><div class="v">{html.escape(str(v))}</div>'
        f'<div class="k">{html.escape(str(k))}</div></div>' for k, v in tiles)

    # ---- leaderboard table ----
    lead = []
    for tag, rows in grids.items():
        a = agg(rows)
        lead.append((tag, a))
    lead.sort(key=lambda x: -x[1]["win"])
    lead_html = ("<table><tr><th>eval run (tag)</th><th>win</th><th>leakers/ep</th>"
                 "<th>bad-death</th><th>XR</th><th>cells</th></tr>")
    for tag, a in lead:
        lead_html += (f'<tr><td>{html.escape(tag)}</td><td class="num">{a["win"]:.0%}</td>'
                      f'<td class="num">{a["leak"]:.2f}</td><td class="num">{a["bad"]:.1%}</td>'
                      f'<td class="num">{a["xr"]:.2f}</td><td class="num">{a["cells"]}</td></tr>')
    lead_html += "</table>"

    # ---- win-rate bars (top 14 tags) ----
    bars = bar_rows([(t, a["win"]) for t, a in lead[:14]])

    # ---- terrain comparison (if large-model terrain grids exist) ----
    terrain_html = ""
    tcolors = {"hills": "--c1", "urban": "--c2", "maze": "--c3"}
    trows = []
    for t in ("hills", "urban", "maze"):
        rows = grids.get(f"model_large_{t}")
        if rows:
            for r in rows:
                trows.append((t, f'{r["n"]}v{r["m"]} {r["strategy"]}', float(r["win_rate"])))
    if trows:
        legend = "".join(
            f'<span><span class="sw" style="background:var({tcolors[t]})"></span>{t}</span>'
            for t in ("hills", "urban", "maze"))
        terrain_html = (f'<h2>Terrain generalist — win rate by cell</h2>'
                        f'<div class="legend">{legend}</div>'
                        + "\n".join(bar_rows([(f"{lab}", v)], tcolors[t])
                                    for t, lab, v in trows))

    # ---- experiments ----
    exp_html = ""
    for p in sorted(glob.glob("experiments/*/results.md"), reverse=True):
        name = p.split(os.sep)[1]
        with open(p) as f:
            body = f.read()
        exp_html += (f"<details><summary>{html.escape(name)}</summary>"
                     f"<pre>{html.escape(body)}</pre></details>")

    # ---- GIF gallery ----
    gif_html, budget = "", args.max_inline_mb * 1e6
    for p in sorted(glob.glob("models/eval_output/aegis_*.gif"),
                    key=os.path.getmtime, reverse=True):
        cap = os.path.basename(p).replace("aegis_", "").replace(".gif", "").replace("_", " ")
        if args.inline_gifs:
            sz = os.path.getsize(p)
            if sz > budget:
                continue
            budget -= sz
            with open(p, "rb") as f:
                src = "data:image/gif;base64," + base64.b64encode(f.read()).decode()
        else:
            src = os.path.basename(p)
        gif_html += (f'<figure><img loading="lazy" src="{src}" alt="{html.escape(cap)}">'
                     f'<figcaption>{html.escape(cap)}</figcaption></figure>')

    commits = "".join(
        f"<li><b>{html.escape(c.split(' ', 1)[0])}</b> {html.escape(c.split(' ', 1)[1])}</li>"
        for c in head if " " in c)

    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AEGIS Runs</title><style>{CSS}</style></head><body>
<h1>AEGIS — Drone Swarm Defense</h1>
<div class="sub">Results dashboard · generated {now} · branch DD-16 ·
regenerate with <code>python -m eval.dashboard</code></div>
<div class="tiles">{tile_html}</div>
<h2>Win rate by eval run</h2>{bars}
{terrain_html}
<h2>All eval runs</h2>{lead_html}
<h2>Episode replays</h2><div class="gifs">{gif_html}</div>
<h2>Experiment reports</h2>{exp_html}
<h2>Recent commits</h2><ul class="commits">{commits}</ul>
</body></html>"""
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(page)
    print(f"wrote {args.out} ({os.path.getsize(args.out)/1e3:.0f} kB, "
          f"{len(grids)} eval runs, {len(lead)} rows)")


if __name__ == "__main__":
    main()
