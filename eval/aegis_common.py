"""Shared AEGIS eval plumbing: per-model profiles, env construction, frame capture.

Model zips don't store their env config (it is passed at eval time), so
model_profile() maps a model path to the settings its policy was trained and
evaluated with. Sources: the eval flags in experiments/*/run_queue.sh and
docs/CAPABILITIES.md. Used by eval.render_aegis (GIFs), eval.aegis_viewer (live
view) and eval.launcher (browser). warp / SB3 are imported lazily.
"""

import re
from pathlib import Path

STRATEGIES = ("direct", "flanking", "mixed", "jink", "saturation", "waves", "decoy")
TERRAINS = ("hills", "urban", "maze", "none")
DEFAULT_MODEL = "models/aegis_vark_stage3_realism_best/best_model.zip"

# "realism" world (render_aegis default): 5 hills, wind + gusts, 300 m sensing
REALISM = dict(num_obstacles=5, wind_enabled=True, wind_max_force=0.5,
               wind_gust_prob=0.01, wind_gust_force=1.5, sensor_range=300.0)

# the 7 regression-verified champions (docs/CAPABILITIES.md), launcher order
CHAMPIONS = ("aegis_vark_stage3_realism_best", "aegis_big24_stage3_realism_best",
             "aegis_oomval_stage3_realism_best", "aegis_adv_defender_cf_lab_best",
             "aegis_convoy_stage3_realism_best", "aegis_splitspawn_stage3_realism_best",
             "aegis_sensrobust_stage3_realism_best")

# Profile = env settings the policy needs + viewer defaults. `env` sits UNDER the
# scenario (team sizes, strategy, terrain, targets) and the realism world; --set
# overrides beat everything. n_lim/m_lim bound the viewer's N/M keys.
_BASE = dict(family="aegis", label="AEGIS swarm policy", note="",
             kamikaze=False, realism=True, env={}, lab=False, attacker=False,
             nm=(8, 8), nm_large=(20, 20), strategy="mixed", terrain="hills",
             num_targets=1, target_speed=0.0, max_targets=4,
             n_lim=(1, 50), m_lim=(1, 50), terrains=TERRAINS)

# first substring match wins; comments cite each family's eval flags
_FAMILIES = (
    # scorecard_20260829 / cfadv_20260828: kamikaze=False obs_opportunistic=False
    # obs_zero_pip=True, cells 1x1..3x3 (N<=4, M<=3), direct/jink. World = the
    # sim side of deploy/cf_bridge.py (lab_env_kwargs, encoder-verified)
    ("cf_lab", dict(family="cf_lab", label="CrazyFlie lab policy (minimal obs)",
                    note="trained 1-4 v 1-3, open lab, no wind", lab=True,
                    realism=False, nm=(2, 2), nm_large=(4, 3), strategy="direct",
                    terrain="none", terrains=("none",), max_targets=1,
                    n_lim=(1, 4), m_lim=(1, 3))),
    # varktargets_20260830: --realism kamikaze=False num_targets=K target_spread=180
    ("vark", dict(family="vark", label="multi-target generalist (1-4 points)",
                  note="trained 6-20 v 3-20, 1-4 targets, hills/open",
                  nm=(15, 15), strategy="direct", num_targets=3,
                  env=dict(target_spread=180.0))),
    # sensorrobust_20260830: vark flags (+ sensor_range=150/75 stress cells)
    ("sensrobust", dict(family="sensrobust", label="multi-target, sensor-robust",
                        note="trained 6-20 v 3-20, 1-4 targets, sensors 80-350 m",
                        nm=(15, 15), strategy="direct", num_targets=3,
                        env=dict(target_spread=180.0))),
    # split/splitdirect/splitspawn: --realism kamikaze=False num_targets=3 spread 180
    ("split", dict(family="split", label="3-point split defense",
                   note="trained 6-20 v 3-20, 3 targets, hills/open",
                   nm=(15, 15), strategy="direct", num_targets=3,
                   env=dict(target_spread=180.0))),
    # night2_20260828 C8 / convoyadv: --realism kamikaze=False target_speed 10 m/s
    ("convoy", dict(family="convoy", label="moving-target escort",
                    note="trained 4-20 v 1-20, target 0-10 m/s, hills/urban/maze",
                    nm=(8, 10), nm_large=(15, 10), strategy="saturation",
                    target_speed=10.0)),
    # bigswarm_20260829: --realism kamikaze=False 15x15..24x24, scale to 50x40
    ("big24", dict(family="big24", label="single-target scale reference",
                   note="trained 6-24 v 1-24, hills/urban/maze; holds to 50v40",
                   nm=(24, 24), nm_large=(40, 40), strategy="saturation")),
    # oomfix_20260829: --realism kamikaze=False 10x10/20x20, scale 40x30/50x40
    ("oomval", dict(family="oomval", label="single-target scale reference",
                    note="trained 4-20 v 1-20, hills/urban/maze",
                    nm=(20, 20), nm_large=(40, 30), strategy="saturation")),
    # brake_20260829: the speed brake is dynamics, so evals pass it too
    ("brake", dict(family="brake", label="big24 + obstacle speed brake",
                   note="trained 4-20 v 1-20 with the speed brake",
                   nm=(12, 12), strategy="saturation",
                   env=dict(obstacle_speed_limit=0.25, obstacle_brake_range=60.0))),
    # overnight_20260827: large/largeP terrain generalists, kamikaze=False
    ("large", dict(family="large", label="large-swarm terrain generalist",
                   note="trained 4-20 v 1-20, hills/urban/maze",
                   nm=(10, 10), strategy="saturation")),
)

# DD-16-era policies kept the AegisConfig default kamikaze=True (stage-3 realism,
# 3-8 v 1-8); the reference GIF is render_aegis --kamikaze --nm 7x5 --strategy mixed
_KAMIKAZE = re.compile(r"aegis_(stage\d|patch_|scratch3|adv_defender(_r\d+)?_(best|final))")
_KAMIKAZE_PROFILE = dict(kamikaze=True, label="kamikaze interceptor (DD-16)",
                         note="trained 3-8 v 1-8, kamikaze", nm=(7, 5), nm_large=(8, 8))
# early curriculum stages trained without terrain/wind (train/train_aegis.py)
_STAGE_TWEAKS = (
    ("stage0", dict(nm=(1, 1), nm_large=(3, 3), strategy="direct", realism=False,
                    terrain="none", note="1v1 pursuit stage, no terrain",
                    env=dict(attacker_speed_scale=0.8))),
    ("stage1", dict(nm=(3, 2), nm_large=(4, 3), strategy="direct", realism=False,
                    terrain="none", note="2-4 v 1-3 stage, no terrain")),
)

# RL attackers (evade mode, 41-dim obs) are watched as the 'rl' strategy against
# the defender of their adversarial round (experiments/*adv*/run_queue.sh)
_ATTACKER_PAIRS = {"": "aegis_adv_defender_best", "_r2": "aegis_adv_defender_r2_best",
                   "_r3": "aegis_adv_defender_r2_best",
                   "_reuse": "aegis_adv_defender_reuse_best",
                   "_cf_lab": "aegis_adv_defender_cf_lab_best",
                   "_large": "aegis_large_stage3_realism_best",
                   "_convoyadv": "aegis_convoy_stage3_realism_best",
                   "_varkadv": "aegis_vark_stage3_realism_best"}


def model_name(path):
    """models/<dir>/best_model.zip -> <dir>; models/<x>.zip -> <x>."""
    p = Path(path)
    return p.parent.name if p.name == "best_model.zip" else p.stem


def model_profile(path):
    """Model path -> env settings its policy needs (env, kamikaze, realism) plus
    viewer defaults (team sizes, strategy, terrain, targets, limits). Name-based."""
    name = model_name(path).lower()
    if "aegis_adv_attacker" in name:
        tag = re.match(r"aegis_adv_attacker(.*?)_(best|final)$", name)
        defender = _ATTACKER_PAIRS.get(tag.group(1) if tag else None,
                                       model_name(DEFAULT_MODEL))
        prof = model_profile(defender)          # the matchup is played in its world
        prof.update(name=name, attacker=True, defender=defender, strategy="rl")
        return prof
    prof = dict(_BASE, name=name)
    for key, over in _FAMILIES:
        if key in name:
            prof.update(over)
            break
    if _KAMIKAZE.match(name):
        prof.update(_KAMIKAZE_PROFILE)
    for key, over in _STAGE_TWEAKS:
        if key in name:
            prof.update(over)
    prof["env"] = dict(prof["env"])
    if prof["lab"]:
        from deploy.cf_bridge import lab_env_kwargs
        prof["env"].update(lab_env_kwargs())
    return prof


def paired_defender(path):
    """RL attacker model -> the defender zip it is watched against."""
    p = Path(path)
    root = p.parent.parent if p.name == "best_model.zip" else p.parent
    return str(root / model_profile(path)["defender"] / "best_model.zip")


def env_kwargs(n, m, kamikaze, strategy, realism, seed, terrain="hills",
               target_speed=0.0, num_targets=1, extra=None):
    """AegisConfig kwargs for one fixed-(N, M) eval world.
    Precedence: extra (model env) < scenario < realism world."""
    kw = dict(extra or {})
    kw.update(n_max=n, m_max=m, fixed_nm=(n, m), n_range=(n, n), m_range=(m, m),
              kamikaze=kamikaze, attacker_strategies=(strategy,), seed=seed,
              terrain=terrain, num_targets=num_targets,
              target_speed_range=(target_speed, target_speed))
    if realism:
        kw.update(REALISM)
    return kw


def default_scenario(profile, seed=7):
    """Viewer scenario (what the keys edit) from a profile's defaults."""
    n, m = profile["nm"]
    return dict(n=n, m=m, strategy=profile["strategy"], terrain=profile["terrain"],
                num_targets=profile["num_targets"], target_speed=profile["target_speed"],
                kamikaze=profile["kamikaze"], realism=profile["realism"], seed=seed)


def config_kwargs(profile, sc, attacker_model="", overrides=None):
    """Profile + scenario -> AegisConfig kwargs; `overrides` (--set) win."""
    kw = env_kwargs(sc["n"], sc["m"], sc["kamikaze"], sc["strategy"], sc["realism"],
                    sc["seed"], terrain=sc["terrain"], target_speed=sc["target_speed"],
                    num_targets=sc["num_targets"], extra=profile["env"])
    if attacker_model and sc["strategy"] == "rl":   # frozen RL attacker flies 'rl'
        kw["attacker_model_path"] = attacker_model
    kw.update(overrides or {})
    return kw


def make_env(kw, num_worlds=1):
    import warp as wp
    wp.init()
    from envs.aegis_env import AegisSwarmEnv, AegisConfig
    return AegisSwarmEnv(num_worlds=num_worlds, config=AegisConfig(**kw))


def load_policy(path, device="cuda"):
    from stable_baselines3 import PPO
    return PPO.load(path, device=device)


def frame(env, w=0):
    """World w's drawable state (the render_aegis GIF frame schema)."""
    return dict(dp=env.def_pos[w].copy(), ap=env.att_pos[w].copy(),
                da=env.def_alive[w].copy(), aa=env.att_active[w].copy(),
                ab=env.att_breached[w].copy(), at=env.att_terrain[w].copy(),
                assign=env.assign[w].copy(), t=int(env.step_count[w]),
                ic=int(env.ep_intercepts[w]), tp=env.target_pos[w].copy())


def world_meta(env, w=0):
    """World w's per-episode scene (terrain, targets, bounds). Capture it before
    stepping: a finished world auto-resets inside step_wait()."""
    return dict(target=env.target_pos[w].copy(),
                hills=(env.obstacle_pos[w][:env.hill_count[w]].copy()
                       if env.hill_count[w] else None),
                boxes=(env.box_obs[w][:env.box_count[w]].copy()
                       if env.box_count[w] else None),
                hill_r=env.cfg.obstacle_radius, bounds=env.cfg.arena_bounds,
                targets=env.targets[w].copy())
