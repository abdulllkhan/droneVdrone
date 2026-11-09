"""AEGIS live viewer: per-model env profiles, launcher commands, headless smoke.

Pure tests need no GPU; the env/viewer tests need CUDA and skip when a champion
checkpoint is missing."""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

from eval.aegis_common import (CHAMPIONS, REALISM, config_kwargs, default_scenario,
                               env_kwargs, model_profile, paired_defender)

VARK = "models/aegis_vark_stage3_realism_best/best_model.zip"
CF_LAB = "models/aegis_adv_defender_cf_lab_best/best_model.zip"
CF_ATT = "models/aegis_adv_attacker_cf_lab_best/best_model.zip"


def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")


def _needs(path):
    return pytest.mark.skipif(not os.path.exists(path), reason=f"{path} not present")


# ------------------------------------------------------------------ #
# Per-model profiles (pure)
# ------------------------------------------------------------------ #

# eval flags from experiments/*/run_queue.sh (see eval/aegis_common.py)
EXPECTED = {
    "aegis_vark_stage3_realism_best": dict(family="vark", nm=(15, 15), num_targets=3,
                                           max_targets=4, strategy="direct", realism=True),
    "aegis_big24_stage3_realism_best": dict(family="big24", nm=(24, 24), num_targets=1,
                                            strategy="saturation", realism=True),
    "aegis_oomval_stage3_realism_best": dict(family="oomval", nm=(20, 20), num_targets=1,
                                             realism=True),
    "aegis_adv_defender_cf_lab_best": dict(family="cf_lab", nm=(2, 2), num_targets=1,
                                           max_targets=1, n_lim=(1, 4), m_lim=(1, 3),
                                           terrain="none", realism=False, lab=True),
    "aegis_convoy_stage3_realism_best": dict(family="convoy", nm=(8, 10), target_speed=10.0,
                                             num_targets=1, realism=True),
    "aegis_splitspawn_stage3_realism_best": dict(family="split", nm=(15, 15), num_targets=3,
                                                 realism=True),
    "aegis_sensrobust_stage3_realism_best": dict(family="sensrobust", nm=(15, 15),
                                                 num_targets=3, realism=True),
}


def test_champion_profiles():
    assert set(EXPECTED) == set(CHAMPIONS)
    for name, exp in EXPECTED.items():
        prof = model_profile(f"models/{name}/best_model.zip")
        assert prof["kamikaze"] is False and prof["attacker"] is False, name
        for k, v in exp.items():
            assert prof[k] == v, (name, k, prof[k])
        if prof["num_targets"] > 1:
            assert prof["env"]["target_spread"] == 180.0


def test_cf_lab_profile_is_the_bridge_lab_world():
    from deploy.cf_bridge import lab_env_kwargs
    prof = model_profile(CF_LAB)
    assert prof["env"] == lab_env_kwargs()
    kw = config_kwargs(prof, default_scenario(prof))
    assert kw["obs_zero_pip"] is True and kw["obs_opportunistic"] is False
    assert kw["sensor_range"] == 0.0 and kw["num_obstacles"] == 0
    assert kw["wind_enabled"] is False and kw["terrain"] == "none"
    assert kw["kamikaze"] is False and kw["fixed_nm"] == (2, 2)
    # the pre-hardening cf_lab model and the final zip share the lab profile
    for p in ("models/aegis_cf_lab_stage3_realism_best/best_model.zip",
              "models/aegis_adv_defender_cf_lab_final.zip"):
        assert model_profile(p)["env"] == lab_env_kwargs()


def test_config_precedence():
    prof = model_profile(VARK)
    sc = default_scenario(prof, seed=3)
    kw = config_kwargs(prof, sc)
    assert kw["num_targets"] == 3 and kw["target_spread"] == 180.0
    assert kw["fixed_nm"] == (15, 15) and kw["attacker_strategies"] == ("direct",)
    assert all(kw[k] == v for k, v in REALISM.items()) and kw["seed"] == 3
    # realism world beats the model env; --set overrides beat everything
    lab = model_profile(CF_LAB)
    kw = config_kwargs(lab, dict(default_scenario(lab), realism=True, terrain="hills"),
                       overrides={"sensor_range": 150.0})
    assert kw["num_obstacles"] == 5 and kw["wind_enabled"] and kw["terrain"] == "hills"
    assert kw["sensor_range"] == 150.0 and kw["obs_zero_pip"] is True
    kw = config_kwargs(prof, dict(sc, strategy="rl"), attacker_model=CF_ATT)
    assert kw["attacker_model_path"] == CF_ATT and kw["attacker_strategies"] == ("rl",)
    assert "attacker_model_path" not in config_kwargs(prof, sc, attacker_model=CF_ATT)


def test_env_kwargs_matches_render_aegis_rollout():
    """render_aegis.rollout() builds its env from env_kwargs: same dict as before."""
    kw = env_kwargs(5, 8, False, "mixed", True, 7, terrain="urban", target_speed=4.0,
                    num_targets=2)
    assert kw == dict(n_max=5, m_max=8, fixed_nm=(5, 8), n_range=(5, 5), m_range=(8, 8),
                      kamikaze=False, attacker_strategies=("mixed",), seed=7,
                      terrain="urban", num_targets=2, target_speed_range=(4.0, 4.0),
                      num_obstacles=5, wind_enabled=True, wind_max_force=0.5,
                      wind_gust_prob=0.01, wind_gust_force=1.5, sensor_range=300.0)
    assert "num_obstacles" not in env_kwargs(3, 3, True, "direct", False, 0)


def test_profile_name_variants():
    # final zips and checkpoints resolve to the same family as best_model dirs
    assert model_profile("models/aegis_vark_stage3_realism_final.zip")["family"] == "vark"
    assert model_profile("models/aegis_big24_stage3_realism_ckpt/"
                         "aegis_big24_stage3_realism_1000_steps.zip")["family"] == "big24"
    # DD-16 policies trained kamikaze; reusable ones (reuse/large/adv reuse) not
    for name in ("aegis_stage3_realism_best", "aegis_patch_stage3_realism_best",
                 "aegis_scratch3_stage3_realism_best", "aegis_adv_defender_best",
                 "aegis_adv_defender_r2_best"):
        assert model_profile(f"models/{name}/best_model.zip")["kamikaze"] is True, name
    for name in ("aegis_reuse_stage3_realism_best", "aegis_large_stage3_realism_best",
                 "aegis_adv_defender_reuse_best", "aegis_brake_stage3_realism_best"):
        assert model_profile(f"models/{name}/best_model.zip")["kamikaze"] is False, name
    s0 = model_profile("models/aegis_patch_stage0_pursuit_best/best_model.zip")
    assert s0["nm"] == (1, 1) and s0["realism"] is False and s0["kamikaze"] is True
    assert s0["terrain"] == "none"
    brake = model_profile("models/aegis_brake_stage3_realism_best/best_model.zip")
    assert brake["env"]["obstacle_speed_limit"] == 0.25


def test_attacker_models_pair_with_their_defender():
    prof = model_profile(CF_ATT)
    assert prof["attacker"] and prof["strategy"] == "rl" and prof["family"] == "cf_lab"
    assert paired_defender(CF_ATT) == str(Path(CF_LAB))
    assert model_profile("models/aegis_adv_attacker_varkadv_best/best_model.zip")[
        "defender"] == "aegis_vark_stage3_realism_best"
    assert paired_defender("models/aegis_adv_attacker_final.zip") == str(
        Path("models/aegis_adv_defender_best/best_model.zip"))
    # unknown round / checkpoint name -> the general default (vark)
    for p in ("models/aegis_adv_attacker_new_best/best_model.zip",
              "models/aegis_adv_attacker_99840_steps.zip"):
        assert model_profile(p)["defender"] == "aegis_vark_stage3_realism_best", p


# ------------------------------------------------------------------ #
# Viewer CLI + launcher commands (pure)
# ------------------------------------------------------------------ #

def test_viewer_cli_overrides_profile_defaults():
    from eval.aegis_viewer import outcome_title, parse_args, scenario_from_args
    prof = model_profile(VARK)
    args = parse_args(["--model", VARK])
    assert args.kamikaze is None and args.realism is None
    assert scenario_from_args(prof, args) == default_scenario(prof)
    args = parse_args(["--model", VARK, "--nm", "4x6", "--num-targets", "2", "--plain",
                       "--kamikaze", "--terrain", "maze", "--target-speed", "5"])
    sc = scenario_from_args(prof, args)
    assert (sc["n"], sc["m"], sc["num_targets"], sc["terrain"]) == (4, 6, 2, "maze")
    assert sc["realism"] is False and sc["kamikaze"] is True and sc["target_speed"] == 5.0
    assert scenario_from_args(prof, parse_args([]), attacker_model=CF_ATT)["strategy"] == "rl"
    assert outcome_title({"win": True}, k=3)[0] == "TARGETS DEFENDED"
    assert outcome_title({"win": False, "breaches": 2})[0] == "BREACHED x2"
    assert outcome_title({"win": False, "breaches": 0, "defenders_left": 0})[0] == "DEFENDERS LOST"


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_launcher_commands_aegis_vs_legacy(tmp_path):
    from eval.launcher import AEGIS_MODES, LEGACY_MODES, ModelInfo, build_command, modes_for
    py = sys.executable
    root = tmp_path / "models"
    vark = ModelInfo(_touch(root / "aegis_vark_stage3_realism_best" / "best_model.zip"))
    assert vark.name == "aegis_vark_stage3_realism_best" and vark.is_aegis
    assert vark.category == "AEGIS Champions" and modes_for(vark) == AEGIS_MODES
    assert (vark.n_def, vark.n_att) == (15, 15)
    base = [py, "-m", "eval.aegis_viewer", "--model", str(vark.path)]
    assert build_command(vark, "Watch") == base
    assert build_command(vark, "Small Team") == base + ["--nm", "3x3"]
    assert build_command(vark, "Large Swarm") == base + ["--nm", "20x20"]

    lab = ModelInfo(_touch(root / "aegis_adv_defender_cf_lab_best" / "best_model.zip"))
    assert build_command(lab, "Large Swarm")[-2:] == ["--nm", "4x3"]
    assert build_command(lab, "Small Team")[-2:] == ["--nm", "3x3"]

    att_path = _touch(root / "aegis_adv_attacker_cf_lab_best" / "best_model.zip")
    att = ModelInfo(att_path)
    assert att.category == "AEGIS RL Attackers"
    assert build_command(att, "Watch") == [
        py, "-m", "eval.aegis_viewer",
        "--model", str(root / "aegis_adv_defender_cf_lab_best" / "best_model.zip"),
        "--attacker-model", str(att_path)]

    # legacy SAC models: the original eval.game commands, unchanged
    leg_path = _touch(root / "swarm_15v10_300m_stage3_final.zip")
    leg = ModelInfo(leg_path)
    assert not leg.is_aegis and modes_for(leg) == LEGACY_MODES
    assert leg.category == "Swarm 15v10 (300m)" and leg.name == "swarm_15v10_300m_stage3_final"
    assert build_command(leg, "Watch Swarm") == [
        py, "-m", "eval.game", "--mode", "swarm", "--defender", str(leg_path),
        "--defenders", "15", "--attackers", "10", "--realistic", "--strategy", "mixed"]
    assert build_command(leg, "Interactive Spawn") == [
        py, "-m", "eval.game", "--mode", "spawn", "--defender", str(leg_path), "--realistic"]
    one = ModelInfo(_touch(root / "phase3_intercept_final.zip"))
    assert build_command(one, "1v1 Watch") == [
        py, "-m", "eval.game", "--mode", "watch", "--defender", str(one.path)]


# ------------------------------------------------------------------ #
# GPU: the viewer's lab world vs the deploy encoder, headless viewer loop
# ------------------------------------------------------------------ #

@needs_cuda
def test_viewer_lab_world_matches_cf_bridge_encoder():
    """The cf_lab world the viewer builds yields the deployment encoder's obs."""
    from deploy.cf_bridge import ObsEncoder
    from eval.aegis_common import make_env
    prof = model_profile(CF_LAB)
    env = make_env(config_kwargs(prof, dict(default_scenario(prof), n=3, m=3, seed=9)))
    env.reset()
    rng = np.random.default_rng(0)
    for _ in range(40):
        env.step_async(rng.uniform(-1, 1, (env.n_envs, 3)).astype(np.float32))
        env.step_wait()
    enc = ObsEncoder(3, 3)
    enc.assign, enc.stations = env.assign.copy(), env.stations.copy()
    enc.steps_since_reassign = env.steps_since_reassign - 1   # encode() adds 1
    enc.step = 1                                             # skip internal realloc
    obs = enc.encode(env.def_pos.copy(), env.def_vel.copy(), env.att_pos.copy(),
                     env.att_vel.copy(), env.def_alive.copy(), env.att_active.copy(),
                     env.target_pos.copy())
    assert np.abs(obs - env.def_obs_buf.reshape(3, 56)).max() < 2e-5
    env.close()


@pytest.fixture
def headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")


@needs_cuda
@_needs(VARK)
def test_viewer_headless_smoke(headless, tmp_path):
    """~30 frames of the real loop on the vark champion, 3 targets, + screenshot."""
    from PIL import Image
    from eval import aegis_viewer
    shot = tmp_path / "vark.png"
    v = aegis_viewer.main(["--model", VARK, "--nm", "6x6", "--frames", "30",
                           "--warmup", "12", "--speed", "4", "--screenshot", str(shot)])
    assert v.env is None and v.sc["num_targets"] == 3 and v.cur["t"] > 0
    img = np.asarray(Image.open(shot).convert("RGB"))
    assert img.shape[0] > 600 and len(np.unique(img.reshape(-1, 3), axis=0)) > 50
    gold = np.all(np.abs(img.astype(int) - (255, 215, 0)) < 8, axis=-1)
    assert gold[:, :v.panel_x].sum() > 100            # target stars on the map


@needs_cuda
@_needs(VARK)
def test_viewer_keys_rebuild_and_capture_terminal_frame(headless):
    import pygame
    from eval import aegis_viewer
    from eval.aegis_common import load_policy
    prof = model_profile(VARK)
    v = aegis_viewer.Viewer(load_policy(VARK), VARK, prof,
                            dict(default_scenario(prof), n=4, m=3))
    for _ in range(5):
        assert v.tick()
    v.handle_key(pygame.K_n)                          # N 4 -> 5
    v.handle_key(pygame.K_m, pygame.KMOD_SHIFT)       # M 3 -> 2
    v.handle_key(pygame.K_2)                          # 2 targets
    v.handle_key(pygame.K_t)                          # hills -> urban
    v.handle_key(pygame.K_s)                          # direct -> flanking
    assert v.pending is not None and v.env.n_max == 4  # debounced, not applied yet
    v.apply_pending()
    c = v.env.cfg
    assert (v.env.n_max, v.env.m_max, v.env.K) == (5, 2, 2)
    assert c.terrain == "urban" and c.attacker_strategies == ("flanking",)
    assert c.target_spread == 180.0 and c.num_obstacles == 5
    for _ in range(5):
        assert v.tick()
    # play to the end: the hook must hand back the real final frame even though
    # step_wait() has already respawned the world
    for _ in range(v.env.max_steps + 1):
        if v.step_once():
            break
    o, f = v.outcome, v.cur
    assert o is not None and v.env.step_count[0] == 0
    assert f["t"] == o["length"] and f["ic"] == o["intercepts"]
    assert int(f["ab"].sum()) == o["breaches"] and int(f["da"].sum()) == o["defenders_left"]
    v.draw()                                          # outcome banner path
    v.update(aegis_viewer.HOLD_SECS + 0.1)            # banner expires -> next episode
    assert v.cur["t"] == 0 and v.stats["episodes"] == 1 and v.episode == 2
    v.show_help = True
    v.draw()
    assert v.handle_key(pygame.K_q) is False
    v.close()


@needs_cuda
@_needs(CF_ATT)
@_needs(CF_LAB)
def test_viewer_rl_attacker_matchup(headless):
    """An RL attacker zip opens against its round's defender as strategy 'rl'."""
    from eval import aegis_viewer
    from eval.aegis_common import load_policy
    v = aegis_viewer.main(["--model", CF_ATT, "--frames", "10", "--warmup", "6"])
    assert v.model_path == str(Path(CF_LAB)) and v.attacker_model == CF_ATT
    assert v.sc["strategy"] == "rl" and v.prof["family"] == "cf_lab"
    assert v.cur["t"] > 0
    # the env really flies the frozen attacker (lab world, 41-dim attacker obs)
    v = aegis_viewer.Viewer(load_policy(v.model_path), v.model_path, v.prof, v.sc,
                            attacker_model=CF_ATT)
    assert len(v.env.attacker_models) == 1 and v.env.cfg.attacker_strategies == ("rl",)
    assert v.env.cfg.obs_zero_pip and v.env.att_obs_buf.shape[1] == 41
    for _ in range(10):
        assert v.tick()
    v.close()
