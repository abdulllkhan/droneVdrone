"""Pre-flight test battery for AEGIS (docs/AEGIS_PLAN.md §3.7).

Pure-numpy tests always run; GPU env tests are skipped without CUDA.
Run: python -m pytest tests/test_aegis.py -v
"""

import numpy as np
import pytest

from envs.allocator import (
    allocate, build_cost, greedy_sweep, guard_stations, time_to_go,
    GUARD, INFEASIBLE_TGO,
)
from envs.aegis_env import segment_min_dist, OBS_DIM


# ------------------------------------------------------------------ #
# Allocator — pure numpy
# ------------------------------------------------------------------ #

def test_tgo_stationary_target():
    d = np.array([[[0.0, 0.0, 0.0]]])
    a = np.array([[[300.0, 0.0, 0.0]]])
    av = np.array([[[0.0, 0.0, 0.0]]])
    t = time_to_go(d, 60.0, a, av)
    assert np.allclose(t[0, 0, 0], 300.0 / 60.0, atol=1e-4)


def test_tgo_head_on():
    d = np.array([[[0.0, 0.0, 0.0]]])
    a = np.array([[[220.0, 0.0, 0.0]]])
    av = np.array([[[-50.0, 0.0, 0.0]]])
    t = time_to_go(d, 60.0, a, av)
    assert np.allclose(t[0, 0, 0], 220.0 / 110.0, atol=1e-4)


def test_tgo_receding_faster_is_infeasible():
    d = np.array([[[0.0, 0.0, 0.0]]])
    a = np.array([[[100.0, 0.0, 0.0]]])
    av = np.array([[[70.0, 0.0, 0.0]]])   # receding at 70 > 60 pursuer
    t = time_to_go(d, 60.0, a, av)
    assert t[0, 0, 0] == pytest.approx(INFEASIBLE_TGO)


def test_allocator_full_coverage_when_enough_defenders():
    rng = np.random.default_rng(0)
    for _ in range(200):
        W, N, M = 4, 6, rng.integers(1, 7)
        dp = rng.uniform(-200, 200, (W, N, 3))
        ap = rng.uniform(-400, 400, (W, M, 3))
        av = rng.uniform(-50, 50, (W, M, 3))
        alive = np.ones((W, N), dtype=bool)
        active = rng.random((W, M)) < 0.8
        target = np.zeros((W, 3))
        assign = allocate(dp, alive, ap, av, active, np.ones((W, M), bool),
                          target, 60.0)
        for w in range(W):
            owned = set(assign[w][assign[w] >= 0])
            need = set(np.nonzero(active[w])[0])
            assert need.issubset(owned), f"unowned attackers {need - owned}"


def test_hungarian_sweep_is_optimal_and_handles_inf():
    from scipy.optimize import linear_sum_assignment
    from envs.allocator import hungarian_sweep
    rng = np.random.default_rng(1)
    for _ in range(300):
        n, m = rng.integers(2, 9), rng.integers(2, 9)
        cost = rng.uniform(0, 60, (1, n, m))
        cost[0][rng.random((n, m)) < 0.2] = np.inf   # infeasible pairs
        h = hungarian_sweep(cost)
        h_total = sum(cost[0, d, a] for d, a in enumerate(h[0]) if a >= 0)
        # reference optimum on inf-masked matrix
        finite = np.isfinite(cost[0])
        r, c = linear_sum_assignment(np.where(finite, cost[0], 1e9))
        ref = sum(cost[0][d, a] for d, a in zip(r, c) if finite[d, a])
        assert h_total == pytest.approx(ref, rel=1e-9)
        # never assigns an infeasible pair
        for d, a in enumerate(h[0]):
            if a >= 0:
                assert np.isfinite(cost[0, d, a])


def test_greedy_fallback_gap_is_bounded():
    # Greedy is only the no-scipy fallback; on uniform-random costs it runs a
    # documented ~13% mean gap vs optimal. Guard against regressions beyond that.
    from scipy.optimize import linear_sum_assignment
    rng = np.random.default_rng(1)
    gaps = []
    for _ in range(500):
        n, m = rng.integers(2, 9), rng.integers(2, 9)
        cost = rng.uniform(0, 60, (1, n, m))
        g = greedy_sweep(cost.copy())
        g_total = sum(cost[0, d, a] for d, a in enumerate(g[0]) if a >= 0)
        r, c = linear_sum_assignment(cost[0])
        h_total = cost[0][r, c].sum()
        if h_total > 1e-6:
            gaps.append((g_total - h_total) / h_total)
    assert np.mean(gaps) < 0.20, f"mean greedy gap {np.mean(gaps):.3f}"


def test_greedy_index_math_and_exclusivity():
    cost = np.array([[[1.0, 2.0], [0.5, 9.0], [3.0, 0.7]]])   # (1, 3, 2)
    g = greedy_sweep(cost.copy())
    # global min 0.5 -> d1:a0; next best remaining col 1: 0.7 -> d2:a1; d0 guard
    assert g[0, 1] == 0 and g[0, 2] == 1 and g[0, 0] == GUARD
    owned = g[0][g[0] >= 0]
    assert len(owned) == len(set(owned))


def test_hysteresis_keeps_incumbent_on_near_tie():
    dp = np.array([[[0.0, 0, 0], [1.0, 0, 0]]])
    ap = np.array([[[100.0, 0, 0], [100.5, 0, 0]]])
    av = np.zeros((1, 2, 3))
    alive = np.ones((1, 2), bool)
    act = np.ones((1, 2), bool)
    prev = np.array([[1, 0]])   # crossed assignment, nearly equal costs
    assign = allocate(dp, alive, ap, av, act, act, np.zeros((1, 3)), 60.0,
                      prev_assign=prev)
    assert assign[0, 0] == 1 and assign[0, 1] == 0


def test_surplus_defenders_double_up_then_guard():
    dp = np.random.default_rng(2).uniform(-50, 50, (1, 5, 3))
    ap = np.array([[[200.0, 0, 50]]])
    av = np.array([[[-40.0, 0, 0]]])
    alive = np.ones((1, 5), bool)
    act = np.ones((1, 1), bool)
    assign = allocate(dp, alive, ap, av, act, act, np.zeros((1, 3)), 60.0)
    owners = (assign[0] == 0).sum()
    guards = (assign[0] == GUARD).sum()
    assert owners == 2 and guards == 3   # 2:1 double-up, rest guard


def test_assigned_tgo_gathers_attacker_axis():
    from envs.allocator import assigned_tgo, time_to_go
    rng = np.random.default_rng(5)
    W, N, M = 3, 2, 4   # M > N: the old axis=1 gather crashed or lied here
    dp = rng.uniform(-100, 100, (W, N, 3))
    ap = rng.uniform(-300, 300, (W, M, 3))
    av = rng.uniform(-50, 50, (W, M, 3))
    assign = rng.integers(-1, M, (W, N))
    got = assigned_tgo(dp, 60.0, ap, av, assign)
    full = time_to_go(dp, 60.0, ap, av)
    for w in range(W):
        for d in range(N):
            want = full[w, d, assign[w, d]] if assign[w, d] >= 0 else 0.0
            assert got[w, d] == pytest.approx(want)


def test_guard_station_ring():
    assign = np.full((1, 3), GUARD)
    alive = np.ones((1, 3), bool)
    target = np.array([[10.0, -5.0, 40.0]])
    st = guard_stations(assign, alive, target, ring_radius=100.0)
    r = np.linalg.norm(st[0, :, :2] - target[0, :2], axis=-1)
    assert np.allclose(r, 100.0, atol=1e-4)
    assert np.allclose(st[0, :, 2], 40.0)


# ------------------------------------------------------------------ #
# Swept-segment collision — tunneling proof
# ------------------------------------------------------------------ #

def test_segment_min_dist_tunneling():
    # 110 m/s closing at 12 Hz: relative position swaps sides in one step.
    p0 = np.array([[5.0, 0.0, 0.0]])
    p1 = np.array([[-5.0, 0.0, 0.0]])
    d = segment_min_dist(p0, p1)
    assert d[0] < 1e-6           # crosses through zero
    assert np.linalg.norm(p0) > 3.0 and np.linalg.norm(p1) > 3.0


def test_segment_min_dist_no_motion():
    p = np.array([[4.0, 3.0, 0.0]])
    assert segment_min_dist(p, p)[0] == pytest.approx(5.0)


def test_segment_min_dist_endpoint_min():
    p0 = np.array([[10.0, 0.0, 0.0]])
    p1 = np.array([[6.0, 0.0, 0.0]])
    assert segment_min_dist(p0, p1)[0] == pytest.approx(6.0)


# ------------------------------------------------------------------ #
# GPU env tests
# ------------------------------------------------------------------ #

def _cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


needs_cuda = pytest.mark.skipif(not _cuda_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def small_env():
    import warp as wp
    wp.init()
    from envs.aegis_env import AegisSwarmEnv, AegisConfig
    cfg = AegisConfig(n_max=3, m_max=2, n_range=(3, 3), m_range=(2, 2),
                      num_obstacles=2, sensor_range=350.0, seed=7)
    env = AegisSwarmEnv(num_worlds=4, config=cfg)
    return env


@needs_cuda
def test_env_reset_contract(small_env):
    obs = small_env.reset()
    assert obs.shape == (12, OBS_DIM)
    assert np.isfinite(obs).all()


@needs_cuda
def test_env_step_invariants(small_env):
    env = small_env
    env.reset()
    rng = np.random.default_rng(3)
    targets_before = env.target_pos.copy()
    steps_before = env.step_count.copy()
    for i in range(300):
        acts = rng.uniform(-1, 1, (env.n_envs, 3)).astype(np.float32)
        env.step_async(acts)
        obs, rew, done, infos = env.step_wait()
        assert obs.shape == (env.n_envs, OBS_DIM) and np.isfinite(obs).all()
        assert rew.shape == (env.n_envs,) and np.isfinite(rew).all()
        assert np.abs(rew).max() <= 100.0
        # dead/unused slots emit the all-zero sentinel
        dead = ~env.def_alive.reshape(-1)
        assert np.abs(obs[dead]).sum() == 0.0
        # every detected active attacker owned when enough defenders alive
        for w in range(env.num_worlds):
            need = env.att_active[w] & env.att_detected[w]
            if env.def_alive[w].sum() >= need.sum():
                owned = set(env.assign[w][env.assign[w] >= 0])
                assert set(np.nonzero(need)[0]).issubset(owned)
        # per-world target immutable unless that world reset this step
        reset_worlds = env.step_count < steps_before
        stable = ~reset_worlds
        assert np.array_equal(env.target_pos[stable], targets_before[stable])
        targets_before = env.target_pos.copy()
        steps_before = env.step_count.copy()


@needs_cuda
def test_env_episode_infos_and_dones(small_env):
    env = small_env
    env.reset()
    env.pop_outcomes()
    saw_episode_info = saw_terminal_obs = False
    for _ in range(900):
        acts = np.zeros((env.n_envs, 3), dtype=np.float32)
        env.step_async(acts)
        obs, rew, done, infos = env.step_wait()
        for i, inf in enumerate(infos):
            if "episode" in inf:
                saw_episode_info = True
                assert done[i]
            if "terminal_observation" in inf:
                saw_terminal_obs = True
                assert inf["terminal_observation"].shape == (OBS_DIM,)
        if saw_episode_info and saw_terminal_obs:
            break
    assert saw_episode_info and saw_terminal_obs
    outs = env.pop_outcomes()
    assert len(outs) > 0
    for o in outs:
        assert o["n"] == 3 and o["m"] == 2
        assert o["intercepts"] + o["breaches"] + o["terrain_kills"] >= 0


@needs_cuda
def test_reassignment_is_reward_neutral(small_env):
    """White-box: after a forced reallocation, shaping anchors equal current dist."""
    env = small_env
    env.reset()
    for _ in range(5):
        env.step_async(np.zeros((env.n_envs, 3), dtype=np.float32))
        env.step_wait()
    # Force: eliminate every assigned attacker of world 0 and reallocate
    w = 0
    assigned = env.assign[w][env.assign[w] >= 0]
    if len(assigned):
        env.att_active[w, assigned[0]] = False
        env.realloc_needed[w] = True
        mask = np.zeros(env.num_worlds, dtype=bool)
        mask[w] = True
        old_assign = env.assign.copy()
        env._reallocate(mask)
        changed = (env.assign != old_assign)
        pot = env._pot_dist()
        # anchors for changed agents must equal their CURRENT potential distance
        assert np.allclose(env.prev_pot_dist[changed], pot[changed], atol=1e-4)


@needs_cuda
def test_obs_contract_across_sizes():
    import warp as wp
    wp.init()
    from envs.aegis_env import AegisSwarmEnv, AegisConfig
    e1 = AegisSwarmEnv(2, AegisConfig(n_max=2, m_max=2, n_range=(2, 2),
                                      m_range=(2, 2), seed=1))
    e2 = AegisSwarmEnv(2, AegisConfig(n_max=8, m_max=8, n_range=(3, 8),
                                      m_range=(1, 8), seed=1))
    assert e1.observation_space == e2.observation_space
    assert e1.action_space == e2.action_space
    assert e1.reset().shape[1] == OBS_DIM and e2.reset().shape[1] == OBS_DIM
