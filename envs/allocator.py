"""Classical target-allocation layer for AEGIS (see docs/AEGIS_PLAN.md §3.1).

Assigns each alive defender at most one detected active attacker per world using a
vectorized greedy global-min sweep over a time-to-go cost matrix. Zero learning —
fully testable in pytest without RL. Self-healing falls out of re-running this on
every death/elimination/detection event: orphaned attackers are re-owned within one
allocation call.

Cost:  C[d,a] = t_go(d,a)
              + 100 * [t_go > t_breach(a) - 1s]   (infeasible-intercept penalty)
              - min(20 / max(t_breach, 0.5), 10)  (urgency: prefer imminent threats)
              - 0.5 * [d was already assigned a]  (hysteresis against flapping)

Greedy resolves ties toward lower (defender, attacker) slot index by argmin order,
which gives stable ownership between near-equidistant defenders.
"""

import numpy as np

INFEASIBLE_TGO = 60.0  # sentinel time-to-go (s) when no intercept solution exists
GUARD = -1             # assignment value for unassigned (guard-duty) defenders


def time_to_go(def_pos, def_speed, att_pos, att_vel):
    """Closed-form constant-velocity intercept time, vectorized.

    Solves |r + u*t| = v_d * t for the smallest positive t, where r = att - def
    and u = attacker velocity. Shapes: def_pos (..., N, 3), att_pos/att_vel (..., M, 3).
    Returns (..., N, M) times, INFEASIBLE_TGO where no solution.
    """
    r = att_pos[..., None, :, :] - def_pos[..., :, None, :]      # (..., N, M, 3)
    u = np.broadcast_to(att_vel[..., None, :, :], r.shape)
    a = np.sum(u * u, axis=-1) - def_speed**2                    # (..., N, M)
    b = 2.0 * np.sum(r * u, axis=-1)
    c = np.sum(r * r, axis=-1)

    tgo = np.full(a.shape, INFEASIBLE_TGO, dtype=np.float32)

    # Near-linear case (|u| ~ def_speed): b*t + c = 0
    lin = np.abs(a) < 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        t_lin = -c / b
    lin_ok = lin & (b < 0) & (t_lin > 0)
    tgo[lin_ok] = t_lin[lin_ok]

    quad = ~lin
    disc = b * b - 4.0 * a * c
    ok = quad & (disc >= 0)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-b - sq) / (2.0 * a)
        t2 = (-b + sq) / (2.0 * a)
    t1 = np.where(ok & (t1 > 0), t1, np.inf)
    t2 = np.where(ok & (t2 > 0), t2, np.inf)
    tmin = np.minimum(t1, t2)
    valid = np.isfinite(tmin)
    tgo[valid] = tmin[valid].astype(np.float32)
    return np.minimum(tgo, INFEASIBLE_TGO)


def assigned_tgo(def_pos, def_speed, att_pos, att_vel, assign):
    """t_go of each defender vs ITS assigned attacker. (W, N) — 0 for guards.

    Gathers the (W, N, M) time_to_go matrix along the ATTACKER axis (axis=2).
    A previous inline version gathered along axis=1 (the defender axis), which
    silently returned wrong values and crashed when assign >= N.
    """
    tgo = time_to_go(def_pos, def_speed, att_pos, att_vel)        # (W, N, M)
    idx = np.clip(assign, 0, None)[..., None]                      # (W, N, 1)
    out = np.take_along_axis(tgo, idx, axis=2)[..., 0]
    return np.where(assign >= 0, out, 0.0)


def build_cost(def_pos, def_alive, att_pos, att_vel, att_assignable,
               target_pos, def_speed, prev_assign, hysteresis=0.5):
    """Cost matrix (W, N, M) with np.inf for invalid pairs."""
    W, N, _ = def_pos.shape
    M = att_pos.shape[1]

    tgo = time_to_go(def_pos, def_speed, att_pos, att_vel)        # (W, N, M)

    att_speed = np.linalg.norm(att_vel, axis=-1)                  # (W, M)
    dist_to_target = np.linalg.norm(att_pos - target_pos[:, None, :], axis=-1)
    with np.errstate(divide="ignore"):
        t_breach = np.where(att_speed > 0.1, dist_to_target / np.maximum(att_speed, 0.1),
                            INFEASIBLE_TGO)
    urgency = np.minimum(20.0 / np.maximum(t_breach, 0.5), 10.0)  # (W, M)

    cost = tgo + 100.0 * (tgo > (t_breach[:, None, :] - 1.0)) - urgency[:, None, :]

    if prev_assign is not None:
        d_idx = np.arange(N)
        for w in np.nonzero((prev_assign >= 0).any(axis=1))[0]:
            mask = prev_assign[w] >= 0
            cost[w, d_idx[mask], prev_assign[w, mask]] -= hysteresis

    cost = cost.astype(np.float64)
    cost[~def_alive[:, :, None] | ~att_assignable[:, None, :]] = np.inf
    return cost


def hungarian_sweep(cost):
    """Optimal one-to-one matching per world via scipy Hungarian.

    Greedy leaves a ~13% mean total-cost gap on random matrices; Hungarian on
    <=8x8 costs microseconds per world, so optimal is the default. np.inf pairs
    are excluded from the result. Does not mutate `cost`.
    """
    from scipy.optimize import linear_sum_assignment
    W, N, M = cost.shape
    assign = np.full((W, N), GUARD, dtype=np.int64)
    BIG = 1e9
    for w in range(W):
        finite = np.isfinite(cost[w])
        if not finite.any():
            continue
        rows, cols = linear_sum_assignment(np.where(finite, cost[w], BIG))
        for d, a in zip(rows, cols):
            if finite[d, a]:
                assign[w, d] = a
    return assign


def greedy_sweep(cost):
    """One-to-one greedy matching per world on a (W, N, M) cost matrix.

    Fallback when scipy is unavailable. Repeatedly takes the global minimum cost
    pair per world, then invalidates that row and column. Returns assign (W, N)
    int with GUARD for unmatched defenders. Mutates `cost`.
    """
    W, N, M = cost.shape
    assign = np.full((W, N), GUARD, dtype=np.int64)
    flat = cost.reshape(W, N * M)
    for _ in range(min(N, M)):
        idx = np.argmin(flat, axis=1)                 # (W,)
        vals = flat[np.arange(W), idx]
        live = np.isfinite(vals)
        if not live.any():
            break
        d, a = idx // M, idx % M
        ws = np.nonzero(live)[0]
        assign[ws, d[ws]] = a[ws]
        cost[ws, d[ws], :] = np.inf                   # defender used
        cost[ws, :, a[ws]] = np.inf                   # attacker owned
    return assign


def allocate(def_pos, def_alive, att_pos, att_vel, att_active, att_detected,
             target_pos, def_speed, prev_assign=None, hysteresis=0.5,
             double_up=True):
    """Full AEGIS allocation for a batch of worlds.

    Args:
        def_pos (W,N,3), def_alive (W,N) bool
        att_pos (W,M,3), att_vel (W,M,3), att_active (W,M) bool, att_detected (W,M) bool
        target_pos (W,3), def_speed scalar (m/s)
        prev_assign (W,N) int or None — previous assignment for hysteresis
    Returns:
        assign (W,N) int64 — attacker index per defender, GUARD (-1) for guards/dead.
    """
    att_assignable = att_active & att_detected

    try:
        sweep = hungarian_sweep
        import scipy.optimize  # noqa: F401
    except ImportError:
        sweep = greedy_sweep

    cost = build_cost(def_pos, def_alive, att_pos, att_vel, att_assignable,
                      target_pos, def_speed, prev_assign, hysteresis)
    assign = sweep(cost)

    if double_up:
        # Second round: surplus defenders double up (max 2 owners per attacker),
        # preferring urgent threats via the same cost structure.
        unassigned = def_alive & (assign == GUARD)
        if unassigned.any():
            cost2 = build_cost(def_pos, unassigned, att_pos, att_vel, att_assignable,
                               target_pos, def_speed, prev_assign, hysteresis)
            assign2 = sweep(cost2)
            take = assign2 >= 0
            assign[take] = assign2[take]

    return assign


def guard_stations(assign, def_alive, target_pos, ring_radius=100.0):
    """Ring stations for guard-duty defenders.

    Guards are alive defenders with assign == GUARD. Each world's guards get evenly
    spaced ring slots (stable order by defender slot index) at the target's altitude.
    Returns (W, N, 3) station positions (zeros for non-guards).
    """
    W, N = assign.shape
    stations = np.zeros((W, N, 3), dtype=np.float32)
    guards = def_alive & (assign == GUARD)
    for w in np.nonzero(guards.any(axis=1))[0]:
        idxs = np.nonzero(guards[w])[0]
        angles = 2.0 * np.pi * np.arange(len(idxs)) / max(len(idxs), 1)
        stations[w, idxs, 0] = target_pos[w, 0] + ring_radius * np.cos(angles)
        stations[w, idxs, 1] = target_pos[w, 1] + ring_radius * np.sin(angles)
        stations[w, idxs, 2] = target_pos[w, 2]
    return stations
