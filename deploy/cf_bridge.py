"""CrazyFlie deployment bridge for the aegis_cf_lab policy.

Maps a real 8x8 m lab into AEGIS sim units (docs: experiments/night2_20260828/
results.md) and closes the loop:

    mocap (real meters) --x L--> sim units --> ObsEncoder (56-dim, minimal obs)
      --> PPO policy @ 8.49 Hz real --> action --> velocity setpoint (x 0.3394 m/s)
      --> cflib world-frame setpoints streamed at 30 Hz

Unit mapping (design-reviewed, night 2): L = 125 (length), T = sqrt(2) (time).
  x_sim = 125 * x_real          v_sim = 176.78 * v_real
  policy period real = sqrt(2)/12 s = 117.9 ms (8.49 Hz)
  defender cap 0.3394 m/s real, accel cap 3.0 m/s^2 (17 deg tilt)
Referee thresholds (ground station, not the policy): intercept 0.12 m,
breach 0.15 m, floor 0.15 m.

STATUS: encoder is cross-checked against the sim env bit-for-bit
(tests/test_cf_bridge.py). The cflib path is UNTESTED ON HARDWARE — first
flight checklist at the bottom of this file.

Usage:
    python -m deploy.cf_bridge --model models/aegis_cf_lab_stage3_realism_best/best_model.zip --dry-run
    python -m deploy.cf_bridge --model ... --uris radio://0/80/2M/E7E7E7E701 ...  # hardware
"""

import argparse
import time

import numpy as np

from envs.allocator import allocate, assigned_tgo, guard_stations, GUARD

# --- unit mapping (frozen; see night-2 design review) ---
L = 125.0
T = np.sqrt(2.0)
V_DEF_REAL = 60.0 / (L * T)          # 0.3394 m/s
SIM_DT = 1.0 / 12.0                  # policy period, sim seconds
REAL_DT = SIM_DT * T                 # 117.9 ms
STREAM_HZ = 30                       # cflib setpoint rate

# sim-side constants the encoder mirrors (must match AegisConfig lab config)
DEF_SPEED = 60.0
ARENA = 500.0
ALT_MIN, ALT_MAX = 5.0, 200.0
GUARD_RING = 100.0
REALLOC_EVERY = 6


class ObsEncoder:
    """Numpy mirror of AegisSwarmEnv._compute_obs for ONE world, minimal-obs
    profile (obs_zero_pip=True, obs_opportunistic=False), mocap sensing
    (everything detected). Cross-checked against the env in tests/test_cf_bridge.py."""

    def __init__(self, n, m):
        self.n, self.m = n, m
        self.assign = np.full((1, n), GUARD, dtype=np.int64)
        self.stations = np.zeros((1, n, 3), dtype=np.float32)
        self.steps_since_reassign = np.zeros((1, n), dtype=np.int64)
        self.step = 0

    def allocate_now(self, dp, dv, ap, av, d_alive, a_active, target):
        new = allocate(dp, d_alive, ap, av, a_active,
                       np.ones((1, self.m), dtype=bool), target, DEF_SPEED,
                       prev_assign=self.assign)
        changed = new != self.assign
        self.assign = new
        self.stations = guard_stations(self.assign, d_alive, target, GUARD_RING)
        self.steps_since_reassign[changed] = 0

    def encode(self, dp, dv, ap, av, d_alive, a_active, target, target_vel=None):
        """All inputs in SIM units. dp/dv (1,N,3), ap/av (1,M,3), target (1,3)."""
        n, m = self.n, self.m
        if self.step % REALLOC_EVERY == 0:
            self.allocate_now(dp, dv, ap, av, d_alive, a_active, target)
        self.step += 1
        self.steps_since_reassign += 1

        obs = np.zeros((1, n, 56), dtype=np.float32)
        rel_t = target[:, None, :] - dp
        dist_t = np.linalg.norm(rel_t, axis=-1)
        obs[..., 0:3] = dv / 60.0
        obs[..., 3] = dp[..., 2] / 200.0
        obs[..., 4:7] = rel_t / 500.0
        obs[..., 7] = dist_t / 500.0
        obs[..., 8] = d_alive.sum() / n
        obs[..., 9] = a_active.sum() / m

        has_t = self.assign >= 0
        a_idx = np.clip(self.assign, 0, None)[..., None]
        goal_pos = np.where(has_t[..., None],
                            np.take_along_axis(ap, a_idx, axis=1), self.stations)
        goal_vel = np.where(has_t[..., None],
                            np.take_along_axis(av, a_idx, axis=1), 0.0)
        rel = goal_pos - dp
        d = np.linalg.norm(rel, axis=-1)
        relv = goal_vel - dv
        safe = np.maximum(d, 0.01)
        obs[..., 10:13] = rel / 500.0
        obs[..., 13:16] = relv / 120.0
        obs[..., 16] = d / 500.0
        obs[..., 17] = -np.sum(rel * relv, axis=-1) / safe / 120.0
        obs[..., 18:21] = rel / safe[..., None]
        obs[..., 21] = np.linalg.norm(goal_pos - target[:, None, :], axis=-1) / 500.0
        obs[..., 22:25] = 0.0                        # obs_zero_pip (minimal build)
        obs[..., 25] = 1.0
        obs[..., 26] = has_t.astype(np.float32)
        obs[..., 27] = np.clip(self.steps_since_reassign * SIM_DT / 5.0, 0, 1)
        share = np.zeros((1, n), dtype=np.float32)
        for a in range(m):
            cnt = ((self.assign == a) & d_alive).sum()
            share = np.where(self.assign == a, max(cnt - 1, 0), share)
        obs[..., 28] = share / 4.0

        # teammate/hill sentinels then teammates (no terrain in the lab)
        obs[..., 32] = 1.0
        obs[..., 36] = 1.0
        obs[..., 40] = 1.0
        obs[..., 44] = 1.0
        dd = np.linalg.norm(dp[:, :, None, :] - dp[:, None, :, :], axis=-1)
        dd = np.where(np.eye(n, dtype=bool)[None] | ~d_alive[:, None, :], np.inf, dd)
        near2 = np.argsort(dd, axis=-1)[..., :2]
        for k in range(min(2, max(n - 1, 0))):
            idx = near2[..., k]
            td = np.take_along_axis(dd, idx[..., None], axis=-1)[..., 0]
            valid = np.isfinite(td)
            trel = np.take_along_axis(
                dp[:, None, :, :].repeat(n, 1),
                idx[..., None, None].repeat(3, -1), axis=2)[:, :, 0, :] - dp
            base = 29 + k * 4
            obs[..., base:base + 3] = np.where(valid[..., None],
                                               np.clip(trel / 100.0, -1, 1), 0.0)
            obs[..., base + 3] = np.where(valid, np.clip(td / 100.0, 0, 1), 1.0)

        obs[..., 45] = (ARENA - np.abs(dp[..., 0])) / 500.0
        obs[..., 46] = (ARENA - np.abs(dp[..., 1])) / 500.0
        obs[..., 47] = np.minimum(dp[..., 2] - ALT_MIN, ALT_MAX - dp[..., 2]) / 200.0
        # [48:56] opportunistic channel: off in the minimal build (zeros)

        obs[~d_alive] = 0.0
        return obs.reshape(n, 56)


def real_to_sim_pos(p):
    return np.asarray(p, dtype=np.float32) * L


def real_to_sim_vel(v):
    return np.asarray(v, dtype=np.float32) * L * T


def action_to_real_vel(a):
    return np.clip(a, -1, 1) * V_DEF_REAL


class SyntheticMocap:
    """Dry-run stand-in: kinematic attackers beeline the target; defenders
    integrate the commanded setpoints. All in REAL lab meters."""

    def __init__(self, n, m, seed=0):
        rng = np.random.default_rng(seed)
        self.target = np.array([0.0, 0.0, 0.25], dtype=np.float32)
        ang = rng.uniform(0, 2 * np.pi, n)
        self.dp = np.stack([0.6 * np.cos(ang), 0.6 * np.sin(ang),
                            np.full(n, 0.4)], -1).astype(np.float32)
        self.dv = np.zeros((n, 3), dtype=np.float32)
        ang = rng.uniform(0, 2 * np.pi, m)
        self.ap = np.stack([2.8 * np.cos(ang), 2.8 * np.sin(ang),
                            np.full(m, 0.5)], -1).astype(np.float32)
        self.att_speed = 50.0 / (L * T)   # 0.283 m/s real

    def step(self, def_vel_cmd, dt):
        self.dv = def_vel_cmd
        self.dp += self.dv * dt
        to_t = self.target[None] - self.ap
        dist = np.linalg.norm(to_t, axis=-1, keepdims=True)
        self.av = to_t / np.maximum(dist, 1e-6) * self.att_speed
        self.ap += self.av * dt

    def get(self):
        return self.dp, self.dv, self.ap, getattr(self, "av", np.zeros_like(self.ap)), self.target


def run(args):
    from stable_baselines3 import PPO
    model = PPO.load(args.model, device=args.device)
    n = len(args.uris.split(",")) if args.uris else args.n
    m = args.m
    enc = ObsEncoder(n, m)

    cfs = None
    if not args.dry_run:
        # --- UNTESTED ON HARDWARE: review the first-flight checklist below ---
        import cflib.crtp
        from cflib.crazyflie.swarm import CachedCfFactory, Swarm
        cflib.crtp.init_drivers()
        cfs = Swarm(args.uris.split(","), factory=CachedCfFactory(rw_cache="./cfcache"))
        cfs.open_links()
        cfs.parallel_safe(lambda scf: scf.cf.platform.send_arming_request(True))
        mocap = None  # TODO: wire your mocap source (lighthouse/Qualisys/Vicon)
        raise SystemExit("hardware path needs a mocap source wired in — see TODO")
    else:
        mocap = SyntheticMocap(n, m, seed=args.seed)

    stream_dt = 1.0 / STREAM_HZ
    steps_per_decision = max(1, round(REAL_DT / stream_dt))
    vel_cmd = np.zeros((n, 3), dtype=np.float32)
    d_alive = np.ones((1, n), dtype=bool)
    a_active = np.ones((1, m), dtype=bool)
    kills = 0

    for tick in range(args.ticks):
        dp_r, dv_r, ap_r, av_r, tgt_r = mocap.get()
        obs = enc.encode(real_to_sim_pos(dp_r)[None], real_to_sim_vel(dv_r)[None],
                         real_to_sim_pos(ap_r)[None], real_to_sim_vel(av_r)[None],
                         d_alive, a_active, real_to_sim_pos(tgt_r)[None])
        acts, _ = model.predict(obs, deterministic=True)
        vel_cmd = action_to_real_vel(acts)
        vel_cmd[~d_alive[0]] = 0.0

        for _ in range(steps_per_decision):
            if cfs is not None:
                pass  # cfs.parallel_safe(send_velocity_world_setpoint per drone)
            else:
                mocap.step(vel_cmd, stream_dt)

        # referee (ground station role): intercepts + breaches in real meters
        pd = np.linalg.norm(mocap.dp[:, None, :] - mocap.ap[None, :, :], axis=-1)
        for d, a in zip(*np.nonzero((pd < 0.12) & d_alive[0][:, None] & a_active[0][None, :])):
            a_active[0, a] = False
            kills += 1
        if (np.linalg.norm(mocap.ap - mocap.target[None], axis=-1) < 0.15)[a_active[0]].any():
            print(f"[t={tick*REAL_DT:5.1f}s] BREACH")
            break
        if not a_active.any():
            print(f"[t={tick*REAL_DT:5.1f}s] ALL ATTACKERS INTERCEPTED ({kills} kills)")
            break
    else:
        print(f"timeout: kills={kills} attackers_left={int(a_active.sum())}")
    print(f"dry-run done: {kills}/{m} intercepted, defenders at "
          f"{np.round(mocap.dp[:, :2], 2).tolist()}")


def main():
    p = argparse.ArgumentParser(description="CrazyFlie bridge for aegis_cf_lab")
    p.add_argument("--model", required=True)
    p.add_argument("--uris", default="", help="comma cf radio URIs (hardware mode)")
    p.add_argument("--n", type=int, default=2, help="defenders (dry-run)")
    p.add_argument("--m", type=int, default=2, help="attackers")
    p.add_argument("--ticks", type=int, default=720)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--device", default="cpu", help="cpu is fine at 8.49 Hz")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    run(args)


# FIRST-FLIGHT CHECKLIST (hardware path):
#  1. Wire a mocap source returning positions+velocities in lab meters at >=30 Hz.
#  2. Geofence in the referee: kill motors outside |x|,|y|<3.8 m or z>2.2 m.
#  3. One defender, one SLOW scripted "attacker" (a human-carried marker works).
#  4. Verify setpoint tracking with the policy REPLACED by a fixed 0.1 m/s circle.
#  5. Then policy-in-the-loop at 0.5x V_DEF_REAL before full speed.

if __name__ == "__main__":
    main()
