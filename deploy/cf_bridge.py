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

STATUS: encoder is cross-checked against the sim env (max abs diff < 2e-5)
(tests/test_aegis.py::test_cf_bridge_encoder_matches_env). Mocap sources and
the cflib command path are wired (deploy/mocap.py, deploy/cf_link.py) and
exercised end-to-end over UDP in tests/test_cf_deploy.py, but have NOT flown
real hardware yet. Procedure: docs/FIRST_FLIGHT.md.

Safety layer (Bridge, ground-station role): geofence kill, stale-mocap
watchdog, floor/ceiling clamp on the vertical command, speed scaling for
early flights, and a policy-free circle test for setpoint tracking.

Usage:
    # dry run, no hardware
    python -m deploy.cf_bridge --model models/aegis_adv_defender_cf_lab_best/best_model.zip --dry-run
    # hardware, external rig pushing UDP frames (see deploy/mocap.py)
    python -m deploy.cf_bridge --model ... --uris radio://0/80/2M/E7E7E7E701 --mocap udp --speed-scale 0.5
    # hardware, Lighthouse/Loco onboard estimate, attacker CF observed only
    python -m deploy.cf_bridge --model ... --uris radio://0/80/2M/E7E7E7E701 --att-uris radio://0/80/2M/E7E7E7E702 --mocap cf
"""

import argparse
import csv
import os
import time
from dataclasses import dataclass

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


def lab_env_kwargs():
    """AegisConfig fields for the sim side of this lab, i.e. exactly what
    ObsEncoder mirrors: minimal obs, mocap sensing (every attacker detected),
    no terrain or wind indoors. eval.aegis_viewer builds its cf_lab world from this."""
    return dict(defender_speed_limit=DEF_SPEED, arena_bounds=ARENA,
                altitude_min=ALT_MIN, altitude_max=ALT_MAX,
                guard_ring_radius=GUARD_RING, realloc_every=REALLOC_EVERY,
                obs_zero_pip=True, obs_opportunistic=False, obs_noise_std=0.0,
                sensor_range=0.0, terrain="none", num_obstacles=0,
                wind_enabled=False)


class ObsEncoder:
    """Numpy mirror of AegisSwarmEnv._compute_obs for ONE world, minimal-obs
    profile (obs_zero_pip=True, obs_opportunistic=False), mocap sensing
    (everything detected). Cross-checked against the env in
    tests/test_aegis.py::test_cf_bridge_encoder_matches_env."""

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


from deploy.mocap import SyntheticMocap  # noqa: E402  (re-exported for callers)

INTERCEPT_M = 0.12      # referee thresholds, real metres
BREACH_M = 0.15


@dataclass
class Safety:
    fence_xy: float = 3.8        # |x|,|y| beyond this -> kill (8x8 m lab)
    fence_z: float = 2.2         # ceiling
    z_floor: float = 0.15        # no downward command below this altitude
    z_ceil: float = 2.0          # no upward command above this
    max_stale_s: float = 0.2     # mocap silence beyond this -> kill
    speed_scale: float = 1.0     # first flights: 0.5 (checklist step 5)
    max_flight_s: float = 120.0


def clamp_vertical(v, z, safety):
    """Never command through the floor or ceiling (ground-station guard)."""
    v = v.copy()
    v[:, 2] = np.where((z < safety.z_floor) & (v[:, 2] < 0), 0.0, v[:, 2])
    v[:, 2] = np.where((z > safety.z_ceil) & (v[:, 2] > 0), 0.0, v[:, 2])
    return v


def fence_violation(p, safety):
    return (np.abs(p[:, 0]) > safety.fence_xy) | (np.abs(p[:, 1]) > safety.fence_xy) \
        | (p[:, 2] > safety.fence_z)


class Telemetry:
    def __init__(self, path, n, m):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.f = open(path, "w", newline="")
        self.w = csv.writer(self.f)
        cols = ["t", "tick"]
        cols += [f"d{i}_{c}" for i in range(n) for c in "xyz"]
        cols += [f"d{i}_v{c}" for i in range(n) for c in "xyz"]
        cols += [f"a{j}_{c}" for j in range(m) for c in "xyz"]
        cols += [f"cmd{i}_{c}" for i in range(n) for c in "xyz"]
        cols += [f"assign{i}" for i in range(n)]
        self.w.writerow(cols)

    def row(self, t, tick, dp, dv, ap, cmd, assign):
        self.w.writerow([f"{t:.4f}", tick] + [f"{x:.4f}" for x in
                        np.concatenate([dp.ravel(), dv.ravel(), ap.ravel(), cmd.ravel()])]
                        + assign.ravel().tolist())

    def close(self):
        self.f.close()


class Bridge:
    """mocap -> encoder -> policy -> link, with the referee and safety guards.

    circle_test: if set (radius m), the policy is bypassed and every defender
    flies a 0.1 m/s horizontal circle (checklist step 4)."""

    def __init__(self, model, mocap, link, n, m, safety=None, circle_test=None,
                 telemetry=None, realtime=None, send_extpos=False, log=print):
        self.model, self.mocap, self.link = model, mocap, link
        self.n, self.m = n, m
        self.safety = safety or Safety()
        self.circle_test = circle_test
        self.telemetry = telemetry
        self.realtime = link.hardware if realtime is None else realtime
        self.send_extpos = send_extpos
        self.log = log
        self.enc = ObsEncoder(n, m)

    def _command(self, obs, tick):
        if self.circle_test is not None:
            w = 0.1 / self.circle_test                     # 0.1 m/s tangential
            ph = w * tick * REAL_DT + np.arange(self.n) * 2 * np.pi / self.n
            return np.stack([-0.1 * np.sin(ph), 0.1 * np.cos(ph),
                             np.zeros(self.n)], -1).astype(np.float32)
        acts, _ = self.model.predict(obs, deterministic=True)
        return action_to_real_vel(acts) * self.safety.speed_scale

    def run(self, ticks):
        s = self.safety
        stream_dt = 1.0 / STREAM_HZ
        steps_per_decision = max(1, round(REAL_DT / stream_dt))
        d_alive = np.ones((1, self.n), dtype=bool)
        a_active = np.ones((1, self.m), dtype=bool)
        kills = 0
        outcome = "timeout"
        t0 = time.monotonic()
        tick = 0
        try:
            for tick in range(ticks):
                t_now = tick * REAL_DT
                age = self.mocap.age()
                if age > s.max_stale_s:
                    self.log(f"[t={t_now:5.1f}s] MOCAP STALE ({age:.2f}s) -> STOP")
                    outcome = "stale"
                    break
                dp_r, dv_r, ap_r, av_r, tgt_r = self.mocap.get()
                bad = fence_violation(dp_r, s)
                if bad.any():
                    self.log(f"[t={t_now:5.1f}s] GEOFENCE defenders {np.nonzero(bad)[0].tolist()} "
                             f"at {np.round(dp_r[bad], 2).tolist()} -> STOP")
                    outcome = "fence"
                    break
                if t_now > s.max_flight_s:
                    self.log(f"[t={t_now:5.1f}s] MAX FLIGHT TIME -> LAND")
                    outcome = "max_time"
                    break

                obs = self.enc.encode(real_to_sim_pos(dp_r)[None], real_to_sim_vel(dv_r)[None],
                                      real_to_sim_pos(ap_r)[None], real_to_sim_vel(av_r)[None],
                                      d_alive, a_active, real_to_sim_pos(tgt_r)[None])
                vel_cmd = clamp_vertical(self._command(obs, tick), dp_r[:, 2], s)
                vel_cmd[~d_alive[0]] = 0.0

                for k in range(steps_per_decision):
                    for i in range(self.n):
                        if self.send_extpos:
                            self.link.send_extpos(i, self.mocap.get()[0][i])
                        self.link.send_velocity(i, vel_cmd[i])
                    self.mocap.integrate(vel_cmd, stream_dt)
                    if self.realtime:
                        deadline = t0 + t_now + (k + 1) * stream_dt
                        time.sleep(max(0.0, deadline - time.monotonic()))

                # referee (ground station): intercepts + breaches in real metres
                dp_r, dv_r, ap_r, _, tgt_r = self.mocap.get()
                if self.telemetry is not None:
                    self.telemetry.row(t_now, tick, dp_r, dv_r, ap_r, vel_cmd, self.enc.assign)
                pd = np.linalg.norm(dp_r[:, None, :] - ap_r[None, :, :], axis=-1)
                hit = (pd < INTERCEPT_M) & d_alive[0][:, None] & a_active[0][None, :]
                for _, a in zip(*np.nonzero(hit)):
                    if a_active[0, a]:
                        a_active[0, a] = False
                        kills += 1
                        self.log(f"[t={t_now:5.1f}s] INTERCEPT attacker {a}")
                if (np.linalg.norm(ap_r - tgt_r[None], axis=-1) < BREACH_M)[a_active[0]].any():
                    self.log(f"[t={t_now:5.1f}s] BREACH")
                    outcome = "breach"
                    break
                if not a_active.any():
                    self.log(f"[t={t_now:5.1f}s] ALL ATTACKERS INTERCEPTED ({kills} kills)")
                    outcome = "intercepted"
                    break
        except KeyboardInterrupt:
            self.log("Ctrl-C -> LAND")
            outcome = "aborted"
        finally:
            if outcome in ("fence", "stale"):
                self.link.stop()                       # motors off NOW
            else:
                self.link.land(duration=3.0)
            if self.telemetry is not None:
                self.telemetry.close()
        return {"outcome": outcome, "kills": kills, "ticks": tick + 1,
                "t": (tick + 1) * REAL_DT, "attackers_left": int(a_active.sum())}


def build(args):
    """CLI -> (model, mocap, link, n, m). Kept separate so tests can compose."""
    from stable_baselines3 import PPO
    model = PPO.load(args.model, device=args.device)
    def_uris = [u for u in args.uris.split(",") if u]
    att_uris = [u for u in args.att_uris.split(",") if u]
    n = len(def_uris) if def_uris else args.n
    m = args.m
    target = tuple(float(x) for x in args.target.split(","))

    if args.dry_run:
        from deploy.cf_link import NullLink
        link = NullLink()
        mocap = SyntheticMocap(n, m, seed=args.seed)
        return model, mocap, link, n, m
    if not def_uris:
        raise SystemExit("hardware mode needs --uris (or pass --dry-run)")

    from deploy.cf_link import CflibLink
    link = CflibLink(def_uris, att_uris, rate_hz=STREAM_HZ)
    if args.mocap == "udp":
        from deploy.mocap import UdpMocap
        mocap = UdpMocap(n, m, port=args.udp_port, target=target)
    elif args.mocap == "cf":
        from deploy.mocap import CrazyflieStateMocap
        if len(att_uris) != m:
            raise SystemExit("--mocap cf observes attackers via --att-uris; need exactly --m of them")
        mocap = CrazyflieStateMocap(link, n, m, target=target)
    elif args.mocap == "qualisys":
        from deploy.mocap import QualisysMocap
        mocap = QualisysMocap(args.def_bodies.split(","), args.att_bodies.split(","),
                              host=args.qtm_host, target=target)
    else:
        raise SystemExit("hardware mode needs --mocap udp|cf|qualisys")
    return model, mocap, link, n, m


def run(args):
    model, mocap, link, n, m = build(args)
    safety = Safety(fence_xy=args.fence_xy, fence_z=args.fence_z,
                    max_stale_s=args.max_stale, speed_scale=args.speed_scale,
                    max_flight_s=args.max_flight)
    external = args.mocap in ("udp", "qualisys")
    telem = None
    if args.telemetry:
        telem = Telemetry(args.telemetry, n, m)

    link.open()
    mocap.start()
    try:
        if link.hardware:
            link.setup_estimator(external)
            if hasattr(mocap, "wait_for_frame") and not mocap.wait_for_frame(5.0):
                raise SystemExit("no mocap frames within 5 s — check the forwarder/port")
            dp0 = mocap.get()[0]
            if fence_violation(dp0, safety).any():
                raise SystemExit(f"a defender starts outside the fence: {dp0.tolist()}")
            print(f"defenders at {np.round(dp0, 2).tolist()}, speed x{args.speed_scale}, "
                  f"fence +-{args.fence_xy} m / {args.fence_z} m, stale>{args.max_stale}s kills")
            input("ENTER to arm + take off (Ctrl-C to abort) > ")
            link.arm()
            link.takeoff(args.takeoff_z, duration=2.0)
        bridge = Bridge(model, mocap, link, n, m, safety=safety,
                        circle_test=args.circle_test, telemetry=telem,
                        send_extpos=external and link.hardware and not args.no_extpos)
        res = bridge.run(args.ticks)
    finally:
        mocap.stop()
        link.close()

    dp = mocap.get()[0]
    mode = "dry-run" if args.dry_run else "flight"
    print(f"{mode} {res['outcome']}: {res['kills']}/{m} intercepted in {res['t']:.1f}s, "
          f"defenders at {np.round(dp[:, :2], 2).tolist()}")
    return res


def main():
    p = argparse.ArgumentParser(description="CrazyFlie bridge for aegis_cf_lab")
    p.add_argument("--model", required=True)
    p.add_argument("--uris", default="", help="comma defender CF radio URIs (hardware)")
    p.add_argument("--att-uris", default="", help="attacker CF URIs, observed only (--mocap cf)")
    p.add_argument("--n", type=int, default=2, help="defenders (dry-run)")
    p.add_argument("--m", type=int, default=2, help="attackers")
    p.add_argument("--ticks", type=int, default=720)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--device", default="cpu", help="cpu is fine at 8.49 Hz")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--mocap", default="", choices=["", "udp", "cf", "qualisys"])
    p.add_argument("--udp-port", type=int, default=9870)
    p.add_argument("--qtm-host", default="127.0.0.1")
    p.add_argument("--def-bodies", default="", help="qualisys 6DOF body names, comma")
    p.add_argument("--att-bodies", default="")
    p.add_argument("--target", default="0,0,0.25", help="protected point x,y,z metres")
    p.add_argument("--takeoff-z", type=float, default=0.4)
    p.add_argument("--speed-scale", type=float, default=1.0, help="first flights: 0.5")
    p.add_argument("--circle-test", type=float, default=None, metavar="RADIUS_M",
                   help="bypass policy: 0.1 m/s circle for setpoint-tracking check")
    p.add_argument("--fence-xy", type=float, default=3.8)
    p.add_argument("--fence-z", type=float, default=2.2)
    p.add_argument("--max-stale", type=float, default=0.2)
    p.add_argument("--max-flight", type=float, default=120.0)
    p.add_argument("--no-extpos", action="store_true",
                   help="don't stream external positions to the CFs (Lighthouse rigs)")
    p.add_argument("--telemetry", default="", help="CSV path (default logs/flights/<ts>.csv on hardware)")
    args = p.parse_args()
    if not args.dry_run and not args.telemetry:
        args.telemetry = os.path.join("logs", "flights", time.strftime("%Y%m%d_%H%M%S") + ".csv")
    run(args)


# FIRST-FLIGHT CHECKLIST (hardware path) — full procedure in docs/FIRST_FLIGHT.md:
#  1. Mocap: --mocap udp (any rig via forwarder), cf (Lighthouse/Loco onboard) or qualisys.
#  2. Geofence is on by default: motors stop outside |x|,|y|<3.8 m or z>2.2 m; stale mocap >0.2 s stops.
#  3. One defender, one SLOW "attacker" (a human-carried marker works with --mocap udp).
#  4. Setpoint tracking first: --circle-test 0.5 (policy bypassed, 0.1 m/s circle).
#  5. Then policy-in-the-loop with --speed-scale 0.5 before full speed.

if __name__ == "__main__":
    main()
