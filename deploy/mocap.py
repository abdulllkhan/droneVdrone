"""Motion-capture sources for the CrazyFlie bridge (deploy/cf_bridge.py).

Every source speaks the same contract, in REAL lab units: metres, lab world
frame (x, y horizontal, z up), velocities in m/s.

    src.start()
    dp, dv, ap, av, target = src.get()   # (N,3) (N,3) (M,3) (M,3) (3,)
    src.age()                            # seconds since the last fresh frame
    src.integrate(def_vel_cmd, dt)       # no-op except for SyntheticMocap
    src.stop()

Sources
  SyntheticMocap      dry-run stand-in (kinematic, no hardware)
  UdpMocap            generic JSON-over-UDP feed: point ANY rig at it
                      (Vicon/OptiTrack/Qualisys/Lighthouse) with a 20-line
                      forwarder; see PACKET FORMAT below
  CrazyflieStateMocap the CrazyFlies' own onboard estimate (Lighthouse or Loco
                      deck) streamed over the radio via cflib log blocks; needs
                      no external mocap server
  QualisysMocap       direct QTM real-time client (pip install qtm)

PACKET FORMAT (UdpMocap), one JSON object per datagram:
    {"t": 12.345,                      # sender clock, seconds (monotonic ok)
     "def": [[x,y,z], ...],            # N defender positions, metres
     "att": [[x,y,z], ...],            # M attacker positions, metres
     "target": [x,y,z],                # optional; else the configured point
     "def_v": [[vx,vy,vz], ...],       # optional; else finite-differenced
     "att_v": [[vx,vy,vz], ...]}       # optional; else finite-differenced
Missing bodies (occluded marker) must be sent as null; the last good position
is held and the frame still counts as fresh.
"""

import json
import socket
import threading
import time

import numpy as np


class VelocityEstimator:
    """Finite-difference + exponential smoothing. alpha=0.3 at 100 Hz gives a
    ~30 ms lag, well under the 118 ms policy period, and cuts 1 mm marker
    jitter (0.1 m/s raw) to ~0.03 m/s."""

    def __init__(self, k, alpha=0.3, max_gap=0.5):
        self.alpha, self.max_gap = alpha, max_gap
        self.v = np.zeros((k, 3), dtype=np.float32)
        self._p = None
        self._t = None

    def update(self, p, t):
        p = np.asarray(p, dtype=np.float32)
        if self._p is not None:
            dt = t - self._t
            if dt > 1e-6 and dt <= self.max_gap:
                raw = (p - self._p) / dt
                self.v = self.alpha * raw + (1.0 - self.alpha) * self.v
            elif dt > self.max_gap:
                self.v[:] = 0.0          # long dropout: don't trust the delta
        self._p, self._t = p.copy(), t
        return self.v.copy()


class MocapSource:
    n = 0
    m = 0

    def start(self):
        pass

    def stop(self):
        pass

    def get(self):
        raise NotImplementedError

    def age(self):
        return 0.0

    def integrate(self, def_vel_cmd, dt):
        pass


class SyntheticMocap(MocapSource):
    """Dry-run stand-in: kinematic attackers beeline the target; defenders
    integrate the commanded setpoints. All in REAL lab metres."""

    def __init__(self, n, m, seed=0, att_speed=None):
        from deploy.cf_bridge import L, T  # unit mapping lives with the bridge
        self.n, self.m = n, m
        rng = np.random.default_rng(seed)
        self.target = np.array([0.0, 0.0, 0.25], dtype=np.float32)
        ang = rng.uniform(0, 2 * np.pi, n)
        self.dp = np.stack([0.6 * np.cos(ang), 0.6 * np.sin(ang),
                            np.full(n, 0.4)], -1).astype(np.float32)
        self.dv = np.zeros((n, 3), dtype=np.float32)
        ang = rng.uniform(0, 2 * np.pi, m)
        self.ap = np.stack([2.8 * np.cos(ang), 2.8 * np.sin(ang),
                            np.full(m, 0.5)], -1).astype(np.float32)
        self.av = np.zeros((m, 3), dtype=np.float32)
        self.att_speed = 50.0 / (L * T) if att_speed is None else att_speed  # 0.283 m/s

    def step(self, def_vel_cmd, dt):
        self.dv = np.asarray(def_vel_cmd, dtype=np.float32)
        self.dp += self.dv * dt
        to_t = self.target[None] - self.ap
        dist = np.linalg.norm(to_t, axis=-1, keepdims=True)
        self.av = to_t / np.maximum(dist, 1e-6) * self.att_speed
        self.ap += self.av * dt

    integrate = step

    def get(self):
        return self.dp, self.dv, self.ap, self.av, self.target


class UdpMocap(MocapSource):
    """Receive frames from any rig over UDP (format in module docstring)."""

    def __init__(self, n, m, port=9870, host="0.0.0.0", target=(0.0, 0.0, 0.25),
                 alpha=0.3):
        self.n, self.m = n, m
        self.host, self.port = host, port
        self.target = np.asarray(target, dtype=np.float32)
        self.dp = np.zeros((n, 3), dtype=np.float32)
        self.dv = np.zeros((n, 3), dtype=np.float32)
        self.ap = np.zeros((m, 3), dtype=np.float32)
        self.av = np.zeros((m, 3), dtype=np.float32)
        self._dv_est = VelocityEstimator(n, alpha)
        self._av_est = VelocityEstimator(m, alpha)
        self._lock = threading.Lock()
        self._last = None          # local monotonic time of last frame
        self._frames = 0
        self._sock = None
        self._thread = None
        self._run = False

    def start(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self.port = self._sock.getsockname()[1]      # resolves port=0
        self._sock.settimeout(0.1)
        self._run = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._run = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._sock is not None:
            self._sock.close()

    def _loop(self):
        while self._run:
            try:
                data, _ = self._sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                self._ingest(json.loads(data.decode()))
            except (ValueError, KeyError, TypeError):
                continue                               # malformed packet: skip

    @staticmethod
    def _fill(dst, rows):
        """Copy rows into dst, holding the previous value where a row is null."""
        for i, r in enumerate(rows[:len(dst)]):
            if r is not None:
                dst[i] = r

    def _ingest(self, pkt):
        t = float(pkt["t"])
        with self._lock:
            self._fill(self.dp, pkt["def"])
            self._fill(self.ap, pkt["att"])
            if pkt.get("target") is not None:
                self.target = np.asarray(pkt["target"], dtype=np.float32)
            if pkt.get("def_v") is not None:
                self._fill(self.dv, pkt["def_v"])
            else:
                self.dv = self._dv_est.update(self.dp, t)
            if pkt.get("att_v") is not None:
                self._fill(self.av, pkt["att_v"])
            else:
                self.av = self._av_est.update(self.ap, t)
            self._last = time.monotonic()
            self._frames += 1

    def get(self):
        with self._lock:
            return (self.dp.copy(), self.dv.copy(), self.ap.copy(),
                    self.av.copy(), self.target.copy())

    def age(self):
        return np.inf if self._last is None else time.monotonic() - self._last

    @property
    def frames(self):
        return self._frames

    def wait_for_frame(self, timeout=5.0):
        t0 = time.monotonic()
        while self._last is None and time.monotonic() - t0 < timeout:
            time.sleep(0.01)
        return self._last is not None


def udp_publish(sock, addr, t, dp, ap, target=None, dv=None, av=None):
    """Helper for forwarders/tests: send one UdpMocap packet."""
    pkt = {"t": float(t),
           "def": np.asarray(dp, dtype=float).tolist(),
           "att": np.asarray(ap, dtype=float).tolist()}
    if target is not None:
        pkt["target"] = np.asarray(target, dtype=float).tolist()
    if dv is not None:
        pkt["def_v"] = np.asarray(dv, dtype=float).tolist()
    if av is not None:
        pkt["att_v"] = np.asarray(av, dtype=float).tolist()
    sock.sendto(json.dumps(pkt).encode(), addr)


class CrazyflieStateMocap(MocapSource):
    """Positions/velocities from each CrazyFlie's onboard Kalman estimate
    (stateEstimate.*), streamed via cflib log blocks. Works with the Lighthouse
    or Loco positioning deck. Defender CFs come from the link; attacker CFs are
    a separate list (they are only observed, never commanded here).

    Not exercised on hardware yet — see docs/FIRST_FLIGHT.md."""

    def __init__(self, link, n, m, target=(0.0, 0.0, 0.25), period_ms=20):
        self.link, self.n, self.m = link, n, m
        self.target = np.asarray(target, dtype=np.float32)
        self.period_ms = period_ms
        self.dp = np.zeros((n, 3), dtype=np.float32)
        self.dv = np.zeros((n, 3), dtype=np.float32)
        self.ap = np.zeros((m, 3), dtype=np.float32)
        self.av = np.zeros((m, 3), dtype=np.float32)
        self._stamp = np.full(n + m, -np.inf)
        self._lock = threading.Lock()
        self._configs = []

    def start(self):
        from cflib.crazyflie.log import LogConfig
        scfs = self.link.defender_cfs() + self.link.attacker_cfs()
        assert len(scfs) == self.n + self.m, "uri count != n + m"
        for i, scf in enumerate(scfs):
            lc = LogConfig(name=f"aegis{i}", period_in_ms=self.period_ms)
            for v in ("x", "y", "z", "vx", "vy", "vz"):
                lc.add_variable(f"stateEstimate.{v}", "float")
            scf.cf.log.add_config(lc)
            lc.data_received_cb.add_callback(
                lambda ts, data, lc, i=i: self._on_data(i, data))
            lc.start()
            self._configs.append(lc)

    def stop(self):
        for lc in self._configs:
            try:
                lc.stop()
            except Exception:
                pass

    def _on_data(self, i, d):
        p = (d["stateEstimate.x"], d["stateEstimate.y"], d["stateEstimate.z"])
        v = (d["stateEstimate.vx"], d["stateEstimate.vy"], d["stateEstimate.vz"])
        with self._lock:
            if i < self.n:
                self.dp[i], self.dv[i] = p, v
            else:
                self.ap[i - self.n], self.av[i - self.n] = p, v
            self._stamp[i] = time.monotonic()

    def get(self):
        with self._lock:
            return (self.dp.copy(), self.dv.copy(), self.ap.copy(),
                    self.av.copy(), self.target.copy())

    def age(self):
        with self._lock:
            return time.monotonic() - self._stamp.min()   # oldest body decides


class QualisysMocap(MocapSource):
    """Direct QTM real-time client (pip install qtm). Bodies are matched by the
    6DOF body names configured in QTM. Runs the asyncio client in a thread and
    finite-differences velocities. Not exercised on hardware yet."""

    def __init__(self, def_bodies, att_bodies, host="127.0.0.1",
                 target=(0.0, 0.0, 0.25), alpha=0.3):
        self.def_bodies, self.att_bodies = list(def_bodies), list(att_bodies)
        self.n, self.m = len(def_bodies), len(att_bodies)
        self.host = host
        self.target = np.asarray(target, dtype=np.float32)
        self.dp = np.zeros((self.n, 3), dtype=np.float32)
        self.dv = np.zeros((self.n, 3), dtype=np.float32)
        self.ap = np.zeros((self.m, 3), dtype=np.float32)
        self.av = np.zeros((self.m, 3), dtype=np.float32)
        self._dv_est = VelocityEstimator(self.n, alpha)
        self._av_est = VelocityEstimator(self.m, alpha)
        self._lock = threading.Lock()
        self._last = None
        self._thread = None
        self._loop = None

    def start(self):
        import asyncio
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import asyncio
        import xml.etree.ElementTree as ET
        import qtm

        async def main():
            conn = await qtm.connect(self.host)
            if conn is None:
                raise RuntimeError(f"QTM not reachable at {self.host}")
            xml_str = await conn.get_parameters(parameters=["6d"])
            names = [b.text.strip() for b in ET.fromstring(xml_str).iter("Name")]
            idx = {nm: i for i, nm in enumerate(names)}
            order = [idx[b] for b in self.def_bodies + self.att_bodies]

            def on_packet(packet):
                _, bodies = packet.get_6d()
                t = packet.timestamp / 1e6          # QTM stamps in microseconds
                rows = []
                for j in order:
                    pos, _rot = bodies[j]
                    rows.append(None if np.isnan(pos.x) else
                                (pos.x / 1000.0, pos.y / 1000.0, pos.z / 1000.0))
                with self._lock:
                    UdpMocap._fill(self.dp, rows[:self.n])
                    UdpMocap._fill(self.ap, rows[self.n:])
                    self.dv = self._dv_est.update(self.dp, t)
                    self.av = self._av_est.update(self.ap, t)
                    self._last = time.monotonic()

            await conn.stream_frames(components=["6d"], on_packet=on_packet)
            while True:
                await asyncio.sleep(1)

        self._loop.run_until_complete(main())

    def get(self):
        with self._lock:
            return (self.dp.copy(), self.dv.copy(), self.ap.copy(),
                    self.av.copy(), self.target.copy())

    def age(self):
        return np.inf if self._last is None else time.monotonic() - self._last
