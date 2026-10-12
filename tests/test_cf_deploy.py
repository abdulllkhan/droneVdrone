"""Deployment bridge: mocap sources, safety layer, and the end-to-end loop over
a real UDP socket. No hardware, no cflib radio. (The encoder-vs-sim bit check
lives in tests/test_aegis.py::test_cf_bridge_encoder_matches_env.)"""

import socket
import threading
import time

import numpy as np
import pytest

from deploy.cf_bridge import (Bridge, REAL_DT, Safety, V_DEF_REAL, clamp_vertical,
                              fence_violation)
from deploy.cf_link import FakeLink
from deploy.mocap import SyntheticMocap, UdpMocap, VelocityEstimator, udp_publish

MODEL = "models/aegis_adv_defender_cf_lab_best/best_model.zip"


class ConstPolicy:
    """Stands in for PPO: constant action so tests don't need the checkpoint."""

    def __init__(self, a):
        self.a = np.asarray(a, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return np.repeat(self.a[None], obs.shape[0], 0), None


@pytest.fixture(scope="module")
def ppo():
    pytest.importorskip("stable_baselines3")
    import os
    if not os.path.exists(MODEL):
        pytest.skip("champion checkpoint not present")
    from stable_baselines3 import PPO
    return PPO.load(MODEL, device="cpu")


def test_velocity_estimator_converges_on_linear_motion():
    est = VelocityEstimator(2, alpha=0.3)
    v = np.array([[0.3, -0.1, 0.05], [0.0, 0.2, 0.0]])
    p = np.zeros((2, 3))
    for k in range(60):                       # 0.6 s at 100 Hz
        out = est.update(p + v * k * 0.01, k * 0.01)
    assert np.allclose(out, v, atol=1e-3)
    # a long dropout resets rather than extrapolating a huge delta
    out = est.update(p + 5.0, 10.0)
    assert np.all(out == 0)


def test_udp_mocap_roundtrip_and_velocity():
    src = UdpMocap(n=1, m=1, port=0, target=(0, 0, 0.25))
    src.start()
    try:
        assert src.age() == np.inf
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = ("127.0.0.1", src.port)
        for k in range(40):
            udp_publish(sock, addr, k * 0.01, [[0.2 * k * 0.01, 0.0, 0.4]],
                        [[1.0, 1.0 - 0.1 * k * 0.01, 0.5]])
            time.sleep(0.002)
        deadline = time.monotonic() + 2.0
        while src.frames < 40 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert src.frames == 40
        dp, dv, ap, av, tgt = src.get()
        assert np.allclose(dp[0], [0.2 * 39 * 0.01, 0.0, 0.4], atol=1e-6)
        assert np.allclose(dv[0], [0.2, 0.0, 0.0], atol=2e-3)
        assert np.allclose(av[0], [0.0, -0.1, 0.0], atol=2e-3)
        assert np.allclose(tgt, [0, 0, 0.25])
        assert src.age() < 0.5
        # occluded body (null) holds the last position, frame still fresh
        sock.sendto(b'{"t": 0.5, "def": [null], "att": [[9, 9, 9]]}', addr)
        sock.sendto(b"garbage", addr)         # malformed packet is skipped
        time.sleep(0.05)
        dp2, _, ap2, _, _ = src.get()
        assert np.allclose(dp2[0], dp[0]) and np.allclose(ap2[0], [9, 9, 9])
    finally:
        src.stop()


def test_fence_and_vertical_clamp():
    s = Safety()
    p = np.array([[0.0, 0.0, 0.5], [3.9, 0.0, 0.5], [0.0, -4.0, 0.5], [0.0, 0.0, 2.3]])
    assert fence_violation(p, s).tolist() == [False, True, True, True]
    v = np.array([[0.1, 0.0, -0.2], [0.1, 0.0, 0.2], [0.0, 0.0, -0.2]], dtype=np.float32)
    z = np.array([0.10, 2.10, 1.0])
    out = clamp_vertical(v, z, s)
    assert out[0, 2] == 0.0 and out[1, 2] == 0.0 and out[2, 2] == -0.2


def test_dry_run_reproduces_reference(ppo):
    """2v2 seed 3 must still intercept both attackers at 4.1 s (night-2 number)."""
    mocap = SyntheticMocap(2, 2, seed=3)
    link = FakeLink()
    res = Bridge(ppo, mocap, link, 2, 2, realtime=False, log=lambda *_: None).run(720)
    assert res["outcome"] == "intercepted" and res["kills"] == 2
    assert abs(res["t"] - 4.2) < 0.2
    assert link.events[-1] == "land" and not link.stopped
    # 4 stream substeps per decision (118 ms / 33 ms), one command per defender each
    assert len(link.cmds) == res["ticks"] * 4 * 2


def test_geofence_stops_motors():
    mocap = SyntheticMocap(1, 1, seed=0)
    mocap.dp[0] = [3.95, 0.0, 0.4]
    link = FakeLink()
    res = Bridge(ConstPolicy([0, 0, 0]), mocap, link, 1, 1, realtime=False,
                 log=lambda *_: None).run(50)
    assert res["outcome"] == "fence" and link.stopped and link.cmds == []


def test_stale_mocap_stops_motors():
    class Stale(SyntheticMocap):
        def age(self):
            return 1.0
    link = FakeLink()
    res = Bridge(ConstPolicy([0, 0, 0]), Stale(1, 1), link, 1, 1, realtime=False,
                 log=lambda *_: None).run(50)
    assert res["outcome"] == "stale" and link.stopped


def test_speed_scale_and_circle_test():
    link = FakeLink()
    Bridge(ConstPolicy([1, 1, 0]), SyntheticMocap(1, 1), link, 1, 1,
           safety=Safety(speed_scale=0.5), realtime=False, log=lambda *_: None).run(3)
    mags = [abs(c[1]) for c in link.cmds]
    assert max(mags) == pytest.approx(0.5 * V_DEF_REAL, rel=1e-5)

    link = FakeLink()
    Bridge(ConstPolicy([1, 1, 1]), SyntheticMocap(2, 1), link, 2, 1,
           circle_test=0.5, realtime=False, log=lambda *_: None).run(20)
    speeds = [np.hypot(c[1], c[2]) for c in link.cmds]
    assert np.allclose(speeds, 0.1, atol=1e-6) and all(c[3] == 0.0 for c in link.cmds)


def test_max_flight_time_lands():
    link = FakeLink()
    res = Bridge(ConstPolicy([0, 0, 0]), SyntheticMocap(1, 1, att_speed=0.0), link, 1, 1,
                 safety=Safety(max_flight_s=1.0), realtime=False, log=lambda *_: None).run(500)
    assert res["outcome"] == "max_time" and link.events[-1] == "land"


def test_end_to_end_over_udp_with_extpos(ppo):
    """Real socket path: a forwarder thread runs the synthetic lab physics,
    integrates whatever the FakeLink last sent, and streams frames at 100 Hz.
    Velocities are finite-differenced on the receive side (no def_v/att_v)."""
    n = m = 2
    physics = SyntheticMocap(n, m, seed=3)
    link = FakeLink()
    src = UdpMocap(n, m, port=0)
    src.start()
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    addr = ("127.0.0.1", src.port)
    stop = threading.Event()

    def forwarder():
        dt = 0.01
        t0 = time.monotonic()
        k = 0
        while not stop.is_set():
            cmd = np.zeros((n, 3), dtype=np.float32)
            for i, vx, vy, vz in link.cmds[-n:]:
                cmd[i] = (vx, vy, vz)
            physics.step(cmd, dt)
            udp_publish(sock, addr, k * dt, physics.dp, physics.ap, physics.target)
            k += 1
            time.sleep(max(0.0, t0 + k * dt - time.monotonic()))

    th = threading.Thread(target=forwarder, daemon=True)
    th.start()
    try:
        assert src.wait_for_frame(2.0)
        res = Bridge(ppo, src, link, n, m, realtime=True, send_extpos=True,
                     log=lambda *_: None).run(150)
    finally:
        stop.set()
        th.join(1.0)
        src.stop()
    assert res["outcome"] == "intercepted", res
    assert res["t"] < 12.0
    assert len(link.extpos) == len(link.cmds)
    i, x, y, z = link.extpos[-1]
    assert np.allclose([x, y, z], physics.dp[i], atol=0.05)
