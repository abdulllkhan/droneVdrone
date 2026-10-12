"""Radio links for the CrazyFlie bridge.

    CflibLink   real drones over Crazyradio (cflib). UNTESTED ON HARDWARE.
    NullLink    dry-run: accepts every command, does nothing.
    FakeLink    tests: records every command.

All velocities are world-frame m/s; positions metres. The link never decides
anything — the Bridge owns safety (geofence, watchdog) and calls stop().
"""

import time

import numpy as np


class NullLink:
    hardware = False

    def open(self):
        pass

    def defender_cfs(self):
        return []

    def attacker_cfs(self):
        return []

    def setup_estimator(self, external):
        pass

    def arm(self):
        pass

    def takeoff(self, z, duration):
        pass

    def send_velocity(self, i, v):
        pass

    def send_extpos(self, i, p):
        pass

    def land(self, duration):
        pass

    def stop(self):
        pass

    def close(self):
        pass


class FakeLink(NullLink):
    """Records everything so tests can assert on what would have been sent."""

    def __init__(self):
        self.cmds = []          # (i, vx, vy, vz)
        self.extpos = []        # (i, x, y, z)
        self.events = []        # 'open', 'arm', 'takeoff', 'land', 'stop', 'close'
        self.stopped = False

    def open(self):
        self.events.append("open")

    def arm(self):
        self.events.append("arm")

    def takeoff(self, z, duration):
        self.events.append("takeoff")

    def send_velocity(self, i, v):
        self.cmds.append((i, float(v[0]), float(v[1]), float(v[2])))

    def send_extpos(self, i, p):
        self.extpos.append((i, float(p[0]), float(p[1]), float(p[2])))

    def land(self, duration):
        self.events.append("land")

    def stop(self):
        self.stopped = True
        self.events.append("stop")

    def close(self):
        self.events.append("close")


class CflibLink:
    """Defender CFs are commanded; attacker CFs (optional) are only opened so
    CrazyflieStateMocap can log their state.

    Estimator: with the Lighthouse/Loco deck the CF localises itself and
    external=False. With an external rig (UDP/Qualisys) external=True switches
    the CF to the Kalman estimator and the bridge streams send_extpos() at the
    setpoint rate so the onboard controller has a position to track."""

    hardware = True

    def __init__(self, def_uris, att_uris=(), cache="./cfcache", rate_hz=30):
        self.def_uris = list(def_uris)
        self.att_uris = list(att_uris)
        self.cache = cache
        self.rate_hz = rate_hz
        self.swarm = None
        self._def = []
        self._att = []

    def open(self):
        import cflib.crtp
        from cflib.crazyflie.swarm import CachedCfFactory, Swarm
        cflib.crtp.init_drivers()
        uris = self.def_uris + self.att_uris
        self.swarm = Swarm(uris, factory=CachedCfFactory(rw_cache=self.cache))
        self.swarm.open_links()
        self._def = [self.swarm._cfs[u] for u in self.def_uris]
        self._att = [self.swarm._cfs[u] for u in self.att_uris]

    def defender_cfs(self):
        return self._def

    def attacker_cfs(self):
        return self._att

    def setup_estimator(self, external):
        def cfg(scf):
            if external:
                scf.cf.param.set_value("stabilizer.estimator", "2")   # Kalman
                scf.cf.param.set_value("locSrv.extPosStdDev", "0.01")
            scf.cf.param.set_value("kalman.resetEstimation", "1")
            time.sleep(0.1)
            scf.cf.param.set_value("kalman.resetEstimation", "0")
        self.swarm.parallel_safe(lambda scf: cfg(scf))
        time.sleep(2.0)                                # let the filter settle

    def arm(self):
        def _arm(scf):
            try:
                scf.cf.platform.send_arming_request(True)
            except Exception:
                pass                                   # older firmware: no arming
        for scf in self._def:
            _arm(scf)
        time.sleep(0.5)

    def _stream(self, v_fn, duration):
        dt = 1.0 / self.rate_hz
        t0 = time.monotonic()
        k = 0
        while (t := time.monotonic() - t0) < duration:
            for i, scf in enumerate(self._def):
                vx, vy, vz = v_fn(i, t)
                scf.cf.commander.send_velocity_world_setpoint(vx, vy, vz, 0.0)
            k += 1
            time.sleep(max(0.0, t0 + k * dt - time.monotonic()))

    def takeoff(self, z, duration=2.0):
        """Vertical climb at z/duration m/s. Call after arm()."""
        self._stream(lambda i, t: (0.0, 0.0, z / duration), duration)

    def send_velocity(self, i, v):
        self._def[i].cf.commander.send_velocity_world_setpoint(
            float(v[0]), float(v[1]), float(v[2]), 0.0)

    def send_extpos(self, i, p):
        self._def[i].cf.extpos.send_extpos(float(p[0]), float(p[1]), float(p[2]))

    def land(self, duration=3.0):
        self._stream(lambda i, t: (0.0, 0.0, -0.25), duration)
        self.stop()

    def stop(self):
        for scf in self._def:
            try:
                scf.cf.commander.send_stop_setpoint()
                scf.cf.commander.send_notify_setpoint_stop()
                scf.cf.platform.send_arming_request(False)
            except Exception:
                pass

    def close(self):
        if self.swarm is not None:
            self.swarm.close_links()
