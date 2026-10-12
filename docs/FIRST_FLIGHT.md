# First CrazyFlie flight — procedure

Bridge: `deploy/cf_bridge.py` (loop + safety), `deploy/mocap.py` (position
sources), `deploy/cf_link.py` (cflib radio). Model to fly:
`models/aegis_adv_defender_cf_lab_best/best_model.zip`. Everything below the
dry run is **untested on hardware**; the software path is verified end to end
over a real UDP socket in `tests/test_cf_deploy.py`.

## 0. What the bridge assumes

| Quantity | Value | Where |
|---|---|---|
| Lab frame | x, y horizontal, z up, metres, origin = protected point by default | `--target x,y,z` |
| Fence | \|x\|,\|y\| < 3.8 m, z < 2.2 m, else motors stop | `--fence-xy --fence-z` |
| Mocap watchdog | no frame for 0.2 s → motors stop | `--max-stale` |
| Policy rate | 8.49 Hz (118 ms), setpoints streamed at 30 Hz | fixed |
| Defender speed cap | 0.34 m/s at `--speed-scale 1` | `--speed-scale` |
| Referee | intercept < 0.12 m, breach < 0.15 m from target | fixed |
| Takeoff | vertical climb to 0.4 m over 2 s, land at 0.25 m/s for 3 s | `--takeoff-z` |

Runs on hardware always write telemetry to `logs/flights/<timestamp>.csv`
(positions, velocities, commands, allocator assignment per tick).

## 1. Pick a position source

**`--mocap udp` (any rig).** Write a forwarder that reads your rig's SDK and
sends one JSON datagram per frame to the bridge (default port 9870):

```json
{"t": 12.345, "def": [[x,y,z]], "att": [[x,y,z]], "target": [0,0,0.25]}
```

Send `null` for an occluded body; the last position is held. Velocities are
finite-differenced and smoothed on the bridge side (or send `def_v`/`att_v`
if your rig provides them). `deploy.mocap.udp_publish()` builds the packet.
The bridge streams these positions to each CF as external position
(`extpos`) so the onboard Kalman filter tracks the rig. An attacker can be
a human-carried marker for the first flight.

**`--mocap cf` (Lighthouse or Loco deck).** No external rig. Each CF reports
its own `stateEstimate` over the radio at 50 Hz. Attackers must be CFs too,
listed in `--att-uris` (observed only, never commanded). Add `--no-extpos`.

**`--mocap qualisys`.** Direct QTM client (`pip install qtm`), bodies by
6DOF name: `--def-bodies cf1 --att-bodies marker1 --qtm-host <ip>`.

## 2. Bench checks (props off)

```bash
# software path, no radio
pytest tests/test_cf_deploy.py tests/test_aegis.py::test_cf_bridge_encoder_matches_env -q

# radio + mocap plumbing: props OFF, drone on the bench, expect it to print its
# position and wait for ENTER. Ctrl-C at the prompt.
python -m deploy.cf_bridge --model models/aegis_adv_defender_cf_lab_best/best_model.zip \
    --uris radio://0/80/2M/E7E7E7E701 --mocap udp --m 1
```

Confirm the printed position matches where the drone physically is, in
metres, in the lab frame. If it does not, fix the forwarder before flying.

## 3. Flight 1: setpoint tracking, policy bypassed

One defender, no attacker needed (pass `--m 1` and send any attacker position
far away, e.g. `[3,3,0.5]`).

```bash
python -m deploy.cf_bridge --model ... --uris radio://0/80/2M/E7E7E7E701 \
    --mocap udp --m 1 --circle-test 0.5 --max-flight 20
```

The drone should climb to 0.4 m, then fly a 0.5 m radius circle at 0.1 m/s.
Check the telemetry CSV: commanded vs measured velocity should agree within
~0.05 m/s after the first second. If it drifts or oscillates, the estimator
setup (`extPosStdDev`, frame axes, units) is wrong, not the policy.

## 4. Flight 2: policy in the loop at half speed

One defender, one slow attacker (walked in by hand or a second CF beelining
the target at ~0.15 m/s).

```bash
python -m deploy.cf_bridge --model ... --uris radio://0/80/2M/E7E7E7E701 \
    --mocap udp --m 1 --speed-scale 0.5 --max-flight 60
```

Expected: the defender moves toward the attacker's predicted intercept point
and the referee prints `INTERCEPT` when within 0.12 m. The run ends with an
automatic landing.

## 5. Flight 3+: full speed, 2v2

Drop `--speed-scale`, add the second defender URI and `--m 2`. Sim reference
for 2v2 (seed 3): both attackers intercepted at 4.1 s.

## Abort paths

- Ctrl-C at any point: controlled landing.
- Geofence or stale mocap: `send_stop_setpoint` + disarm immediately (drone
  falls; fly over a net or foam for the first flights).
- Kill the Crazyradio USB if the software is unresponsive: the CF firmware
  cuts motors when setpoints stop arriving.
