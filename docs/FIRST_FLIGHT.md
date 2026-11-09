# First CrazyFlie flight — procedure

Bridge: `deploy/cf_bridge.py` (control loop, referee and safety layer),
`deploy/mocap.py` (position sources), `deploy/cf_link.py` (cflib radio link). Model to
fly: `models/aegis_adv_defender_cf_lab_best/best_model.zip`, the adversarially hardened
lab policy (`docs/CAPABILITIES.md` §4). Nothing beyond the dry run has run on hardware.
The software path is tested end to end over a real UDP socket in
`tests/test_cf_deploy.py` (9 tests, fake radio), and the observation encoder is checked
against the simulator in `tests/test_aegis.py::test_cf_bridge_encoder_matches_env`
(agreement to within 2e-5).

Requirements: a Crazyradio, `cflib` (in `requirements.txt`), the `qtm` package only for
`--mocap qualisys`, and the model checkpoint (checkpoints are not tracked in git).

## 0. What the bridge assumes

| Quantity | Value | Set by |
|---|---|---|
| Lab frame | x, y horizontal, z up, metres; origin at the centre of the flight area and z = 0 at the floor (the policy's boundary and altitude inputs assume both) | rig / forwarder |
| Protected point | (0, 0, 0.25) m | `--target x,y,z` |
| Obstacles | none; the encoder reports an empty arena, so clear the flight area | fixed |
| Geofence | motors stop if any defender leaves \|x\|, \|y\| ≤ 3.8 m or rises above 2.2 m; also checked before arming | `--fence-xy`, `--fence-z` |
| Mocap watchdog | motors stop if no fresh frame for 0.2 s (checked every policy tick) | `--max-stale` |
| Vertical clamp | no downward command below 0.15 m, no upward command above 2.0 m | fixed (`Safety` in `cf_bridge.py`) |
| Flight-time limit | landing after 120 s | `--max-flight` |
| Run length | 720 policy ticks (≈ 85 s), then landing | `--ticks` |
| Policy rate | 8.49 Hz (117.9 ms); 4 velocity setpoints per tick (nominal 30 Hz stream) | fixed |
| Defender speed cap | 0.34 m/s at `--speed-scale 1` | `--speed-scale` |
| Referee | intercept < 0.12 m; breach < 0.15 m from the protected point | fixed |
| Take-off | climb at `takeoff_z` / 2 s for 2 s (0.2 m/s, about 0.4 m) | `--takeoff-z` |
| Landing | descend at 0.25 m/s for 3 s, then motors off | fixed |
| Telemetry | every hardware run writes `logs/flights/<YYYYmmdd_HHMMSS>.csv`: defender positions and velocities, attacker positions, commanded velocities and allocator assignment, one row per policy tick | `--telemetry PATH` |

## 1. Pick a position source

**`--mocap udp` (any rig).** Write a forwarder that reads your rig's SDK and sends one
JSON datagram per frame to the bridge (default port 9870, `--udp-port`):

```json
{"t": 12.345, "def": [[x,y,z]], "att": [[x,y,z]], "target": [0,0,0.25]}
```

Send `null` for an occluded body: its last position is held and the frame still counts
as fresh, so the watchdog does not catch a body that stays occluded. Velocities are
finite-differenced and exponentially smoothed on the bridge side (or send
`def_v`/`att_v` if your rig provides them); `deploy.mocap.udp_publish()` builds and
sends the packet. The bridge waits up to 5 s for the first frame and aborts otherwise.
With an external rig (`udp` or `qualisys`) the bridge switches each defender to the
onboard Kalman estimator (`locSrv.extPosStdDev = 0.01`), resets it, and streams the rig
positions to the CF as external position (`extpos`) with every setpoint. Pass
`--no-extpos` if the CFs localise themselves (for example with Lighthouse) and the rig
only feeds the policy. An attacker can be a hand-carried marker for the first flight.

**`--mocap cf` (Lighthouse or Loco deck).** No external rig. Each CF streams its own
state estimate (`stateEstimate.x/y/z/vx/vy/vz`) over the radio at 50 Hz. Attackers must
be CFs too, listed in `--att-uris` (observed only, never commanded); the bridge needs
exactly `--m` of them. No external position is sent in this mode. The watchdog uses the
oldest body's last update, so any CF that stops reporting stops the flight.

**`--mocap qualisys`.** Direct QTM real-time client (`pip install qtm`); bodies are
matched by their 6DOF names: `--def-bodies cf1 --att-bodies marker1 --qtm-host <ip>`.
List one defender body per `--uris` entry, in the same order, and exactly `--m` attacker
bodies. QTM reports millimetres; the client converts to metres and finite-differences
velocities.

## 2. Bench checks (props off)

```bash
# software path, no radio (the encoder check runs the GPU simulator and is skipped without CUDA)
pytest tests/test_cf_deploy.py tests/test_aegis.py::test_cf_bridge_encoder_matches_env -q

# full loop with synthetic mocap, no radio: 2v2 reference
python -m deploy.cf_bridge --model models/aegis_adv_defender_cf_lab_best/best_model.zip --dry-run

# radio + mocap plumbing: props OFF, drone on the bench. Expect it to print its
# position and wait for ENTER; press Ctrl-C at the prompt (nothing is armed).
python -m deploy.cf_bridge --model models/aegis_adv_defender_cf_lab_best/best_model.zip \
    --uris radio://0/80/2M/E7E7E7E701 --mocap udp --m 1
```

The dry run should log `INTERCEPT attacker 1` at t = 3.7 s and `INTERCEPT attacker 0`
at t = 4.1 s; its summary line reports 4.2 s because it counts the final tick.

On the radio check, confirm the printed position matches where the drone physically
is, in metres, in the lab frame. If it does not, fix the forwarder before flying. Only
the UDP source is explicitly waited for: with `cf` or `qualisys`, a printed position of
`[0, 0, 0]` means no data has arrived yet.

## 3. Flight 1: setpoint tracking, policy bypassed

One defender, no attacker needed: pass `--m 1` and have the forwarder send a fixed
attacker position well away from the protected point, e.g. `[3, 3, 0.5]`.

```bash
python -m deploy.cf_bridge --model ... --uris radio://0/80/2M/E7E7E7E701 \
    --mocap udp --m 1 --circle-test 0.5 --max-flight 20
```

The drone should climb about 0.4 m, then fly at 0.1 m/s along a circle of 0.5 m radius
centred 0.5 m in −x from its start, so keep 1 m clear on that side. A full lap takes
about 31 s, so `--max-flight 20` covers roughly two thirds of it, which is enough to
judge tracking. In the telemetry CSV, commanded velocity (`cmd0_x`, `cmd0_y`) and
measured velocity (`d0_vx`, `d0_vy`) should agree within ~0.05 m/s after the first
second. If they drift or oscillate, suspect the estimator setup (`extPosStdDev`, frame
axes, units), not the policy.

## 4. Flight 2: policy in the loop at half speed

One defender and one slow attacker: a hand-carried marker, or a second CF flown
separately and tracked by the rig, moving toward the protected point at about
0.15 m/s. At `--speed-scale 0.5` the defender is capped at 0.17 m/s.

```bash
python -m deploy.cf_bridge --model ... --uris radio://0/80/2M/E7E7E7E701 \
    --mocap udp --m 1 --speed-scale 0.5 --max-flight 60
```

Expected: the defender closes on the attacker, the referee prints `INTERCEPT attacker 0`
when they are within 0.12 m and then `ALL ATTACKERS INTERCEPTED`, and the bridge lands
automatically. If the attacker gets within 0.15 m of the protected point first, it
prints `BREACH` and lands.

## 5. Flight 3 onward: full speed, 2v2

Drop `--speed-scale`, give both defender URIs (comma-separated in `--uris`) and pass
`--m 2`. Simulation reference (dry run, 2v2, seed 3): both attackers intercepted by
t = 4.1 s.

## Abort paths

- **Ctrl-C** during the control loop: controlled landing (0.25 m/s for 3 s, then motors
  off). At the arming prompt, Ctrl-C exits without arming.
- **Geofence violation or stale mocap:** immediate stop — `send_stop_setpoint`,
  `send_notify_setpoint_stop` and a disarm request. The drone falls; fly over a net or
  foam for the first flights.
- **Flight-time or tick limit, all attackers intercepted, or breach:** the normal
  landing above.
- **Software unresponsive:** unplug the Crazyradio; the CF firmware cuts the motors when
  setpoints stop arriving.

## Known gaps in the bridge

- External position is streamed only inside the control loop: not while waiting at the
  arming prompt, nor during the 2 s take-off or the landing. With an external rig, the
  onboard filter gets no position updates in those phases.
- Landing is open-loop: 3 s at 0.25 m/s covers 0.75 m, after which the motors are cut. A
  drone higher than that when landing starts drops the rest of the way.
- A Ctrl-C during the take-off climb is not caught by the landing handler: the links
  close and the firmware's setpoint timeout stops the motors.
