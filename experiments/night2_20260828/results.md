# Night 2 (2026-08-27/28) — CrazyFlie-Ready Policy + Convoy Escort

## ⭐ `aegis_cf_lab` — the validated CrazyFlie 2.1 policy (Phase-5 gateway)
Strategy: **pure unit mapping** (design-reviewed): L=125, T=√2 → the sim policy
runs a real 8×8 m lab at **0.34 m/s max, 3 m/s² (17° tilt), 8.49 Hz control**,
gravity compensation cancels on both sides. Trained with the **minimal sensor
suite** (PIP + opportunistic channels zeroed; per the knockout study) on small
teams N∈[1,4] vs M∈[1,3], kamikaze=False. Gate (200 eps/cell) vs the largeP
control on the same obs:

| Cell class | cf_lab | largeP control |
|---|---|---|
| All N≥M cells (1v1…3v3) | **1.00 every cell** | 0.82–0.98 |
| 1v2 / 2v3 | 0.95 / 1.00 | 0.67 / 0.68 |
| 1v3 (solo vs three) | 0.68 | 0.42 |
| Mean breach rate | **0.015** | 0.110 |
| Bad deaths | 0.0% | 0.0% |

**GATE: PASS.** The champion's small-team gap (it never trained below N=4) was
real and is closed. Deployment bridge documented in the design review: mocap →
sim units ×125, setpoints streamed at 30 Hz, referee thresholds 0.12/0.15 m.

## Convoy escort (moving protected target)
New env capability: per-world `target_speed_range`, terrain-aware waypoint
wander (urban cap 60 m, maze static), exact target-relative swept breach check,
target-frame allocator feasibility. **Finding: escort came free** — the
ego-centric obs generalize, so the static champion already scores 1.00 on a
10 m/s wandering target; the convoy fine-tune (20M, OOM-shortened, sufficient)
matches at 0.99 with attrition 15.3%→**10.7%**. Static regression: none (0.99).
Design-review catches that made this clean: breach-check target frame,
allocator t_breach (±1.2 s error could flip assignments), renderer
next-episode-terrain bug (pre-existing, now fixed).

## Replays
`aegis_convoy_hills_8v10.gif` (gold convoy trail through hills),
`aegis_convoy_urban_10v12.gif` (urban escort vs decoy raid).

## Ops notes
- Second OOM kill at n_max=20 (exit 137, 384 worlds, ~20 M in) — checkpointing
  + queue guards made it a non-event; use ≤288 worlds for future n_max=20 runs.
- Branch: DD-16 squash-merged to main (b42b97e); DD-17 carries night-2 work.
  **User: `git push origin main && git push -u origin DD-17` (pushes are
  policy-blocked for Claude).**

## Next
1. Real-hardware bridge script (cflib): mocap→obs encoder + setpoint streamer
   using the L=125/T=√2 table — first actual CrazyFlie flight test.
2. n_max=20 memory profiling (fix the OOM class properly).
3. Population round vs cf_lab at lab scale; convoy adversarial round.
