# Safety-Shaping A/B (2026-08-29, DD-17) — VERDICT: keep big24, do NOT adopt

**Question:** does stronger safety shaping (r_safety_coeff −0.10→−0.15,
obstacle_danger_radius 90→110 m) cut the 16% terrain attrition at high density?
Retrained from big24 at 448 worlds; measured real terrain collisions (bad_death;
the danger radius only scales the shaping reward, not the kill geometry, so the
comparison is fair).

## Result: it's a DENSITY-DEPENDENT TRADEOFF, not a win
| Range | big24 bad-death | safe bad-death | Δ | win |
|---|---|---|---|---|
| Dense obstacle 10v10–20v20 (trained density) | 9.1% | **5.9%** | **−3.2 pt** | 1.00 both |
| Scale 40v40–50v40 (zero-shot up) | 14.6% | **25.4%** | **+10.9 pt** | 1.00 both |

Stronger avoidance helps in the density it was trained at, but **generalizes
worse to crowded arenas**: at 50v40 the larger 110 m danger rings make 40+ drones
over-avoid and pile up, roughly DOUBLING collisions. Win stays 100% throughout
(reusable interceptors absorb the losses), so this is purely an efficiency
tradeoff.

## Decision
**Keep `aegis_big24` as champion.** Do not adopt the stronger constants as
default — they'd trade the project's headline zero-shot up-scaling (24v24→50v40)
for lower attrition only at the trained density. The stronger-safety model
(`aegis_safe`) is kept for the specific case of operating AT ~10–20v20 density in
heavy terrain, where it's a clean 3-pt improvement.

## The real lesson
Safety shaping tuned at one swarm density does not transfer across densities.
Attrition is better treated as density-matched (train the safety knob for the
deployment density) than globally maximized. Genuine terrain-attrition reduction
at all scales likely needs a density-invariant mechanism (e.g. obstacle-relative
speed limiting, or curriculum over danger_radius), not a fixed stronger coefficient.

## Erratum (2026-10-12)

- `obstacle_danger_radius` has not been read by `envs/aegis_env.py` since commit
  b42b97e (the danger ramp is fixed at 40 m from the obstacle surface), so setting it to
  110 m had no effect; the only effective change was `r_safety_coeff` −0.10 → −0.15.
- The scale rows (40v40–50v40) ran without `--realism`, so they contained no obstacles.
  The rise from 14.6% to 25.4% is in ground, ceiling or boundary crashes, so the
  explanation that larger danger rings make crowded swarms over-avoid and pile up is not
  supported. The two rows differ in obstacles (five hills vs none) as well as in
  density.
