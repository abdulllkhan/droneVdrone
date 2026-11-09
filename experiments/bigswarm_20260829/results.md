# Big-Swarm Scale-Up (2026-08-29, DD-17) — enabled by the OOM fix

First run to exploit the lifted ≤256-world cap: a **24v24 generalist trained at
448 worlds** (both beyond the prior 20v20 / 256-384 ceiling). Champion:
`models/aegis_big24_stage3_realism_best/best_model.zip`. Gate win 0.98, RAM flat
~12 GB throughout (the .assign OOM fix, commit 29a874d, is what made it possible).

## Trained range (15v15–24v24, mixed terrain, coordinated raids)
**100% win, zero leakers on every cell/strategy.** Terrain deaths 5–12%
(saturation/decoy in obstacle worlds are the hardest).

## Generalizes UP with ZERO extra training (the headline)
| Cell | Win | Leakers | Bad-death |
|---|---|---|---|
| 30v30 | **100%** | 0.00 | 13% |
| 40v40 | **100%** | 0.00 | 14% |
| 50v40 (90 drones) | **100%** | 0.00 | 16% |
| **20v40 (outnumbered 2:1)** | **100%** | 0.00 | 11% |

A policy trained only to 24v24 defends 40v40 and 50v40 perfectly, and — because
reusable interceptors re-engage — **20 defenders beat 40 attackers with zero
leakers**. Size-invariance now proven from 1v1 (cf_lab) through 50v40 across the
whole model family; the ego-centric obs + allocator carry it.

## Significance
The OOM fix wasn't just reliability — it raised the trainable ceiling. Bigger
training swarms (24v24) produce the same perfect generalization the 20v20
champion did, now with more headroom for future work (deeper population loops,
30v30+ native training). Bad-death rises with swarm density in obstacle worlds
(16% at 50v40); a survival-bonus shaping pass is the lever if that matters.

## Artifacts
Champion `aegis_big24`; replay `aegis_big24_24v24.gif` (a 48-drone battle). No
OOM at 448 worlds / n_max=24 — the fix holds at the new scale.

## Erratum (2026-10-12)

- The trained-range evaluation (15v15–24v24) used the realism setting, which places
  five hills; urban and maze terrain were not evaluated here.
- The scale evaluation (30v30–50v40, 20v40) ran without `--realism`, so it had no
  obstacles (`num_obstacles` defaults to 0). The 11–16% bad-death rates in that table
  are ground, ceiling or boundary crashes, not obstacle collisions, and they do not show
  attrition rising with density in obstacle worlds. `aegis_split` lost no defenders in
  the same open-field cells (scorecard, 2026-08-29).
