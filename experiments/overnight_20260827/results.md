# Overnight 2026-08-27 — Large Swarms, Labyrinths, Coordinated Attackers

Queue ran 02:55–03:34 (one OOM restart at 03:02, resumed from checkpoint, worlds
512→384). New champion: **`models/aegis_large_stage3_realism_best/best_model.zip`**
— a kamikaze=False terrain-generalist trained N∈[4,20] vs M∈[1,20] on randomized
hills/urban/maze against direct/flanking/jink/**saturation/waves/decoy** raids.

## Training (q1, 90M steps warm-started from the reuse champion)
Gate: **win 0.96** (256 episodes, target 0.60), 6.2 intercepts/episode, first
kill 2.6 s. Terrain deaths fell 0.24 → 0.05 during training as it learned the
labyrinths.

## Adversarial check (q2): NO GAP on the first pass
Fresh RL attackers (40M steps, terrain worlds): **true breach 0.012** (threshold
0.08) → "no gap" verdict, Round B unnecessary. The adversarial hardening from the
reuse-champion lineage SURVIVED the terrain curriculum — robustness now transfers
through warm-starts instead of needing a fresh co-training loop per model.
(Stall-timeouts 28% are no longer rewarded: attacker checkpoints select on true
penetration since this session.)

## Terrain grids (q3: 5v5→20v20 × direct/saturation/decoy, full realism)
| Terrain | Win | Leakers/ep | Notes |
|---|---|---|---|
| Hills | 0.98 | 0.03 | 20v20 saturation raid: 100% |
| **Urban** (tower grid) | **1.00** | 0.00 | all cells perfect |
| **Maze** (3-ring labyrinth) | **1.00** | 0.00 | all cells perfect |

Known cost: ~23% terrain attrition among defenders in obstacle worlds (open-field
losses are 0.2–1.7%) — the reuse-line shaping issue, unchanged priority.

## Scale (q3, open field)
**40v30, 50v40, 20v30, and 10v20 (outnumbered 2:1): ALL 100% win, zero leakers**,
defender losses ≤0.6% at the biggest sizes. 90 drones in one battle, first kill
in ~2.5 s.

## New replays (dashboard gallery)
`aegis_reuse_8v8_saturation_maze.gif` · `aegis_reuse_10v15_decoy_urban.gif` ·
`aegis_reuse_20v20_waves.gif` · `aegis_reuse_50v40_flanking_none.gif`

## Next
1. Terrain-attrition shaping (per-kill survival bonus) — the one soft spot left.
2. Attacker route-planning (A*/corridor waypoints) so scripted raids use maze
   gates instead of crude repulsion (fewer wall self-kills → harder raids).
3. Sim-to-real: sensor budget study on the large model; obs-knockout ablations.

---

# Round 2 (03:43–03:58) — attrition pricing, routed raids, sensor knockouts

## Attrition A/B (r_bad_death −60, r_safety −0.10) → new champion `aegis_largeP`
Clean comparison (urban, unaffected by the routing change): attrition **22.7% →
13.3%** at identical 1.00 win. Training-time trajectory: 0.24 → **0.047** on
mixed-terrain evals vs routed raids at win 0.96. The maze grid A/B is
confounded (base was measured before gate-routing shipped; largeP faced the
harder routed raids and still won 0.98). Verdict: adopt largeP as champion —
`models/aegis_largeP_stage3_realism_best/best_model.zip`.

## Routed raids
Scripted attackers now thread maze gates (see `aegis_maze_routed_10v12.gif`) —
deeper penetrations, more contested inner-ring fights; the defense holds
0.98–1.00 regardless.

## Sensor knockouts (base large model; which obs the CrazyFlie build needs)
| Zeroed block | Win | Bad-death | Read |
|---|---|---|---|
| none (control) | 0.98 | 16.0% | — |
| PIP hint | 1.00 | 15.0% | **not load-bearing — skip on hardware** |
| opportunistic channel | 0.99 | 18.2% | near-free — optional sensor |
| teammate slots | 0.98 | **26.0%** | wins hold, attrition +10pts — teammate sensing is a survivability aid |
| assignment meta | 0.98 | 5.9% (t1st +0.8s) | policy flies conservative without roles — slower but safer |

**Minimal viable sensor suite: own state + assigned-target track + terrain +
boundary.** Wins are insensitive to everything else; extras trade attrition vs
speed.
