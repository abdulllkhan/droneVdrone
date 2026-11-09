# Overnight AEGIS Experiments — Results (2026-08-19, 02:49–04:01)

All 14 queue steps completed, exit 0. Raw CSVs: `models/eval_output/nvm_grid_*.csv`.
Models: `models/aegis_{patch,reuse,scratch3}_stage*`. Rationale: `docs/REWARDS.md`.

## q1 — Scale extrapolation (trained on ≤8v8, zero extra training)
| Cell | Win | Leakers | Notes |
|---|---|---|---|
| 20v15 | 100% | 0 | first kill 1.7s |
| 30v20 | 100% | 0 | |
| **40v30** | **100%** | 0 | 5× the trained defender count |
| 10v20 | 0% | 10.0 | exact kamikaze identity (10 kills then overrun) |

**Verdict: size-invariance confirmed.** The ego-centric obs + allocator generalize
essentially without limit upward; N<M remains capped by kamikaze math (see q3).

## q2 — Reward patch A/B (bad_death −30, unowned −5, safety −0.05, danger 90m)
Training dynamics clearly better: stage-2 gate win 0.84 vs 0.71, stage-3 gate 0.85
vs 0.81, in-training bad deaths 4–6× lower (0.03–0.04 vs 0.10–0.18). **Final grid
outcomes: statistically identical** (win 0.76/0.76, bad 2.5%/2.5%) — the original's
breach-rate checkpoint selection had already picked its cleanest model. The patch's
value is soundness (closes the dead-owner suicide exploit) + faster convergence, not
end metrics — exactly the ≤1pt movement the audit predicted. **Adopt as default.**

## q3 — Reusable interceptor (kamikaze=False + 2s rearm) — HEADLINE
| Metric (realism grid) | Kamikaze | Reusable |
|---|---|---|
| Overall win | 0.76 | **0.89** |
| N<M win | 0.24 | **0.68** |
| Exchange ratio (N<M) | 0.98 (ceiling 1.0) | **8.97** |
| Bad-death rate | 2.5% | 19.1% (trade-off, see below) |

Focus cells (plain): **3v5, 5v8, 7v10, 10v20 all 100% win / 0 leakers; 3v8 → 81%**
(was 0% — structurally impossible — under kamikaze).
Trade-off: defenders now survive kills, fly ~5× more sorties, and accumulate ~19%
terrain losses (a −30 death amortizes over multiple +30 kills, so risk economics
shifted). Next iteration if attrition matters: raise safety shaping further or add a
small survival bonus per kill. If the real platform is a gun/net interceptor, this
mode should be the primary model.

## q4 — Stress battery (kamikaze best model)
| Stressor | Win | Verdict |
|---|---|---|
| Kill 2 defenders at t=10s (realism, culled) | 0.99 | self-healing holds; +1pt breach only |
| Sensor 300→150 m | 0.92 | mild degradation |
| Sensor 300→75 m | 0.70 | **sensing is the binding constraint** — key sim-to-real datum |
| Attacker speed ×1.2 (60 = parity) | 1.00 | no collapse |
| Allocation every 2 s + culls | 1.00 | allocator cadence is not critical |

## q5 — Curriculum necessity (stage-3-only from scratch, 60M)
scratch3: win 0.72, bad 14.7%, XR 0.71 vs curriculum: 0.76, 2.5%, 0.92.
Mid-training it was far worse (train win 0.17, bad 72% at 30M). **The 4-stage
curriculum earns its keep** — cold-starting full realism converges slower and to a
visibly sloppier policy (6× the terrain losses).

## Recommended next steps
1. Make the reward patch the default constants in `AegisConfig` (done in code? no —
   constants unchanged; flip when adopting).
2. Decide the platform story: kamikaze (FPV ramming) vs reusable (gun/net). If both,
   train a mixed fleet or condition the policy on a mode flag.
3. Phase-3 adversarial co-training (RL attackers vs frozen defender, mixture pool).
4. For CrazyFlie sim-to-real: budget sensors for ≥150 m-equivalent detection scaled
   to lab arena; obs-knockout ablations still pending.
