# CrazyFlie Policy — Adversarial Verification (2026-08-28, DD-17)

**Purpose:** stress-test the deployment policy `aegis_cf_lab` against a *learning*
attacker before any real flight — scripted evals (100% win) can hide adaptive
weaknesses.

## Finding: the small-team lab policy HAD a real adversarial gap (now closed)
Unlike the large terrain-generalist (robust on the first adversarial pass), the
CrazyFlie small-team policy was exploitable:

| Stage | Metric |
|---|---|
| ROUND A gate (attacker vs frozen cf_lab, evade-mode, its home turf) | **breach 0.19** (defender win 0.73) — well past the 0.08 gap threshold |
| ROUND B gate (co-trained defender vs mixture) | **win 0.96, breach 0.020** — gap closed in one round |

## Cross-harness eval (eval_nvm_grid, small-team cells, minimal obs)
| Matchup | Win | Breach | Bad-death |
|---|---|---|---|
| frozen cf_lab vs adaptive RL | 0.966 | 0.022 | 5.0% |
| **hardened (aegis_adv_defender_cf_lab) vs adaptive RL** | **0.991** | **0.005** | 0.6% |
| hardened vs scripted (regression, 18 cells incl. 1v3) | 0.956 | 0.016 | 0.0% |

(The grid harness reads gentler than the evade-mode gate because it runs the
attacker as a frozen model on fixed N≥M-friendly cells rather than as the live
learner on its own training distribution — but the ordering is unambiguous:
hardened beats frozen 4× on breach and 8× on bad-death vs the same attacker,
with no scripted regression.)

## Deployment recommendation
Ship **`models/aegis_adv_defender_cf_lab_best/best_model.zip`** (not the raw
cf_lab) as the CrazyFlie candidate: same unit mapping (L=125, T=√2), same minimal
obs (obs_zero_pip, no opportunistic channel), now adversarially hardened. The
`deploy/cf_bridge.py` encoder is model-agnostic — swap the --model path.

## Ops note (recurring)
Third exit-137 kill this project, at 512 worlds / n_max=4 with only 22 GB RAM
used → transient unified-memory (VRAM+RAM shared pool) spike, NOT steady
exhaustion. Mitigation in place: ≤256 worlds. Root-cause profiling remains a
heartbeat todo.

## Erratum (2026-10-12)

"18 cells incl. 1v3" is 9 cells (1v1, 2v1, 2v2, 3v1, 3v2, 3v3, 1v2, 2v3, 1v3) × 2 raid
types (direct, jink), 200 episodes each.
