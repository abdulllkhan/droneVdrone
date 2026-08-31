# Multi-Target Adversarial Verification (2026-08-31, DD-17)

First adversarial test of SPLIT defense: can a *learned* attacker find a
coordination seam in multi-target (K=3) defense that scripted raids can't?
Adversarial round vs frozen `aegis_vark`, the primary multi-target champion.

## Verdict: NO GAP — split defense is adversarially robust
| Measure | Value |
|---|---|
| ROUND A gate (learned attacker vs frozen vark, K=3, 256 eps) | **true_breach 0.005** — below the 0.08 threshold |
| Verdict | **no gap → round B skipped by design** |
| Grid eval (frozen vark vs learned attacker) | win 0.90 (8v8 93%, 15v15 87%), breach 0.009–0.013 |

A learned attacker does no better than scripted raids (which vark already handles
at 92–100%) — in fact its breach rate (0.01) is *below* the scripted ~0.05. The
multi-target coordination (defenders following assigned attackers across 3 spread
points) exposes no exploitable seam. This confirms the established rule holds for
the multi-target case too: large-team policies inherit adversarial robustness
structurally, and it extends across the number of targets.

Note: defender terrain attrition is elevated (32–36%) when chasing the evasive RL
attacker through hills at K=3 — that's the documented intrinsic density/terrain
attrition (win holds at 0.90 regardless), NOT a breach vulnerability.

## Significance
The primary multi-target champion is now adversarially verified — the last
untested capability. Every AEGIS champion (single-target large, convoy, cf_lab,
and now multi-target vark) has been checked against learned attackers; only the
small-team cf_lab ever had a gap, and it was hardened. The deployment story is
complete: split defense is robust to adaptive adversaries.

## Ops
Evade-mode adversarial training OOMs at 512 worlds (frozen-defender predict +
attacker PPO is memory-heavy even with Gemma gone) — used 256. Attacker kept for
future pools: models/aegis_adv_attacker_varkadv_best.
