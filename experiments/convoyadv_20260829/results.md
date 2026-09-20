# Convoy Escort — Adversarial Verification (2026-08-29, DD-17)

**Question:** the moving-target escort came free vs *scripted* attackers — can a
*learned* attacker exploit the wandering convoy (as one did to the small-team
cf_lab policy)?

## Verdict: escort is adversarially ROBUST out of the box (NO GAP)
40M-step RL attackers vs the frozen `aegis_convoy` escort on 10 m/s moving
targets:

| Measure | Value |
|---|---|
| ROUND A gate (attacker's home turf, 256 eps) | true_breach **0.077** — below the 0.08 gap threshold |
| Verdict | **no gap → round B skipped by design** |
| Grid eval (frozen escort vs the learned attacker, 5v5/8v8/10v12) | win **0.89**, breach **0.016** |

The escort's large-team redundancy (N∈[4,20], unlike cf_lab's 1–4) leaves no
consistent seam for an adaptive attacker — the same reason the large
terrain-generalist was robust first-pass while the small lab policy was not.
The attacker peaked at ~0.08 true breach mid-training but could not push past it;
its residual "success" is mostly stall-timeouts, not real penetration.

## Contrast (the emerging rule)
| Policy | Team size | Adaptive-attacker gap? |
|---|---|---|
| aegis_cf_lab | 1–4 (small) | YES (0.19) → needed hardening |
| aegis_largeP (terrain) | 4–20 | no |
| **aegis_convoy (escort)** | 4–20 | **no** |

**Rule of thumb: small-team policies need an adversarial hardening pass before
deployment; large-team policies inherit robustness structurally.**

## Artifacts
Attacker: `models/aegis_adv_attacker_convoyadv_best/` (the tester, kept for future
pools). New replay: `aegis_convoyadv_hills_8v10.gif`. No new defender — the
existing `aegis_convoy` is confirmed deployment-grade for moving targets.

## Ops
192 worlds, n_max=20 evade mode: no OOM (the ≤256 mitigation holds). ~9 min for
the 40M round.
