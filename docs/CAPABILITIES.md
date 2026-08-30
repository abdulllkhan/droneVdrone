# AEGIS — Capabilities Report

> Authoritative project summary. Numbers below are from a rigorous **n=100**
> (n=60 for the scale grid) fixed-seed evaluation of the champions
> (`experiments/scorecard_20260829/`, 2026-08-29). Design & full history:
> `docs/AEGIS_PLAN.md`. Reward reference: `docs/REWARDS.md`.

## What AEGIS is
An RL-trained interceptor swarm that defends a target from incoming attackers.
**One shared policy** controls every defender via ego-centric observations
(56-dim, size-invariant); an in-environment allocator assigns each defender an
attacker. Coordination is **emergent** — defenders that follow their assigned
attackers automatically split, cover, and self-heal. Trained on GPU (Newton/Warp
physics) at up to 512 parallel worlds.

## 1. Single-target defense — size-invariant, essentially perfect
Full realism (hills/urban/maze, wind, sensor noise), reusable interceptors:

| Cell | Win | Leakers | Attrition |
|---|---|---|---|
| 1v1 | 98% | 0.02 | 2% |
| 3v3 – 7v5 | **100%** | 0.00 | 2–4% |
| 10v10 – 20v20 | **100%** | 0.00 | 3–4% |

One policy, 1v1 through 20v20, zero leakers from 3v3 up.

## 2. Scale generalization — trained ≤24v24, holds to 90 drones
Open field, **zero additional training** beyond the 24v24 champion:

| Cell | Win | Leakers |
|---|---|---|
| 30v30 / 40v40 / 50v40 | **100%** | 0.00 |
| 20v40 (outnumbered 2:1) | **100%** | 0.00 |
| 10v20 (outnumbered 2:1) | **100%** | 0.00 |

Because reusable interceptors re-engage, **N defenders beat 2N attackers with
zero leakers**. The ego-centric obs + allocator carry the policy far past its
training range.

## 3. Multi-target split defense — protect 3 points at once (NEW)
K=3 separated targets, `aegis_splitspawn` (superset of the single-target champion):

| Cell | vs saturation raids | vs direct simultaneous raids |
|---|---|---|
| 10v10 | 100% | 90% |
| 15v15 | 100% | 95% |
| 20v20 | 100% | 95% |

Split defense works, implemented with **zero allocator/interception change** —
coverage emerges from defenders following assigned attackers. `aegis_splitspawn`
retains **100% at K=1** (strict superset). The direct-simultaneous case (all
attackers rush all 3 targets at once) was the last uneven cell — closed from
74–81% → **90–95%** in two steps: direct-weighted fine-tuning (+6pt) and then
**distributed defender spawns** (pre-position across the K targets, the decisive
structural fix that halved the remaining gap). Distributed spawns help *any*
split policy (an env property, not a learned behavior).

## 4. CrazyFlie deployment — sim-to-real ready
The `aegis_adv_defender_cf_lab` model + `deploy/cf_bridge.py` bring the policy to
a real 8×8 m lab via pure unit mapping (L=125, T=√2 → 0.34 m/s, 3 m/s², 8.49 Hz
control; referee thresholds 0.12/0.15 m). Minimal sensor suite (PIP + opportunistic
channels dropped). Small-team, n=100:

| Cell | Win | Attrition |
|---|---|---|
| 2v1 / 3v2 | 96–98% | 10–16% |
| 2v2 / 3v3 | 92–95% | 12–18% |
| 1v1 | 88–89% | 11–12% |
| 1v2 (outnumbered) | 77–83% | 13–17% |

The encoder is **test-proven bit-identical to the sim** (the #1 sim-to-real
killer, eliminated), and the bridge flies the full loop headlessly in a dry run.
Attrition is higher here (small teams, reusable-interceptor risk-miles); wins are
solid except the trivial 1v1 (a few points traded for adversarial robustness) and
outnumbered 1v2.

## 5. Adversarial robustness
The system was stress-tested against *learning* attackers, not just scripts:
- **Rule established:** small-team policies (cf_lab, 1–4) have exploitable seams
  and need an adversarial hardening pass; large-team policies (≥4-per-side)
  inherit robustness structurally (redundancy leaves no consistent seam).
- cf_lab's real 0.19-breach gap was found and **closed** in one co-training round
  (→ the hardened deployment model). The large/convoy champions showed **no gap**
  on the first adversarial pass. Hardening transfers through warm-starts.
- The population loop converged (attacker exploit sizes 0.75 → 0.46 → 0.094).

## Honest limitations
- **Density-scaling terrain attrition is intrinsic.** In obstacle-heavy worlds,
  defender terrain losses rise with swarm density (to ~16% at 50v40). TWO
  independent fixes (reward safety-shaping; a physical speed-brake) each helped at
  low density but *doubled* losses at high density — caution backfires in crowded
  arenas. Wins stay ~100% regardless (interceptors absorb it). A real fix would be
  structural (fewer obstacles / inter-drone spacing), not caution-tuning.
- **Split defense vs direct simultaneous raids** is the hardest case but now
  90–95% (was 74–81%; closed by direct-weighted training + distributed defender
  spawns). Saturation raids and single-target stay 100%.
- **No hardware flight yet** — the CrazyFlie path is validated in sim and dry-run;
  the cflib mocap loop is stubbed pending a real drone + motion-capture rig.
- **Convoy + split are mutually exclusive** (moving multi-target is out of scope).

## Champions (all in `models/`, browse via `python -m eval.launcher`)
| Model | Role |
|---|---|
| `aegis_splitspawn` | **primary** — single + 3-target defense (distributed-spawn, direct-hardened), 1v1→50v40 |
| `aegis_big24` | 24v24-trained scale generalist |
| `aegis_adv_defender_cf_lab` | CrazyFlie deployment (hardened, minimal obs) |
| `aegis_convoy` | moving-target escort |

Live dashboard: `python -m eval.dashboard` → `models/eval_output/dashboard.html`,
and the "AEGIS Flight Log" artifact.
