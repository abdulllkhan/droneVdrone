# AEGIS — Capabilities Report

> Authoritative project summary. All numbers come from fixed-seed evaluations of a
> deterministic policy in simulation. Sections 1, 2 and 4 come from the 2026-08-29
> scorecard (`experiments/scorecard_20260829/`: n=100 episodes per cell and raid type,
> n=60 for the open-field scale grid), with additional CrazyFlie numbers from
> `experiments/cfadv_20260828/` (n=200); section 3 from
> `experiments/varktargets_20260830/` (n=60); the stress and sensor figures from
> `experiments/varkstress_20260830/` and `experiments/sensorrobust_20260830/` (n=48).
> Design and full history: `docs/AEGIS_PLAN.md`. Reward reference: `docs/REWARDS.md`.
> Corrected 2026-10-12 after re-checking every figure against the evaluation scripts
> and logs.

> **Status:** feature-complete. A regression sweep on 2026-08-31 re-loaded and
> re-evaluated all 7 champion models under the final environment (its output was not
> saved in the repository); 22/22 tests passed at that point. Every capability has
> been built, characterized, and checked against learned attackers. Since then, DD-18
> (2026-09-20) wired the CrazyFlie motion-capture sources, cflib command link and
> safety layer, tested over a real UDP socket (9 tests in `tests/test_cf_deploy.py`).
> The remaining roadmap is hardware-gated (a real CrazyFlie flight).

## What AEGIS is
An RL-trained interceptor swarm that defends a target from incoming attackers.
**One shared policy** controls every defender via ego-centric observations
(56-dim, size-invariant); an in-environment allocator assigns each defender an
attacker. Coordination is **emergent** — defenders that follow their assigned
attackers automatically split, cover, and self-heal. Trained on GPU (Newton/Warp
physics) with many worlds simulated in parallel: 2,048 in the first (1v1) curriculum
stage and up to 512 in the large-swarm fine-tunes.

**How to read the numbers.** *Win*: every attacker is destroyed and none reaches a
protected point (zero-leaker win). *Leakers*: attackers per episode that reach a
protected point or are still airborne when the episode ends. *Attrition*: share of
defenders lost to terrain, ground, ceiling or arena exit (bad deaths); the champions
are reusable interceptors, so intercepts themselves cost no defenders. *Breach*:
leakers as a share of attackers; adversarial gates report *true* breach, which
excludes attackers that only stall until the timeout. *Realism setting*: five hills,
wind (0.5 N steady, gusts to 1.5 N), observation noise (σ = 0.02 on normalized
observations) and a 300 m detection range. *Open field*: no obstacles, no wind,
350 m detection range.

## 1. Single-target defense — size-invariant
Realism setting, reusable interceptors, one protected point, n=100 per cell and raid
type (direct, saturation):

| Cell | Win | Leakers | Attrition |
|---|---|---|---|
| 1v1 | 98% | 0.02 | 2% |
| 3v3 – 7v5 | **100%** | 0.00 | 2–4% |
| 10v10 – 20v20 | **100%** | 0.00 | 3–4% |

One policy, 1v1 through 20v20, zero leakers from 3v3 up. The scorecard's
single-target and scale rows (sections 1–2) were measured with `aegis_split`:
`aegis_big24` fine-tuned for three protected points with 6–20 defenders against 3–20
attackers, evaluated here with one point. Urban and maze terrain were evaluated
separately on an earlier champion (`aegis_large`: 1.00 win from 5v5 to 20v20, n=32,
with about 23% defender attrition in those obstacle-dense worlds;
`experiments/overnight_20260827/results.md`).

## 2. Scale generalization — beyond the training sizes
Open field, same `aegis_split` checkpoint (trained with at most 20 defenders and 20
attackers; its parent `aegis_big24` with at most 24v24), n=60 per cell and raid type
(direct, flanking):

| Cell | Win | Leakers | Attrition |
|---|---|---|---|
| 30v30 / 40v40 / 50v40 | **100%** | 0.00 | 0% |
| 20v40 (outnumbered 2:1) | **100%** | 0.00 | 0% |
| 10v20 (outnumbered 2:1) | **100%** | 0.00 | 0% |

30v30 and larger, and 20v40, lie outside the training range and received **zero
additional training**; 10v20 is inside it. Because reusable interceptors re-engage,
**N defenders beat 2N attackers with zero leakers**. The ego-centric obs + allocator
carry the policy past its training range. `aegis_big24` alone also won 100% with 0.00
leakers at 30v30–50v40 and 20v40 (n=20), but lost 10–16% of its defenders in those
open-field battles (see Known limitations).

## 3. Multi-target split defense — one policy for 1–4 points
`aegis_vark` defends **any number of protected points (1–4)**, chosen per episode.
Realism setting, 10v10 / 15v15 / 20v20, n=60 per cell and raid type. The points are
placed 126–180 m from the centre of the defended area, roughly evenly spaced in angle
(with three points, typically 230–300 m apart):

| Points | Overall win | Direct raids | Saturation raids | Attrition |
|---|---|---|---|---|
| 1 | **100%** | 100% | 100% | 3% |
| 2 | 98% | 97% | 100% | 11% |
| 3 | 96% | 92% | 100% | 11% |
| 4 | 95% | 91% | 100% | 11% |

Split defense is implemented with **zero allocator/interception change** —
coverage emerges from defenders following assigned attackers. One policy covers the
whole range: K=1 stays 100% (matching the single-target champions) and K=3 matches
the dedicated split model, so no per-scenario model selection is needed. The cost is
attrition: about 11% of defenders are lost with two or more points, against 3% with
one. Saturation raids are 100% at every count; direct simultaneous raids degrade
gracefully (100→91%) as the swarm covers more spread points. Two enablers raised the
hard direct-raid case at K=3 from 74–81% to 90–95%: direct-weighted training and
**distributed defender spawns** (pre-positioning across the targets — an env property
that helps any split policy). The same invariance that carries the swarm over drone
count carries it over target count.

## 4. CrazyFlie deployment — validated in simulation, not yet flown
The `aegis_adv_defender_cf_lab` model + `deploy/cf_bridge.py` map the policy onto
a real 8×8 m lab via pure unit mapping (length scale L=125, time scale T=√2 →
0.34 m/s, 3 m/s², 8.49 Hz control; referee thresholds 0.12/0.15 m). Minimal sensor
suite (the predicted-intercept-point (PIP) hint and the opportunistic
nearest-attacker channel are dropped). Small teams, scripted direct and jinking raids:

| Cell | Lab-like setting, n=200: win (attrition) | Realism stress test, n=100: win (attrition) |
|---|---|---|
| 1v1 | **100%** (0%) | 88–89% (11–12%) |
| 2v1 / 3v2 | **100%** (0%) | 96–98% (10–16%) |
| 2v2 / 3v3 | **100%** (0%) | 92–95% (12–18%) |
| 3v1 | **100%** (0%) | not evaluated |
| 1v2 (outnumbered) | 94–95% (0%) | 77–83% (13–17%) |
| 2v3 (outnumbered) | 99–100% (0%) | not evaluated |
| 1v3 (one against three) | 66–68% (0%) | not evaluated |

The *lab-like setting* has no obstacles or wind, as indoors and as in the policy's
training (`experiments/cfadv_20260828/`). The *realism stress test* is the
scorecard's: five hills, wind and 300 m detection, none of which the lab policy saw in
training. Defender losses appear only when hills and wind are added, and wins at N≥M
drop from 100% to 88–98%. Outnumbered cells are the weak spot in both settings.

The observation encoder is checked against the simulator by a test (maximum absolute
difference below 2e-5 after a 40-step 3v3 rollout), which rules out encoder/simulator
drift, a common sim-to-real failure. The bridge flies the full loop headlessly in a dry
run and, since DD-18, end to end over a real UDP socket.

## 5. Adversarial robustness
The system was stress-tested against *learning* attackers, not just scripts:
- **Pattern so far:** the first two defenders tested (the kamikaze champion and the
  reusable-interceptor model, 3–8 defenders) both had gaps and were hardened by
  co-training. Of the champions warm-started from the hardened reusable line, the
  large-team ones (4–20 defenders) showed no gap on the first pass, while the
  small-team CrazyFlie policy (1–4 defenders) did and needed one more hardening round.
  Redundancy in larger teams is a plausible explanation, not a tested one.
- cf_lab's real 0.19-breach gap was found and **closed** in one co-training round
  (→ the hardened deployment model). The large terrain generalist, the convoy escort
  and the **multi-target** `aegis_vark` (K=3) all showed **no gap** on the first
  adversarial pass (true breach 0.012, 0.077 and 0.005 against a 0.08 threshold).
  Against vark, the learned attacker's breach rate in the grid evaluation
  (0.009–0.013) was comparable to that of scripted direct raids on the same cells
  (0.006–0.013). Hardening appears to transfer through warm-starts, over target count
  as well as team size.
- The population loop converged (attacker exploit sizes 0.75 → 0.46 → 0.094).

## Model selection guide

| Scenario | Use | Notes |
|---|---|---|
| **General default** | `aegis_vark` | 1–4 targets; trained with 6–20 defenders vs 3–20 attackers, evaluated 8v8–20v20 |
| Pure single-target scale (up to 50v40) | `aegis_big24` / `aegis_oomval` | both won 100% at 50v40 in open field (n=20–24); open-field losses 13–16% (big24) and 5–7% (oomval) |
| CrazyFlie / lab deployment | `aegis_adv_defender_cf_lab` | unit-mapped, minimal obs, adversarially hardened; use with `deploy/cf_bridge.py` |
| Moving protected target (convoy) | `aegis_convoy` | escort a target wandering ≤10 m/s |

`aegis_vark` is the recommended default for most uses. The specialized models
exist for their named cases (extreme single-target scale, hardware deployment,
moving targets) where they were each validated.

## Known limitations

- **Defender attrition depends on the policy and the setting; it has not been shown to
  scale with density.** With obstacles, `aegis_big24` lost 5–12% of its defenders at
  15v15–24v24 (realism setting), the multi-target models about 11%, and the terrain
  generalist `aegis_large` about 23% in hills, urban and maze worlds. In open field,
  losses at 30v30–50v40 ranged from 0% (`aegis_split`) through 5–7% (`aegis_oomval`)
  to 13–16% (`aegis_big24`); with no obstacles present these are ground, ceiling or
  boundary crashes. Two variants of big24 — stronger safety shaping and a physical
  speed brake near obstacles — cut losses in the five-hill 10v10/20v20 test (9.1% →
  5.9% and 4.7%) but raised them in the open-field 40v40/50v40 test (14.6% → 25.4%
  and 34.5%), where the brake never engages; neither was adopted. Wins stayed near
  100% throughout (reusable interceptors absorb the losses).
- **Split defense vs direct simultaneous raids** is the hardest case: 91–97% for
  2–4 points (was 74–81% at K=3 before direct-weighted training and distributed
  defender spawns). Saturation raids and single-target stay 100%.
- **Multi-target defense needs detection range comparable to the spacing between
  protected points.** Single-target tolerates cheap sensing (98–100% even at 75 m).
  With three points (typically 230–300 m apart), `aegis_vark` wins 95% at the
  standard 300 m detection range, 74% at 150 m and 19% at 75 m, and the fragility is
  only *partly* trainable: training with sensor range randomized over 80–350 m
  (`aegis_sensrobust`) recovers 150 m to 80% and 75 m only to 22%. Detection is shared
  but sensors are not: when no defender is within range of the attackers heading for a
  distant point, they are seen too late. **Design rule: size detection range to the
  distance between protected points.** Speed disadvantage and mid-fight kills, by
  contrast, degrade gracefully (tested at K=1 and K=3).
- **No hardware flight yet** — the CrazyFlie path is validated in sim and dry-run;
  mocap sources (UDP/Lighthouse/Qualisys), the cflib command path and the
  safety layer are wired and socket-tested (`docs/FIRST_FLIGHT.md`), pending a
  real drone + rig.
- **Convoy + split are mutually exclusive** (moving multi-target is out of scope).

## Champions (all in `models/`, browse via `python -m eval.launcher`)

| Model | Role |
|---|---|
| `aegis_vark` | **multi-target primary** — defends 1–4 points, one policy (95–100%) |
| `aegis_big24` / `aegis_oomval` | single-target scale reference (to 50v40); `aegis_big24` is the 24v24-trained scale generalist |
| `aegis_adv_defender_cf_lab` | CrazyFlie deployment (hardened, minimal obs) |
| `aegis_convoy` | moving-target escort |
| `aegis_splitspawn` | K=3 split-defense model with distributed spawns; superseded as the multi-target default by `aegis_vark` |
| `aegis_sensrobust` | `aegis_vark` fine-tuned with randomized sensor range; marginally more tolerant of short sensing (see Known limitations) |

`aegis_split`, the checkpoint behind the section 1–2 scorecard numbers, is the
three-point fine-tune of `aegis_big24` from which `aegis_splitspawn` and `aegis_vark`
descend. Checkpoints are not tracked in git (`models/**/*.zip` is ignored).

Results dashboard: `python -m eval.dashboard` → `models/eval_output/dashboard.html`.
