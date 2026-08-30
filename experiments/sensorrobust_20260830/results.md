# Is Multi-Target Sensor Fragility Trainable? — Verdict (2026-08-30, DD-17)

The vark stress sweep found split defense collapses under degraded sensing
(K=3: 74%@150m, 19%@75m). Is that a training gap or fundamental? Fine-tuned
aegis_vark with sensor range sampled 80–350m + variable K (~44M steps across
two OOM-shortened runs under Gemma contention), then re-ran the sensor stress.

## Verdict: PARTIALLY trainable, with a hard PHYSICAL FLOOR (targets 180 m apart)
| K=3 sensor | vark | sensor-trained | Δ | verdict |
|---|---|---|---|---|
| 350 m (baseline) | 95% | 94% | — | unaffected |
| **150 m** | 74% | **80%** | **+6** | partially trainable (plateaus ~80%) |
| **75 m** | 19% | 22% | +3 | **fundamental floor** |

K=1: **100%** (no regression at any point).

## The physical rule (the real finding)
Multi-target sensor robustness is trainable **only down to roughly
sensor-range ≈ target-separation**, and hits a hard floor below it:
- **150 m sensing** (targets 180 m apart, near the threshold): training recovers
  +6 pt (74→80%) by learning better distributed coverage, then plateaus — a
  *partial* training gap.
- **75 m sensing** (well below the 180 m spacing): training moves it only +3 pt
  (19→22%). A defender guarding one target *physically cannot detect* an attacker
  180 m away at a different target — the blind spots are geometric, not a policy
  deficiency. No amount of training overcomes it.

## Deployment implication (now precise)
Size sensors against target spacing: **split defense needs detection range
comparable to or greater than the distance between protected points.** Below
that, coverage falls toward chance regardless of the policy — a hard design
constraint, not a tuning target. Single-target defense has no such issue
(defenders cluster; short range suffices).

## Decision
Adopt `aegis_sensrobust` as a marginally-more-sensor-robust multi-target variant
(80% vs 74% at 150 m, no K=1 regression), but the deliverable is the FINDING: the
sensor-vs-spacing floor. aegis_vark remains fine for good-sensing deployments.
