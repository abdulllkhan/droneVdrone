# Distributed Spawns Close the Split-vs-Direct Gap (2026-08-30, DD-17)

**Hypothesis (from the prior tick):** the residual K=3 split-vs-direct gap is
STRUCTURAL — defenders spawn clustered at the target centroid and can't
pre-position across 3 spread points against an all-at-once rush. Fix: distribute
defender spawns round-robin across the K targets (env change, commit faa82d3).

## CONFIRMED — the gap is essentially closed
The full arc on K=3 direct simultaneous raids:

| Model / env | 10v10 | 15v15 | 20v20 |
|---|---|---|---|
| aegis_split (clustered, no direct training) | 81% | 78% | 74% |
| aegis_splitdirect (clustered, direct-trained) | 83% | 85% | 83% |
| **any split model + DISTRIBUTED spawns** | **90–93%** | **95%** | **95%** |

Saturation raids stay 100%; single-target stays 100% (no regression).
Time-to-first-intercept dropped from ~2.1s to ~1.1s — pre-positioned defenders
engage attackers before they reach the spread targets.

## The precise finding: it's the ENV, not the retraining
Both models in this eval ran in the new distributed-spawn env, and BOTH score
93–95% on direct raids — including aegis_splitdirect, which was trained with
clustered spawns (it scored only 83–85% in the old clustered-spawn env last
tick). So **the pre-positioning is an environment property that helps any split
policy**, not a learned behavior that needs retraining. The structural diagnosis
was exactly right.

## Decision
Adopt **`aegis_splitspawn`** as the split champion — trained consistent with the
now-default distributed-spawn geometry, K=3 direct 90–95%, saturation 100%,
single-target 100%. The split-vs-direct weakness, the last uneven cell in the
n=100 scorecard, is closed (74–81% → 93–95%). Distributed spawns are now the
default for multi-target (K=1 unchanged/bit-identical).

## Artifacts
Champion `aegis_splitspawn`; replay `aegis_splitspawn_15v15.gif`. 256 worlds
(Gemma probe coexisting), no OOM.

## Erratum (2026-10-12)

- In the distributed-spawn evaluation (n=100), `aegis_splitspawn` won 90% / 95% / 95%
  of direct raids at 10v10 / 15v15 / 20v20 and `aegis_splitdirect` 93% / 95% / 93%. The
  table's 20v20 entry for "any split model" should read 93–95% (not 95%), and "both
  score 93–95%" and "74–81% → 93–95%" should read 90–95%.
- Not reported above: with distributed spawns, defender attrition for both models rose
  from 3–5% to 9–12% (bad-death rates in `eval_splitspawn.log` and
  `eval_splitdirect.log`, against `../splitdirect_20260830/eval_splitdirect.log`).
