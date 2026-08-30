# Split-vs-Direct Gap — Targeted Fine-Tune (2026-08-30, DD-17)

**Gap:** K=3 split defense handled saturation raids at 100% but DIRECT
simultaneous raids at only 74-81% (the one uneven cell in the n=100 scorecard).
Fine-tuned aegis_split with direct raids weighted 3× → `aegis_splitdirect`.

## Result: gap HALVED, no regression → adopt as split champion
| K=3 cell | direct: old → new | saturation |
|---|---|---|
| 10v10 | 81% → 83% (+2) | 100% (unchanged) |
| 15v15 | 78% → **85%** (+7) | 100% |
| 20v20 | 74% → **83%** (+9) | 100% |
| Single-target K=1 (regression) | **100%** (perfect, no loss) | 100% |

Biggest gains on the hardest cells (+9pt at 20v20), leakers down (0.30→0.20),
zero regression on saturation raids or single-target. `aegis_splitdirect` is a
strict improvement over `aegis_split` — **adopted as the split champion.**

## Honest finding: the residual gap is STRUCTURAL, not training-exposure
Direct-heavy training closed ~half the gap but plateaued at 83-85%, not 100%.
The remaining ~15pt is a coordination limit: defenders spawn clustered on a ring
around the target CENTROID, so against an all-at-once rush on 3 spread points they
can't pre-position fast enough — some attackers reach outer targets before a
defender arrives. Training exposure teaches better commitment but can't overcome
the spawn geometry. **The lever to close the rest is structural: distribute
defender spawns across the K targets** (a spawn change), not more training.
Logged as the next multi-target improvement.

## Verdict
The one measured weakness is materially reduced (direct-raid win +6pt average, to
83-85%) with no cost elsewhere. Split defense is now uniformly strong except the
hardest direct-rush case, whose residual is understood and has a clear next fix.
