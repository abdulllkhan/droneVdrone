# aegis_vark Under Stress — Verification Loop (2026-08-30, DD-17)

Does the multi-target champion hold up under deployment stressors (only ever
tested on single-target models)? Swept sensor range, attacker speed, and kill
injection at K=1 and K=3. **The loop caught a real multi-target-specific limit.**

## Stressor × target-count (8v8+15v15, direct+saturation, n=48)
| Stressor | K=1 win | K=3 win | Δ (K3−K1) |
|---|---|---|---|
| baseline | 100% | 95% | −5 |
| sensor 150 m | 100% | **74%** | **−26** |
| sensor 75 m | 98% | **19%** | **−79** |
| attacker 1.2× (speed parity) | 99% | 93% | −6 |
| kill 2 defenders (self-heal) | 100% | 98% | −2 |

## KEY FINDING: multi-target defense is sensor-hungry
Speed disadvantage and mid-fight kills degrade *gracefully* at K=3 (self-healing
and reusable interceptors absorb them, like single-target). But **sensor
degradation amplifies dramatically with target count**: single-target tolerates
cheap sensing (98–100% even at 75 m), while split defense collapses (74% at
150 m, 19% at 75 m).

**Why:** with one target, defenders cluster and collectively cover a small area,
so even short-range sensing catches incoming attackers. With 3 targets 180 m
apart, defenders spread thin — global shared detection ("any defender sees for
the team") fails when *no* defender is near the far target, so attackers on
distant targets go undetected until too late.

## Deployment implication
- **Single/few-target defense tolerates cheap sensors** (the CrazyFlie minimal-
  sensor build stands — 150 m-equivalent was already the single-target floor).
- **Multi-target split defense needs good sensor coverage** (≥300 m-equivalent;
  150 m already halves it). In a lab, split defense wants mocap (full detection)
  or strong onboard sensing — not the minimal suite. This is a real constraint,
  now documented, that the verification pass surfaced before any deployment.
- Not sensor-related: speed parity and kill injection are fine at all K.
