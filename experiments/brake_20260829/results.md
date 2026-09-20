# Physical Speed-Brake A/B (2026-08-29, DD-17) — VERDICT: keep big24, do NOT adopt

**Hypothesis:** a physical obstacle-proximity speed brake (dynamics, not reward)
would be density-INVARIANT and cut terrain attrition at all scales — succeeding
where the reward-shaping A/B failed.

## Result: it fails the SAME way, even harder
| Range | big24 bad-death | brake bad-death | Δbad | Δwin |
|---|---|---|---|---|
| Dense 10v10–20v20 (trained density) | 9.1% | **4.7%** | **−4.4 pt** | −2.3 pt |
| Scale 40v40–50v40 (zero-shot up) | 14.6% | **34.5%** | **+19.9 pt** | −1.2 pt |

The brake helps at low/trained density but **DOUBLES attrition at 50v40** — worse
than the reward-shaping approach's +11 pt. Mechanism: at extreme density the
arena is packed with obstacles, so braked defenders are perpetually slowed,
spend LONGER near hazards, and collide MORE. Caution backfires when crowded.

## The confirmed pattern (two independent mechanisms, same failure)
| Attrition mechanism | 20v20 | 50v40 |
|---|---|---|
| Stronger safety shaping (reward) | −3 pt | **+11 pt** |
| Physical speed brake (dynamics) | −4 pt | **+20 pt** |

**Terrain attrition at high swarm density is NOT a caution-tuning problem.**
Making defenders more careful near obstacles — by reward OR by physics — reduces
collisions only when there's room, and backfires in crowded arenas. Since
win stays ~100% at every density (reusable interceptors absorb the losses),
the density-scaling attrition is an intrinsic cost, not a bug to shape away.

## Decision
**Keep `aegis_big24` as champion.** The brake (`obstacle_speed_limit`, default
off, code retained) is useful ONLY for low-density terrain ops (~≤20v20), like
the safety model. If high-density attrition ever truly matters, the lever is
STRUCTURAL (fewer/smaller obstacles, explicit inter-drone spacing, or a
formation-aware allocator), not caution.

## Ops note
448-world AEGIS OOMs (exit 137) when the concurrent non-AEGIS process
`dev/probe_gemma_robust.py` shares the box — retried clean at 256 worlds. Use
≤256 worlds when sharing the GPU.
