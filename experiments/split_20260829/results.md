# Multi-Target Split Defense (2026-08-29, DD-17) — NEW CAPABILITY, works

Can the swarm protect **3 separate points at once**? Implemented with zero
allocator/interception change (num_targets=1 stays bit-identical): attackers
split across K spread targets, each breaches its own, and because defenders are
assigned to *attackers* (now heading to different points) they fan out to cover
the whole field. Fine-tuned `aegis_split` from big24 at K=3.

## It works — and beats the zero-shot baseline
| Model @ K=3 (3 points, 10v10–20v20) | Win | Leakers | Bad-death |
|---|---|---|---|
| **aegis_split (trained on K=3)** | **0.90** | **0.11** | **4.5%** |
| big24 (zero-shot at K=3) | 0.87 | 0.15 | 8.4% |

Two findings: (1) **big24 already generalizes to split defense zero-shot** (0.87)
— the ego-centric obs + allocator transfer to multiple targets with no training,
another payoff of the size/structure-invariant design. (2) Fine-tuning still
helps meaningfully: +3 pt win, lower leakers, and **half the attrition** (better
coverage discipline learned from the harder task). Gate: 0.89 win / 10.0
intercepts per episode (vs ~7 single-target — more spread threats to catch).

## No regression: strict superset
`aegis_split` at K=1 (single target): **win 1.00, zero leakers** — it lost
nothing on the original task while gaining split defense. It is a superset of
big24, not a specialization.

## Verdict
Split defense is a real, working capability. `aegis_split` handles both 1 and 3
protected points. The architecture extends to K targets essentially for free —
the emergent-coordination design (defenders follow assigned attackers) is what
makes multi-point defense fall out without new machinery.

## Artifacts
Champion `aegis_split` (1-and-3-target); replay `aegis_split3_15v15.gif` (three
gold stars, swarm covering all). 256 worlds (coexisted with the Gemma probe), no OOM.
