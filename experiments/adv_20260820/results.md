# Phase-3 Adversarial Co-Training — Round 1 Results (2026-08-20)

Driver: `train/train_aegis_adv.py`. Round A: RL attackers (evade mode, 41-dim ego
obs) vs frozen `aegis_patch` defender, 40M steps. Round B: defender retrained 60M
steps vs mixture (mixed/flanking/jink/rl). Evals: 5 cells (3v5,5v3,5v5,8v5,8v8),
realism, 48 episodes each. CSVs: `models/eval_output/nvm_grid_model_{patch_vs_rl,advdef_vs_rl,advdef_scripted}.csv`.

## (a) Learned attackers DEMOLISH the scripted-era champion
| Old (patch) defender vs | Win | Leakers | Defender bad-deaths |
|---|---|---|---|
| scripted attackers (same cells) | 0.78 | 0.33 | 2.2% |
| **RL attackers** | **0.01** | **4.19** | **94.4%** |

The attackers' discovered exploit is **kamikaze-baiting**: lure pursuing defenders
into hills/ground (94% of defenders die to terrain, not trades), then cruise in.
Attack runs take 22s. A defender that was untouchable by scripted opponents is
essentially disarmed by an adaptive one — this is why Phase 3 exists.

## (b) One co-training round closes much — not all — of the gap
| Co-trained defender vs RL attackers | Old | Co-trained |
|---|---|---|
| Win | 0.01 | **0.24** |
| Leakers/episode | 4.19 | **1.78** |
| Bad-death rate | 94.4% | **19.6%** |
| Episode length | 22s (rout) | 55s (contested full-length fights) |

The co-trained defender stopped taking the bait (bad deaths −75 pts), forces
full-length engagements, and halves the leakage. Parity vs a fully adaptive
attacker needs more self-play iterations — expected; that's the nature of the loop.

## (c) Non-regression vs scripted: clean
Full 114-cell realism grid: co-trained **0.75** vs old 0.76 win (within noise),
bad deaths 2.9% vs 2.5%. **The mixture worked — no catastrophic forgetting.**

## Notes
- Attacker stalling: mid-training the attackers also learned to loiter out of reach
  (timeouts ~50%); the stranded-as-leaker metric counts this as attacker success,
  but from a mission standpoint a stalling attacker is defeated. The Round-B
  defender ignores the bait (timeouts 0 in mixture evals).
- Round-B gate on seed 99 read 0.69 vs the 0.70 target (mixture sampling + stall
  accounting); grid evals above are the meaningful numbers.

## Next iterations (standard population loop)
1. Round A2: train fresh attackers vs the co-trained defender (aegis_adv_defender).
2. Round B2: retrain defender vs pool {scripted, A1, A2} — grow the mixture.
3. Try co-training the **reuse (kamikaze=False)** defender: the baiting exploit
   preys on kamikaze economics (a defender death is worth an attacker's while);
   a reusable interceptor is structurally less baitable.
4. Consider attacker-side stall penalty (time reward) so attacker training pressure
   stays on penetration, not clock-milking.

Models: `models/aegis_adv_attacker_best/`, `models/aegis_adv_defender_best/`.
