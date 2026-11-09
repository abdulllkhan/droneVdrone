# Adversarial Population Loop (r2/r3) + Reuse Track — Results (2026-08-21)

Queue: 03:52–04:40 + champion evals. CSVs: `models/eval_output/nvm_grid_model_{b2_vs_a1,b2_vs_a2,b2_vs_a3,b2_scripted,reuseadv_vs_ar,reuseadv_nlm}.csv`.

## Population-loop trajectory (kamikaze track) — CONVERGED in 2 rounds
| Round | Fresh attackers found (A-gate breach) | Defender retrain (B-gate win vs mixture) |
|---|---|---|
| 1 (2026-08-20) | **0.75** — kamikaze-baiting into terrain | 0.69 |
| 2 | **0.46** — mostly stalling (82% timeouts) | **0.80** |
| 3 | **0.094** → "no gap" verdict, loop stops | (unnecessary by design) |

A full 40M-step attacker training run can no longer meaningfully exploit the B2
defender. **Champion: `models/aegis_adv_defender_r2_best/best_model.zip`.**

## Champion B2 vs every attacker generation (5 cells, realism)
| Opponent | Win | Leakers | Defender bad-deaths |
|---|---|---|---|
| A1 (baiter) — old defender scored 0.01/4.19/94% | **0.55** | 0.70 | 5.5% |
| A2 (staller) | 0.73 | 0.38 | 4.3% |
| A3 (converged gen) | 0.67 | 0.50 | 5.1% |
| Scripted full grid (non-regression) | **0.75** | 0.83 | 2.3% |

The baiting exploit is defused (bad-deaths vs A1: 94% → 5.5%), robustness holds
across generations it never trained against directly, and two additional
co-training rounds cost **zero** scripted performance (0.75 vs 0.76).

## Reuse (kamikaze=False) track — best overall model after ONE round
Fresh attackers found a 0.30 gap vs the frozen reuse defender — not baiting per
se, but exploiting its old terrain sloppiness (0.42 defender crash rate while
chasing). One co-training round:

- **0.93 win / 0.08 leakers vs its adaptive RL attacker** (kamikaze track needed
  two rounds to reach 0.55–0.73 vs generations)
- **N<M superpower fully retained: 3v5, 5v8, 10v20 all 100% win, zero leakers**
- Known cost unchanged: ~20% terrain attrition (survivors fly more risk-miles)

**`models/aegis_adv_defender_reuse_best/best_model.zip` is now the strongest
model in the project** — adversarially hardened AND unbounded exchange ratio.
The reusable-interceptor hypothesis held: with nothing to gain from trading, the
baiting exploit class never re-emerged; the residual attrition is a shaping
problem (next lever: per-kill survival bonus or stronger safety term), not a
strategic one.

## Refinements noted for future rounds
1. Attacker best-checkpoint criterion counts stranded-at-timeout as success;
   switch to true-breach-rate so stall-heavy checkpoints aren't selected.
2. B2-vs-A1 (0.55) suggests one more kamikaze-track round vs a baiter-enriched
   pool would help — only if the kamikaze platform stays relevant.
3. Reuse track: run its own population loop (A2r/B2r) + attrition shaping.
