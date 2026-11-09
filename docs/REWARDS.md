# AEGIS Reward Structure — Complete Reference

> Written 2026-08-19 for DD-16; reward tables re-checked against the code on
> 2026-10-12. Explains every reward term in `envs/aegis_env.py` (the constants live in
> `AegisConfig`), the attacker reward used for adversarial training, the verified
> exploit analysis behind the numbers, the changes trialed in the 2026-08-19
> overnight experiments, and the settings used by the champion models. Companion to
> `docs/AEGIS_PLAN.md` §3.4, which holds the original design values.

## Design philosophy

Three rules drive everything:

1. **Per-agent, owner-based credit.** A defender is paid/penalized for *its own*
   assignment (the attacker the allocator gave it), not for team-wide outcomes it
   can't influence. This is what fixed the old system, where uniform ±200 team
   rewards made every agent's return depend mostly on teammates → no gradient.
2. **Potential-based shaping only.** Distance shaping is a *telescoping* series:
   each step pays `coeff × (d_prev − d_now)`. Sum over any path = net distance
   closed. Oscillating back and forth nets zero, so shaping cannot be farmed.
   On any reassignment the anchor `d_prev` resets to the new target's distance —
   so reassignment (including a teammate killing your target) pays exactly 0.
3. **Kamikaze = good death, everything else = bad death.** Dying on an intercept
   is the job. Dying to a hill, the ground, or the arena boundary wastes an
   interceptor. In reusable-interceptor mode (`kamikaze=False`, used by every
   champion model) the interceptor survives its intercept, so every death is a bad
   death.

## Current reward terms

Values are the `AegisConfig` defaults in `envs/aegis_env.py`. The champion models
(every champion since `aegis_largeP`, 2026-08-27) are reusable interceptors
(`kamikaze=False`) trained with two overrides, shown in the "Champions" column.

| # | Term | Field | Default | Champions | When it fires | Who gets it |
|---|---|---|---|---|---|---|
| 1a | Assigned-closing shaping | `r_close_coeff` | **+0.03 /m** | same | every step, metres closed toward MY assigned attacker | that defender |
| 1b | Guard shaping | `r_guard_close_coeff`, `r_guard_hold` | **+0.01 /m** toward ring station, **+0.005/step** within 150 m of the nearest protected point | same | unassigned (surplus) defenders | that guard |
| 2 | Intercept | `r_intercept` | **+30** | same | swept contact (closest approach < 3 m during the step) with any attacker, assigned or opportunistic | the interceptor only |
| 3 | Bad death | `r_bad_death` | **−30** | **−60** | within 5 m of an obstacle surface (below its top), below 5 m altitude, above 200 m, or outside the ±500 m arena | the dying defender; stream ends |
| 4a | Breach, owned | `r_breach_owner` | **−25** | same | my assigned attacker reaches its protected point (swept, < 5 m) | each owner, including one that died the same step (unless culled) |
| 4b | Breach, others | `r_breach_other` | **−5** | same | an owned attacker breaches | every other living defender |
| 4c | Breach, unowned | `r_breach_unowned` | **−5** | same | an attacker with no owner breaches | all living defenders |
| 5 | Team win | `r_team_win` | **+10** | same | all attackers eliminated, zero breaches | living defenders only |
| 6 | Time | `r_time` | **−0.01/step** | same | every step (max −7.2/episode at 720 steps) | every living defender |
| 7 | Safety shaping | `r_safety_coeff` | **−0.05 × (h + b)** per step | **−0.10 × (h + b)** | near an obstacle or a boundary (terms below) | that defender |

Safety-shaping terms, each clipped to [0, 1]:

- *h* = max over obstacles of (1 − *s* / 40 m), where *s* is the horizontal distance
  to the obstacle surface and the defender is below the obstacle's top. For a hill of
  the default 50 m radius this ring starts 90 m from the centre, the value adopted on
  2026-08-19.
  The same surface distance drives terrain kills (*s* < 5 m), for hills and for
  urban/maze boxes alike.
- *b* = max over the arena walls (±500 m), the 5 m floor and the 200 m ceiling of
  (1 − *d* / 25 m), where *d* is the distance to that boundary.

The 40 m ramp and the 5 m kill distance are fixed in `envs/aegis_env.py`; the
`obstacle_danger_radius` and `obstacle_collision_radius` fields of `AegisConfig` are
no longer read.

Scale intuition: a clean 300 m approach + kill ≈ **+9 shaping +30 intercept ≈ +39**.
A full passive episode ≈ **−7 time − breach penalties**. Episodes end per-agent at
death (with proper `terminal_observation`), and the world ends at
all-attackers-resolved / all-defenders-dead / 720-step timeout.

Special cases:
- **Exogenous culls** (the self-healing training augmentation that randomly kills a
  defender mid-episode) pay **exactly 0**, including no same-step breach fine — the
  augmentation must not teach fear.
- Breach despawns the attacker and **the episode continues** (partial defense is
  measurable; no first-leak world-kill like the old system).
- A dead defender's stream emits zero-obs/zero-reward filler until the world resets
  — it can never be paid or fined again (this matters below).
- **Reusable mode** (`kamikaze=False`): an intercept pays +30 without ending the
  stream, and the defender cannot intercept again for 24 steps (2 s rearm).
- **Re-anchoring:** any reassignment, and any move of a guard's ring station (for
  example after a teammate dies), resets the shaping anchor, so neither pays a spike.

## Attacker reward (adversarial training only)

In evade mode (`mode="evade"`, used by `train/train_aegis_adv.py`) the learning agents
are the attackers. Their table mirrors the defender philosophy:

| Term | Field | Value |
|---|---|---|
| Closing shaping toward its protected point | `ar_close_coeff` | +0.03 /m |
| Breach | `ar_breach` | +30 |
| Intercepted | `ar_intercepted` | −20 |
| Terrain, ground or out of bounds | `ar_bad_death` | −15 |
| Time | `ar_time` | −0.01/step |
| Obstacle proximity (no boundary term) | `ar_safety_coeff` | −0.02 × (1 − *s* / 40 m) per step |
| Still airborne at the 720-step timeout (stalling) | `ar_stall` | −10 |

Losing all defenders does not end an evade-mode episode: surviving attackers fly on to
breach.

## The exploit audit (2026-08-17, verified against code with γ=0.995 math)

The audit was run against the pre-patch table (bad death −15, unowned breach −10,
safety −0.02 within 70 m of a hill's centre; `docs/AEGIS_PLAN.md` §3.4).

An adversarial reviewer computed episode returns for every known exploit archetype.
**Healthy margins** (intended behavior wins by the stated amount):

| Archetype | Return | Margin vs honest interceptor (+38.6) |
|---|---|---|
| Eternal chaser (closes, never commits) | −27.2 | 66 |
| Passive hoverer | −18.9 to −28.9 | 58–68 |
| Guard-camper ignoring promotion | +10.2 | 28 |
| Kill-stealer (chases the opportunistic channel over assignment) | +14.9 vs +29.9 staying | 15 |
| Shaping oscillator / realloc-launderer | ≈ 0 | — (telescoping kills it) |

**The one hole found (major): the dead-owner breach escape.**
If my assigned attacker becomes uncatchable (jink escape, tail-chase) and will
breach in <8.5 s, the math favored **diving into the ground on purpose**:
eat −15 once and end my stream, dodging the −25 owner penalty plus any later
−10 unowned-breach spam. Stay-alive ≈ −25.9 (discounted) vs suicide ≈ −14.5.
The breach loop only pays *living* defenders, and at N<M there may be no spare
defender to inherit the orphan — the −25 literally evaporates. The observed 1.9–2.9%
"bad deaths" in realism evals may partly be this exploit, not clumsiness.

Two smaller issues: **safety shaping is outgunned 14:1** by chase shaping inside
hill rings (so the hazard signal barely registers), and two bookkeeping leaks
(a culled defender could be fined −25 if its attacker breaches the same step;
a guard whose ring station teleports after a teammate death gets an un-anchored
±2 shaping spike).

**The strategic finding:** the reward table is no longer what limits performance.
Realism win 0.76 is pure grid composition — N>M cells are at 1.00, parity at 0.99,
and N<M is pinned at 0.24 because **kamikaze makes exchange ratio ≤ 1 by
definition** (N defenders can never remove more than N attackers). Reward tuning
can recover ~2 points of exchange ratio at most. Breaking N<M requires a
*mechanical* change, not a reward change.

## Changes trialed overnight (2026-08-19)

Patch A and Patch B were adopted as the `AegisConfig` defaults after this trial
(results below); the reusable-interceptor mode became the basis of every later
champion.

### Patch A — reward rebalance (train A/B vs current)

| Constant | Old | New | Why |
|---|---|---|---|
| `r_bad_death` | −15 | **−30** | closes the dead-owner suicide escape (dive −28.6 vs stay −18.1 → staying wins by ~10) |
| `r_breach_unowned` | −10 | **−5** | shrinks the penalty stack that made suicide attractive; halves uncontrollable advantage noise at N<M |
| `r_safety_coeff` | −0.02 | **−0.05** | hazard signal was 14:1 outgunned; chase still dominates avoidance 5.5:1, so no timidity regression |
| `obstacle_danger_radius` | 70 m | **90 m** | earlier warning ≈ 2.6× integrated hazard signal |

Deliberately **unchanged**: `r_intercept` (+30 already dwarfs per-step terms),
`r_close_coeff` (raising it linearly scales the passive free-income and
launder-rate exploits). Expected effect: bad deaths 2.5% → ~1%, N<M exchange
ratio 0.98 → ~1.00, parity 0.99 → ~1.00. Realism win moves ≤1 pt (capped by N<M).

(Since the 2026-08-27 terrain update the danger ring is the fixed 40 m surface
distance described above, which reproduces the 90 m ring for hills of the default
50 m radius.)

### Patch B — code fixes (both arms get these)
- Culled defenders excluded from same-step breach fines (culls pay exactly 0 again).
- Guards re-anchored when their ring station moves (removes the ±2 spike at
  self-healing events — exactly the transitions stages 2–3 exist to teach).

### Experiment — reusable interceptor (`kamikaze=False` + rearm cooldown)
The only real N<M unlock. The defender survives its intercept (models a gun/net
drone rather than a ramming drone) with a **2 s rearm cooldown** during which it
cannot intercept. Reward consequences, re-audited: +30 per kill *without* ending
the stream (a survivable kill-stream strictly strengthens chase dominance);
kamikaze-avoider archetype vanishes; team-win +10 becomes reachable by
interceptors (it was inert for them before — only guards could outlive a win).
Exchange ratio becomes unbounded; 3v10 stops being structurally lost.

### Not changed in the trial (recorded for later)
`analysis/reward_balance.py` still models the OLD 48 Hz ±200 reward table and
must be ported to AEGIS constants and made a blocking pytest before any future
reward retune (finding #6). Adversarial co-training (Phase 3) is the next big
lift after tonight's results.

*Status:* the port is still open (`analysis/reward_balance.py` is unchanged since
DD-13); the later champion retune (below) was validated by train-and-evaluate A/B
instead. Phase 3 adversarial co-training followed on 2026-08-20/21.

## Results (actuals — full details in `experiments/overnight_20260819/results.md`)

- **Patch A/B**: training dynamics much cleaner (stage gates 0.84/0.85 vs 0.71/0.81,
  in-training bad deaths 4–6× lower) but final grid outcomes identical (win 0.76,
  bad 2.5%) — the audit's ≤1pt prediction was right. Adopted as the recommended
  default: it buys soundness (suicide exploit closed) and faster convergence.
- **kamikaze=False (reusable interceptor)**: the predicted unlock, and then some —
  N<M win 0.24 → **0.68**, exchange ratio 0.98 → **8.97**, and 3v5/5v8/7v10/10v20
  at **100% win** (3v8: 81%, was structurally 0%). Cost: bad deaths 2.5% → 19%
  because survivors fly ~5× more risk-miles and −30 amortizes over multiple kills;
  next lever is stronger safety shaping or a per-kill survival bonus.
- Stress: self-healing survives a 2-defender kill at 0.99 win; **sensor range is
  the binding constraint** (win 0.92 at 150 m, 0.70 at 75 m); speed parity and 2 s
  allocation staleness cost nothing.
- Curriculum-skip control: visibly worse (win 0.72, bad 14.7%, XR 0.71) — keep the
  curriculum.

## Champion settings (2026-08-27 onward)

The attrition cost of reusable interceptors was addressed by pricing it into the
reward rather than by a survival bonus: from `aegis_largeP` (2026-08-27) on, every
champion is trained with `r_bad_death = −60` and `r_safety_coeff = −0.10`. In the
clean urban comparison this cut defender attrition from 22.7% to 13.3% at an
unchanged 1.00 win rate (`experiments/overnight_20260827/results.md`, Round 2).
Stronger safety shaping (−0.15) and a physical obstacle speed brake were tested
later and not adopted: each lowered attrition in a five-hill 10v10/20v20 test but
raised it in an open-field 40v40/50v40 test. That test contains no obstacles, so the
brake never engaged and the extra losses are ground, ceiling and boundary crashes
(`experiments/safety_20260829/results.md`, `experiments/brake_20260829/results.md`,
both with errata). The safety run also set `obstacle_danger_radius = 110`, which had
no effect because the field is no longer read.
