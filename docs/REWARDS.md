# AEGIS Reward Structure — Complete Reference

> Written 2026-08-19 for DD-16. Explains every reward term in `envs/aegis_env.py`
> (the constants live in `AegisConfig`), the verified exploit analysis behind the
> current numbers, and the changes being trialed in the overnight experiments.
> Companion to `docs/AEGIS_PLAN.md` §3.4.

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
   interceptor.

## The current table (what the deployed model was trained with)

| # | Term | Value | When it fires | Who gets it |
|---|---|---|---|---|
| 1a | Assigned-closing shaping | **+0.03 /m** | every step, toward MY assigned attacker | that defender |
| 1b | Guard shaping | **+0.01 /m** toward ring station, **+0.005/step** while within 150 m of target | unassigned (surplus) defenders | that guard |
| 2 | Intercept (kamikaze) | **+30** | my swept-collision with any attacker (assigned or opportunistic) | the interceptor only; stream ends (`done=True`) |
| 3 | Bad death | **−15** | hill / ground / out-of-bounds / ceiling | the dying defender; stream ends |
| 4a | Breach, owned | **−25** | my assigned attacker reaches the target | the owner |
| 4b | Breach, others | **−5** | any breach | every other living defender |
| 4c | Breach, unowned | **−10** | an unowned attacker breaches | all living defenders |
| 5 | Team win | **+10** | all attackers eliminated, zero breaches | living defenders only |
| 6 | Time | **−0.01/step** | every step (max −7.2/episode at 720 steps) | every living defender |
| 7 | Safety shaping | **−0.02 × (1 − dist/70m)** per step | inside a hill danger ring or within 25 m of bounds | that defender |

Scale intuition: a clean 300 m approach + kill ≈ **+9 shaping +30 intercept ≈ +39**.
A full passive episode ≈ **−7 time − breach penalties**. Episodes end per-agent at
death (with proper `terminal_observation`), and the world ends at
all-attackers-resolved / all-defenders-dead / 720-step timeout.

Special cases:
- **Exogenous culls** (the self-healing training augmentation that randomly kills a
  defender mid-episode) pay **exactly 0** — the augmentation must not teach fear.
- Breach despawns the attacker and **the episode continues** (partial defense is
  measurable; no first-leak world-kill like the old system).
- A dead defender's stream emits zero-obs/zero-reward filler until the world resets
  — it can never be paid or fined again (this matters below).

## The exploit audit (2026-08-17, verified against code with γ=0.995 math)

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

## Changes being trialed overnight (2026-08-19)

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

### Not changed tonight (recorded for later)
`analysis/reward_balance.py` still models the OLD 48 Hz ±200 reward table and
must be ported to AEGIS constants and made a blocking pytest before any future
reward retune (finding #6). Adversarial co-training (Phase 3) is the next big
lift after tonight's results.

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
