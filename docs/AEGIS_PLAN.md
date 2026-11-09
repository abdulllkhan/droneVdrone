# AEGIS — Variable-Size Swarm Defense: Diagnosis, Design, and Development Log

> Design document and development log for AEGIS, the allocator-based swarm-defense
> system (branches DD-15 to DD-18). Sections 0–3 are the original diagnosis and design,
> written 2026-07-12 from an 11-agent diagnosis/research/design workflow (~686k tokens)
> over the full codebase, training logs, git history, and MARL literature. Section 4
> covers the implementation and the results of the planned curriculum, section 5 is the
> dated log of all later work (newest first), and section 6 is the runbook. The current
> results summary is `docs/CAPABILITIES.md`; reward details are in `docs/REWARDS.md`.
>
> **Status:** feature-complete. A 2026-08-31 regression sweep re-loaded and re-evaluated
> all 7 champion models under the final environment (its output was not saved), and every
> capability has been checked against learned attackers. DD-17 and DD-18 are merged to
> `main`; DD-18 (2026-09-20) wired the CrazyFlie motion-capture sources, the cflib command
> link and a safety layer, tested over a real UDP socket. No hardware flight has taken
> place; the remaining roadmap is hardware-gated (first flight per
> `docs/FIRST_FLIGHT.md`). Several entries in sections 4 and 5 were corrected on
> 2026-10-12 after re-checking the evaluation scripts and logs; each correction is marked.

## 0. Goal

ONE reusable policy that handles **variable numbers of defenders (N) and attackers (M)**
— train once, run on any NvM (1v1 through 15v10) — with **self-healing** (when defenders
die, survivors re-cover the threats), where defenders actually **coordinate** (split
targets, don't clump) and intercept attackers before they reach the protected target.
Scale: Ukraine preset (1 km arena, 60 m/s FPV defenders, 50 m/s Shahed attackers,
kamikaze interception).

User constraints (from memory + history): keep rewards simple; kamikaze = good death vs
obstacle/ground/OOB = bad death; always CUDA; metrics-driven staged training with gates.

---

## 1. Why the current training fails (verified diagnosis)

All claims verified by reading code. File refs are to the state at commit `9b57f01`.

### FATAL (make large-N learning essentially impossible)

**F1 — No credit assignment or coordination mechanism.**
Team outcomes (+20 any-intercept, +100 win, −200 breach, −100 wipe) are paid uniformly
to every alive defender regardless of contribution (`envs/newton_swarm_env.py:1142-1157`).
The only individual signal is a +200 first-to-touch race — which *rewards* piling onto
the same nearest attacker. Target assignment / anti-clustering were explicitly removed
(`:948-950`). With `k_opponents=3 < M=5`, each defender can only see 3 of 5 attackers —
the leakers that breach are unobservable. At N=7 nearly all of an agent's return variance
is caused by teammates → PPO advantages flatten → the pile-on equilibrium is never
escaped. Observed: commit `eea36a8` "all 7 defenders pile on nearest attacker while 4
others breach, 3% intercept"; 2v2 metrics showed 0.96 m average pairwise distance.

**F2 — Global target teleport bug.**
`_randomize_positions` (`:1413-1423`) re-rolls the **single shared** `target_pos_wp`
altitude (20–60 m range) every time *any* of the 1024 worlds auto-resets (`:531-533`,
`:1405-1411`) — i.e. nearly every step at scale. The defended point that observations
reference, attackers pursue, and breach/intercept checks measure **teleports mid-episode
for every in-progress world**. The MDP is non-stationary in exactly the quantity the task
is defined by. Severity grows with `num_worlds`, so it selectively destroys large runs.

**F3 — Dead-agent reward shield + rollout pollution.**
An eliminated defender receives exactly 0 reward for the rest of the episode
(`:1116-1126`) — immune to the −200 breach and −100 wipe penalties. So kamikaze-ASAP
(+200 > +100 team win) or even a cheap −50 bad-death flee **dominates defending the
target**. Dead agents keep emitting zero-obs/zero-reward/done=False transitions until the
world ends (`:561`) — in 7v5 kamikaze, up to 6 of 7 agent streams are filler for hundreds
of steps in every 3.7M-transition rollout.

### MAJOR (each one independently cripples learning)

**M1 — Sparse rewards outside the discount horizon.** γ=0.99 at 48 Hz control = ~100-step
(~2 s) effective horizon; intercept/breach outcomes occur 150–500+ steps after the
decisions that matter; shaping is 0.0 from stage 3 on. The ±200 terminals are discounted
to 1–8% at decision time. Log signature: `approx_kl≈0.0025`, `clip_fraction≈0.013`,
`pg_loss≈-0.0002`, eval pinned at −140..−200 for 120M steps = a policy receiving no
gradient.

**M2 — Shaping punishes teammate success.** `_prev_min_att_dist` is never re-anchored
when a teammate kills my nearest attacker → I get `0.05 × (−300 m) = −15` spike vs +20
teammate credit (`:1132-1138`). The bootstrap stages actively teach that teammate
intercepts are bad.

**M3 — World-level termination + broken terminal handling.** ANY drone OOB ends the
episode for all agents with zero attributed reward (`:128-151`); no
`terminal_observation` / `TimeLimit.truncated` in infos → SB3 bootstraps V=0 at every
timeout (biased value function for long defensive episodes).

**M4 — Curriculum has no gates and warm-starts from noise.** Fixed 40M steps/stage, then
`best_model.zip` selected by EvalCallback **mean reward** on bimodal evals (best was
−34 ± 164 = one lucky batch). Stages 2–5 inherited random checkpoints from a stage 1 that
never converged. Also: with 7168 envs and `n_steps=512`, one rollout = 3.7M transitions →
40M steps/stage = **only ~11 PPO updates per stage**.

**M5 — Throughput ceiling.** Obs/rewards/attacker logic/wind are nested Python loops over
worlds×N×M every step; per-step Python list of `wp.vec3` (`:469-472`); wind does a full
GPU↔CPU roundtrip per substep (`:1537-1557`). GPU at ~2% utilization.

### MINOR (noise)

Distance-sorted K-nearest slots reshuffle entity↔slot mapping every step; teammate slots
have no validity flag (dead teammate ≡ teammate at origin); `obs[9]` leaks closing rate
to undetected attackers; intercept checked on positions at 48 Hz while closing speeds hit
110 m/s (2.3 m/check vs 3.0 m radius — tunneling); `per_drone_mass` randomization
computed but **never passed to the force kernel** (no-op); eval numbers persisted only in
PNG titles.

### What to KEEP

- CTDE parameter sharing (one policy, `n_envs = num_worlds × N` pseudo-envs) — correct,
  size-agnostic skeleton. There is no monolithic action-space explosion to fix.
- 3-dim velocity-command action + PD force tracking (not raw RPM).
- PPO over SAC (commit `f95a86a` fixed real Q-divergence; optimizer is stable, signal is not).
- Newton/Warp GPU physics with masked auto-reset.
- Simple-rewards philosophy; kamikaze vs bad-death vocabulary.
- `analysis/metrics.py` (clustering, angular coverage, breach rate diagnostics) and
  `analysis/reward_balance.py` (behavior-archetype exploit checks).
- Infra bug knowledge in DD-15 commits: eval_freq must be divided by n_envs;
  `reset_num_timesteps=True` per stage; one-time bad-death elimination.

---

## 2. Research summary (what works elsewhere)

- **Consensus recipe for variable N**: single parameter-shared policy executed per drone
  on ego-centric observations + entity-based obs (nearest-K or padding+mask) + curriculum
  / domain randomization over team sizes. (MAPPO/SMAC lineage, Deep Sets/attention
  encoders optional.)
- **Pursuit-evasion literature**: the most reliable architecture for multi-defender
  interception is **hierarchical** — a cheap classical allocator (greedy/Hungarian
  auction) assigns each defender exactly one attacker; a single shared RL policy is
  trained essentially as 1-v-1 pursuit of "my assigned target". Emergent pure-RL
  coordination is possible but far less reliable to train than assignment-based reward.
- **Self-healing** falls out of the allocator: recompute assignment on every death /
  elimination / new detection → survivors inherit orphaned threats within one step.
  Train with **exogenous defender culling** (random mid-episode kills with zero reward)
  so retasking-under-fire is in-distribution.
- **Credit assignment**: per-agent, owner-based rewards (my target, my breach, my
  intercept) beat uniform team rewards; keep team terms small.

## 3. The design — AEGIS (judge-selected synthesis)

Two layers; only one is learned. The **environment** is responsible for coordination
(allocation); the **policy** is only responsible for flying/intercepting well. Every
subsystem is verifiable without RL: the allocator is plain numpy testable in pytest; the
pursuit policy is a problem this codebase already solved at 1v1.

Rejected alternatives (recorded so nobody re-litigates): attention/entity-transformer
encoders (unnecessary once obs is fixed-size; new NaN surface — documented escalation
path if Phase-3 adversarial co-training needs it); clean-PyTorch MAPPO w/ centralized
critic (orphans all SB3 tooling; owner-based rewards already denoise advantages);
posthumous reward write-back (incompatible with SB3).

### 3.1 Allocator (`envs/allocator.py` — classical, zero learning)

- Cost matrix per world over (alive defenders × detected active attackers):
  `C[d,a] = t_go(d,a) + 100·[t_go > t_breach(a) − 1s] − urgency(a) − 0.5·[incumbent]`
  - `t_go(d,a)`: closed-form constant-velocity intercept time (quadratic solve, pursuer
    at max speed vs attacker straight-line); sentinel 60 s if infeasible.
  - `t_breach(a)`: attacker distance-to-target / attacker speed.
  - `urgency(a) = min(20/max(t_breach, 0.5), 10)`.
  - Hysteresis −0.5 for the incumbent owner; ownership tie-break by stable slot index.
- **Vectorized greedy global-min sweep** across worlds (loop min(N,M) rounds, each round
  one argmin per world). `scipy.optimize.linear_sum_assignment` as test reference; unit
  test asserts greedy within 5% of Hungarian on 10k random matrices.
- Surplus N>M: second greedy round doubles up 2:1 on the most urgent threats;
  still-unassigned defenders **guard** a 100 m ring at angle `2π·d/N_guards`.
- Recompute every 6 steps (0.5 s @ 12 Hz) AND immediately on: defender death, attacker
  elimination, breach, new detection. 0.5 s minimum dwell unless the current target died.
- **This is the self-healing mechanism**: a dead defender's attacker is orphaned →
  next allocation (≤1 step later, event-triggered) hands it to the best surviving
  defender. No learned component involved.

### 3.2 Observation — flat `Box(56,)`, ego-centric, fixed for ANY (N,M)

Normalizers: pos /500, ally rel pos /100 clipped, vel /60, rel vel /120, alt /200.

| idx | content |
|---|---|
| 0:3 | own velocity /60 |
| 3 | altitude /200 |
| 4:7 | (target − me) /500 |
| 7 | dist to target /500 |
| 8 | N_alive / N_spawned |
| 9 | M_active_detected / M_spawned |
| 10:26 | **ASSIGNED-ATTACKER block** (guard mode: ring station, zero velocity): rel pos /500 (3), rel vel /120 (3), dist /500, closing rate /120, LOS unit vector (3), attacker dist-to-target /500, (PIP − me)/500 (3) where PIP = att_pos + att_vel·t_go, detected flag. Undetected assigned target → zeros + flag 0 |
| 26:29 | assignment meta: role (1 interceptor / 0 guard), time-since-reassignment /5s clip, teammates sharing my attacker /4 |
| 29:37 | 2 nearest living teammates: rel pos /100 clip (3) + dist /100, ×2. **Empty-slot sentinel: pos=0 AND dist=1.0** |
| 37:45 | 2 nearest hills: rel center /200 clip (3) + shell-surface dist /200, ×2 |
| 45:48 | boundary: (bounds−\|x\|)/500, (bounds−\|y\|)/500, min(z−z_min, z_max−z)/200 |
| 48:56 | **nearest OTHER attacker** (not assigned to me; opportunistic channel): rel pos /500 (3), rel vel /120 (3), dist /500, valid flag. Config flag `obs_opportunistic`, default on |

### 3.3 Action, timing, network

- Action: `Box(-1,1,(3,))` desired velocity × 60 m/s, tracked by the existing PD force
  kernel (kp=50, kd=15, F_max=150 N).
- **Policy rate 12 Hz** (physics 240 Hz → 20 substeps). 60 s episode = **720 decisions**.
- **γ = 0.995** → ~16.7 s effective horizon (contains intercept/breach events).
  Escalation if stage-3 terminal-miss rate is high: 24 Hz + γ=0.9975.
- Network: stock SB3 PPO `MlpPolicy`, `net_arch=dict(pi=[256,256], vf=[256,256])`, tanh.
  No custom torch modules anywhere.

### 3.4 Reward (per-agent, owner-based; final numbers)

Good-episode return ≈ +10..+40, worst ≈ −50. ONE table for all stages — no stage-varying
shaping cliffs.

1. **Assigned-closing shaping** (potential-based, permanent): `+0.03 × (d_prev − d_now)`
   meters toward MY assigned attacker only (telescopes to +9 over 300 m). On ANY
   assignment change, `d_prev` **re-anchors** to the new target's current distance —
   reassignment is reward-neutral by construction (kills M2). Guard mode: same form
   toward ring station at +0.01/m, plus +0.005/step while within 150 m of target.
2. **Intercept (kamikaze, good death): +30** to the intercepting defender only — assigned
   or opportunistic. Emits `done=True` + `infos['terminal_observation']`. No teammate broadcast.
3. **Bad death** (hill/ground/OOB/ceiling): **−15** once, then done=True. Exogenous
   dropout-culls pay **0** (augmentation must not add penalty noise).
4. **Breach**: attacker despawns, **episode continues** (partial defense is measurable;
   no first-leak world-kill). **−25 to the owner** of that attacker, −5 to every other
   living defender; −10 to all if the leaker was unowned.
5. **Team win** (all attackers eliminated, zero breaches): +10 to living defenders; world
   done. All-resolved with ≥1 breach: no bonus, world done.
6. **Time**: −0.01 per step per living defender (max −7.2/episode). Guards too.
7. **Safety shaping**: −0.02 × (1 − dist/70) per step inside a hill's 70 m danger radius;
   same form within 25 m of arena/altitude bounds.

Lifecycle: `done=True` + `terminal_observation` at every death; `TimeLimit.truncated=True`
at 720-step timeout (fixes V=0 bootstrap bias); after death the stream emits a constant
dead-sentinel obs / reward 0 / done=False until the world resets (inert filler; log
effective-sample fraction, alert < 50%).

> **Current values.** The numbers above are the original design. The 2026-08-19 reward
> audit changed bad death to −30, the unowned-breach penalty to −5 and the safety
> coefficient to −0.05 with a 90 m danger ring; these are now the `AegisConfig`
> defaults. Every champion since `aegis_largeP` (2026-08-27) is a reusable interceptor
> (`kamikaze=False`, 2 s rearm) trained with bad death −60 and safety −0.10, and the
> obstacle terms were re-based on the distance to the obstacle surface when urban and
> maze terrain were added. Full table: `docs/REWARDS.md`.

### 3.5 Mandatory env fixes (fork `envs/aegis_env.py` from `newton_swarm_env.py`)

1. **Per-world `target_pos` array**, re-rolled only for the resetting world (kills F2).
2. Per-drone OOB/altitude death (bad death) instead of world-kill.
3. **Swept-segment intercept AND breach checks** (closed-form min distance over the step
   under constant velocity) — tunneling-proof at 110 m/s closing.
4. Breach despawns the attacker; episode continues.
5. `per_drone_mass` actually passed into `apply_velocity_force_multi`.
6. **Batched numpy** over (worlds, N, M) for obs/rewards/allocator/attacker guidance —
   no Python triple loops. One GPU→CPU sync per control step. Velocity upload via
   `wp.array(vel_np, dtype=wp.vec3)` directly from a float32 (K,3) array.
7. Wind computed once per control step (not per substep), applied inside the force kernel
   via a per-world wind array.
8. Per-world (N_w, M_w) sampled at world reset; slots padded to N_max (unused slots =
   dead-sentinel streams, no episode info emitted).
9. Attacker strategies: direct / flanking / mixed (ported, vectorized) + **jinking**
   (sinusoidal lateral weave). Attackers OOB → eliminated silently (no reward).

### 3.6 Training plan (`train/train_aegis.py`)

All stages: Ukraine scale, 12 Hz, kamikaze on, per-world (N_w, M_w) sampling with
N_max=8 padded slots, ONE reward table. **HOLD OUT 6v4** from all training sampling;
evaluate it (and 9v6, above the training range) every stage — generalization is
verified, not assumed.

PPO: `num_worlds=512` → 4096 streams; `n_steps=128` (rollout 524,288); `batch_size=16384`;
`n_epochs=4`; `gamma=0.995`; `gae_lambda=0.95`; lr 3e-4 → 1e-4 linear within stage;
`clip_range=0.2`; `ent_coef=0.005`; `vf_coef=0.5`; `max_grad_norm=0.5`; `device='cuda'`.
Eval/checkpoint frequencies divided by n_envs (the `d85fd6d` trap).
**Checkpoint selection = lowest breach rate on fixed seeds** (custom callback) — never
mean reward. `reset_num_timesteps=True` + optimizer reset per stage.

Gate policy per stage: fixed-seed eval episodes; on failure extend once by +20M; failing
again → **HALT and dump `analysis/metrics.py` diagnostics** — never blind warm-start.

| Stage | Config | Steps | GATE |
|---|---|---|---|
| 0 pursuit sanity | 1v1, no hills/wind, direct 0.8×, sensor 350 | 20M | intercept ≥90%, bad-death ≤2%, time-to-intercept ≤20 s, approx_kl ∈ 0.01–0.03 |
| 1 core NvM | N~U[2,4], M~U[1,3], direct 1.0×, no hills/wind | 40M | zero-leaker win ≥80%, leakers ≤0.15, bad death ≤3%, no defender >40% of team kills at N≥3, pairwise spacing ≥40 m |
| 2 full range + terrain + dropout | N~U[3,8], M~U[1,8], 5 hills, wind 0.5 N, sensor 300, direct/flanking mix, **exogenous culling p=0.3** | 60M | win ≥70%, parity (N=M) win ≥60%, exchange ratio ≥1.2, spacing ≥100 m, held-out 6v4 within 5 pts of neighbors |
| 3 realism + self-healing | full ranges, mixed/flanking/jinking 1.0×, gusts, obs noise 0.02, mass/speed variance 0.1, culling | 60M | win ≥60%, leakers ≤0.5, kill-injection battery (kill 1/2/3 defenders at t=10 s) breach delta ≤+10 pts for 1-kill, **re-coverage time median ≤3 s / p90 ≤6 s**, kamikaze:bad-death ≥10:1, 9v6 within 5 pts |
| 4 (optional, Phase-3 hook) | adversarial attacker mixture pool | 60M | non-regression on stage-3 grid |

Total ~180M transitions core (+40M contingency). Throughput gate before Stage 1:
≥10k agent-steps/s from the batched-numpy env; port hotspots to Warp **only if** under
target.

Monitoring: check every run at 5-minute intervals from the start of training; watch
`approx_kl` (0.01–0.03), `clip_fraction` (0.05–0.15), `explained_variance`,
effective-sample fraction (≥60%), intercept/leaker curves, `nvidia-smi dmon`. All models
to `models/` with descriptive names (e.g. `models/aegis_stage2_fullrange_best/best_model.zip`).

### 3.7 Pre-flight (before ANY GPU time)

- `analysis/reward_balance.py` archetype battery as blocking pytest: intended behavior
  must strictly dominate ALL known exploit archetypes — eternal chaser, passive ring,
  pile-on, kamikaze avoider, cheap suicide, guard-camper, lazy reserve,
  kill-stealer-ignoring-assignment.
- Tests (`tests/test_aegis.py`): allocation coverage (every detected attacker owned when
  N_alive ≥ M_active); potential telescoping across forced reassignment (no spike);
  dead-stream done/terminal_observation/sentinel semantics; 110 m/s tunneling test;
  per-world target immutability during other worlds' resets; same-checkpoint-runs-
  1v1/3v8/15v10 obs-contract test; greedy-vs-Hungarian 5% gap.

### 3.8 Final eval (`eval/eval_nvm_grid.py`)

Grid N∈{1,3,5,7,10,15} × M∈{1,2,3,5,8,10}, 200 episodes × 3 attacker strategies per
cell, fixed seeds, plus a culled-episode variant and the held-out cells. Metrics per cell
as **CSV+JSON** (numbers in files, not PNG titles) + heatmaps to `models/eval_output/`:
zero-leaker win rate, mean leakers, exchange ratio, bad-death rate, time-to-first-
intercept, re-coverage time, clustering/angular coverage.
**Honest baseline**: scripted greedy-allocator + proportional-navigation defender on the
same grid — the RL policy must beat its breach rate in ≥80% of cells or the learning
isn't paying for itself. N<M cells scored on exchange ratio and leakers, not win rate
(structurally unwinnable with 1:1 kamikaze trades).

---

## 4. Implementation

Diagnosis and plan completed 2026-07-12; implementation followed on branch DD-15
(committed as `07e8afc`).

### 4.1 Checklist

| # | Item | File | Status |
|---|---|---|---|
| 1 | Plan doc | `docs/AEGIS_PLAN.md` | Done (this file) |
| 2 | Allocator (Hungarian primary, greedy fallback, t_go) | `envs/allocator.py` | Done |
| 3 | AEGIS env (all §3.5 fixes, §3.2 obs, §3.4 rewards) | `envs/aegis_env.py` | Done |
| 4 | Tests (§3.7) — 18 passing incl. GPU invariants | `tests/test_aegis.py` | Done |
| 5 | Training pipeline with gates + breach-rate checkpointing | `train/train_aegis.py` | Done |
| 6 | NvM grid eval + scripted baseline | `eval/eval_nvm_grid.py` | Done |
| 7 | Smoke test: 159k–214k agent-steps/s (gate was 10k) | — | Done |
| 8 | Stage 0 training launched + monitored | `logs/train_aegis.log` | Done (2026-07-21) |
| 9 | All 4 stage gates PASSED (retrain post t_go-fix, 2026-07-23, 28 min total) | `models/aegis_stage3_realism_best/best_model.zip` | Done |
| 10 | Final grid eval vs baseline (plain + realism; 38 cells × 3 raid types each) | `models/eval_output/nvm_grid_*_{plain,realism}.csv` | Done |

### 4.2 Implementation deltas vs spec (decided during build)

- Allocator uses **scipy Hungarian** per world as primary (greedy had a 13% mean
  optimality gap on random matrices; Hungarian is µs-cheap at ≤8×8). Greedy remains
  as the no-scipy fallback. Allocation only runs on worlds flagged for realloc.
- Hills are **analytic** (no physics bodies): distance-based kill/shaping checks;
  attackers steer around hills and die on contact (`terrain_kills` in outcomes).
  The urban and maze terrain added on 2026-08-27 uses analytic boxes the same way;
  kill and safety-shaping checks now use the distance to the obstacle surface, and
  the two obstacle observation slots [37:45] report the nearest hills or walls.
- Detection is **latched** (once seen by any defender, tracked forever) for
  allocator stability.
- Empty teammate/hill obs slots carry a dist=1.0 "far away" sentinel (not 0).
- Stage gates as built: `train/train_aegis.py` automates the win, mean-leaker and
  bad-death thresholds of the §3.6 table on 256 fixed-seed episodes (stage 0: win
  ≥0.90, bad-death ≤2%; stage 1: win ≥0.80, leakers ≤0.15, bad-death ≤3%; stage 2:
  win ≥0.70; stage 3: win ≥0.60, leakers ≤0.5). The other gate criteria in the table
  are not checked automatically. Stages 0 and 1 run at 2048 and 1024 worlds rather
  than 512.
- Final grid eval as built: 32 episodes per cell and raid type rather than 200, and the
  scripted baseline uses the same Hungarian allocator but flies straight at the predicted
  intercept point (pursuit-to-PIP) rather than using proportional navigation.
- Early result: Stage 0 (1v1) reached 100% eval win rate / 0 breaches by 3.3M steps.

### 4.3 Results of the planned curriculum (2026-07-23)

Gates: stage0 win=1.00 · stage1 win=0.90 · stage2 win=0.71 · stage3 win=0.81
(bad-death 3%). Plain grid (no terrain/wind; 38 NvM cells × 3 raid types = 114
entries, 32 episodes each): model TIES the scripted baseline (Hungarian allocator +
pursuit-to-PIP) in all 114 entries — both perfect at N≥M (incl. held-out 6v4/9v6 and 15v10),
both exactly 1:1 exchange when outnumbered; model is ~0.3 s faster to first
intercept. REALISM grid (hills+gusts+noise+jink — the point of learning): model beats
baseline on breach rate in 67/114 entries, ties 45, worse in 2. Aggregates
baseline→model: win 0.69→0.76, leakers/ep 1.03→0.81, bad-death 17.4%→2.5%, exchange
ratio 0.68→0.92; parity cells win 0.79→0.99; held-out 6v4 win 0.89→1.00. One shared
56-dim policy runs every cell 1v1→15v10 unchanged.

*Corrected 2026-10-12:* this entry originally called the baseline "Hungarian+PN"
(proportional navigation) and the 114 cell-and-raid-type entries "cells"; the figures
themselves were re-derived from `models/eval_output/nvm_grid_{baseline,model}_{plain,realism}.csv`
and are unchanged.

Next (Phase 3): adversarial co-training via stage 4 hook (mixture-of-opponents
pool per project_status memory); consider raising N<M performance with reserve
tactics (currently XR 0.98, near the 1:1 kamikaze ceiling).

*Follow-up:* the N<M ceiling was removed by the reusable-interceptor mode
(2026-08-19) rather than by reserve tactics; adversarial co-training ran on
2026-08-20/21 with a separate driver, `train/train_aegis_adv.py` (section 5).

---

## 5. Development log (newest first)

Each entry links the dated experiment record in `experiments/` that holds the full
tables. *Follow-up* notes were added later where an entry's next step or open issue
has since been resolved. *Corrected* notes mark entries whose numbers or
interpretation were revised on 2026-10-12 after re-checking the evaluation scripts and
logs; the matching experiment records carry an erratum.

### 2026-09-20 — CrazyFlie hardware path wired (DD-18)

Code: `deploy/mocap.py`, `deploy/cf_link.py`, `deploy/cf_bridge.py`; procedure:
`docs/FIRST_FLIGHT.md`. Position sources share one contract (lab metres, z up): a
JSON-over-UDP feed that any motion-capture rig can drive, the CrazyFlies' own onboard
estimate (Lighthouse or Loco deck) read over cflib, and a direct Qualisys (QTM)
client. The cflib link arms, takes off, streams world-frame velocity setpoints and
external positions, lands and stops. The bridge gained a safety layer: geofence
(motors stop beyond |x|, |y| = 3.8 m or z = 2.2 m), stale-mocap watchdog (0.2 s),
floor/ceiling clamp on vertical commands, speed scaling, a policy-free circle test, a
flight-time limit and per-tick CSV telemetry. Nine tests in `tests/test_cf_deploy.py`,
including an end-to-end run over a real UDP socket; the observation encoder is
unchanged and the 2v2 dry-run reference still intercepts both attackers by 4.1 s. Not
yet flown on hardware.

### 2026-08-31 — Multi-target adversarial verification; champion regression sweep (DD-17)

Record: `experiments/varkadv_20260831/results.md`. Learned attackers trained against
the frozen `aegis_vark` at K=3 reached a true breach rate of 0.005 (gap threshold
0.08), so the hardening round was skipped by design; grid eval of vark against the
learned attacker: win 0.90, breach 0.009–0.013, comparable to scripted direct raids on
the same cells (0.006–0.013). Defender terrain attrition was elevated (32–36%) while
chasing the evasive attacker through hills. With this, every
AEGIS champion (single-target large, convoy, cf_lab, multi-target vark) has been
checked against learned attackers; only the small-team cf_lab ever had a gap, and it
was hardened (2026-08-28). A regression sweep then re-loaded and re-evaluated all 7
champion models (big24, oomval, vark, splitspawn, convoy, sensrobust, cf_lab) under the
final environment; its output was not saved in the repository.

*Corrected 2026-10-12:* the original record called the learned attacker's breach rate
lower than the "scripted ~0.05"; that figure is a fixed string in the training log
message, not a measurement.

### 2026-08-30 — Sensor range vs target spacing (DD-17)

Records: `experiments/varkstress_20260830/results.md`,
`experiments/sensorrobust_20260830/results.md`. Stress sweep of `aegis_vark` at K=1
and K=3 (8v8 and 15v15, n=48): speed parity and a two-defender kill injection degrade
gracefully (K=3: 93% and 98%), but shorter sensor range collapses split defense (K=3:
74% at 150 m, 19% at 75 m, against 95% at the standard 300 m; K=1: 100% and 98%).
Fine-tuning with per-world sensor range sampled from 80–350 m (`aegis_sensrobust`)
recovered 150 m to 80% but moved 75 m only to 22%. The three points sit 126–180 m from
the centre, typically 230–300 m apart, so with short range no defender sees the
attackers heading for a distant point in time. Design rule: detection range must be
comparable to or greater than the spacing between protected points.

*Corrected 2026-10-12:* the original records gave the points as "180 m apart" (180 m is
the placement radius) and the sensor-robust baseline as 350 m (the realism setting uses
300 m).

### 2026-08-30 — Variable target count: one policy for 1–4 points (DD-17)

Record: `experiments/varktargets_20260830/results.md`. `aegis_vark`, fine-tuned from
`aegis_splitspawn` with the number of protected points sampled per world
(K ~ U[1,4]): win 100% / 98% / 96% / 95% at K = 1 / 2 / 3 / 4; saturation raids 100%
at every K, direct raids 100% → 91%. It matches the single-target champions at K=1
and the dedicated K=3 model at K=3, and became the recommended default.

### 2026-08-30 — Closing the split-vs-direct gap (DD-17)

Records: `experiments/splitdirect_20260830/results.md`,
`experiments/splitspawn_20260830/results.md`. Direct simultaneous raids on three
points were the one weak cell of the n=100 scorecard (74–81%). Weighting direct raids
3× in fine-tuning (`aegis_splitdirect`) raised them to 83–85%; distributing defender
spawns round-robin across the K targets (environment change, commit `faa82d3`, now
the multi-target default) raised them to 90–95% for any split model, showing that the
residual gap was spawn geometry rather than policy. `aegis_splitspawn`, trained with
distributed spawns, became the K=3 champion. Defender attrition rose with distributed
spawns, from 3–5% to 9–12% for both models (noted in the record's erratum).

### 2026-08-29 — n=100 scorecard and capabilities report (DD-17)

Logs: `experiments/scorecard_20260829/`. Fixed-seed evaluation with n=100 episodes per
cell and raid type (n=60 for the open-field scale grid), compiled into
`docs/CAPABILITIES.md`. The single-target and scale cells used `aegis_split` (big24
fine-tuned for three points), evaluated with one point; the CrazyFlie cells used
`aegis_adv_defender_cf_lab`. Single-target, split and CrazyFlie cells ran under the
realism setting (five hills, wind with gusts, observation noise, 300 m detection; hills
and wind the CrazyFlie policy never trained on), the scale cells in open field.

### 2026-08-29 — Multi-target split defense (DD-17)

Record: `experiments/split_20260829/results.md`. New capability: the swarm now
protects K separate points at once (num_targets config; K=1 bit-identical, 22/22
tests). Implemented with ZERO allocator/interception change — defenders follow
assigned attackers, which now head to different points, so coverage emerges.
`aegis_split` (K=3 fine-tune) = 0.90 win / 4.5% attrition defending 3 points, beats
big24 zero-shot (0.87 / 8.4%), and retains **1.00 at K=1 (no regression — a strict
superset)**. big24 also generalizes to split zero-shot (0.87), another payoff of the
ego-centric design. Renderer draws all targets; GIF aegis_split3_15v15.

### 2026-08-29 — Speed-brake A/B (DD-17)

Record: `experiments/brake_20260829/results.md`. A physical obstacle speed-brake
(dynamics, not reward; big24 fine-tuned with the brake on) cut defender attrition
4.4 pt in the five-hill 10v10/20v20 test (9.1% → 4.7%, win −2.3 pt) but raised it
19.9 pt in the open-field 40v40/50v40 test (14.6% → 34.5%, win −1.2 pt). The brake
acts only near obstacles, so it never engaged in the open-field test; the increase
there is in ground, ceiling and boundary crashes of the brake-trained policy. Kept
big24. Brake code retained (default off) for ≤20v20 terrain ops. Ops: 448-world AEGIS
OOMs when the concurrent non-AEGIS `dev/probe_gemma_robust.py` shares the box —
use ≤256 worlds when sharing.

*Corrected 2026-10-12:* the original entry explained the 50v40 result as braked
defenders "perpetually slowed in crowded arenas → more collisions" and concluded,
with the safety A/B below, that terrain attrition at high density is intrinsic and not
a caution-tuning problem. The large-scale test contained no obstacles, so neither A/B
measured how obstacle attrition scales with density.

### 2026-08-29 — Safety-shaping A/B (DD-17)

Record: `experiments/safety_20260829/results.md`. Stronger safety shaping
(r_safety_coeff −0.10 → −0.15; the run also set obstacle_danger_radius 110, which the
environment no longer reads) cut defender attrition 3.2 pt in the five-hill 10v10/20v20
test (9.1% → 5.9%) but raised it 10.9 pt in the open-field 40v40/50v40 test (14.6% →
25.4%); win stayed 1.00 in both. **Kept big24 as champion — did NOT adopt.**
`aegis_safe` kept for ~10–20v20 operations in hilly terrain only.

*Corrected 2026-10-12:* the original entry attributed the 50v40 increase to
over-avoidance of the larger 110 m danger rings in crowded arenas, but the danger
radius had no effect (only the coefficient changed) and the 40v40/50v40 test had no
obstacles; what rose there is the rate of ground, ceiling and boundary crashes.

### 2026-08-29 — Big-swarm scale-up (DD-17)

Record: `experiments/bigswarm_20260829/results.md`. First run on the lifted cap: a
**24v24 generalist at 448 worlds** (champion `aegis_big24`, trained on hills, urban
and maze terrain). Trained range 15v15–24v24 = 100% win / zero leakers in the realism
(hills) evaluation, with 5–12% defender attrition. **Generalizes UP with zero extra
training: 30v30, 40v40, 50v40 all 100%, and outnumbered 20v40 (2:1) at 100%
zero-leaker** (open field, n=20 per cell and raid type; reusable interceptors
re-engage). Win rates now hold from 1v1 to 50v40 across the model family. In the
open-field battles big24 lost 10–16% of its defenders; with no obstacles present these
are ground, ceiling or boundary crashes.

*Corrected 2026-10-12:* the original entry described the 15v15–24v24 evaluation as
covering all terrains (it was hills only) and read the 16% open-field losses at 50v40
as terrain attrition rising with density; `aegis_split` lost no defenders in the same
open-field cells (scorecard, 2026-08-29).

### 2026-08-29 — OOM fixed: full 512 worlds restored (DD-17)

Record: `experiments/oomfix_20260829/results.md`. The recurring exit-137 was
GPU-buffer allocation churn (fresh wp.array every step/reset), not RAM. Fix (commit
29a874d): preallocate + `.assign()` in-place, bit-identical (21/21 tests). The
512-world/n_max=20 config that died 4× now runs clean (RAM flat ~12 GB, gate
win 0.97; bonus champion aegis_oomval: maze/scale 1.00 incl. 50v40). **The
≤256-world cap is LIFTED** — full 512 worlds (~2× parallelism) available for all
future large runs.

*Follow-up:* evade-mode adversarial training still ran out of memory at 512 worlds
in the multi-target adversarial run and used 256
(`experiments/varkadv_20260831/results.md`).

### 2026-08-29 — Convoy adversarial verification (DD-17)

Record: `experiments/convoyadv_20260829/results.md`. Learned attackers vs the frozen
moving-target escort: **no gap** (true_breach 0.077 < 0.08; grid 0.89 win / 0.016
breach on 10 m/s targets). Escort is deployment-grade as-is. Emerging **rule**:
small-team policies (cf_lab, 1–4) need an adversarial hardening pass; large-team
policies (largeP, convoy, 4–20) inherit robustness structurally. Attacker kept
for future pools; new GIF aegis_convoyadv_hills_8v10.

### 2026-08-28 — CrazyFlie adversarial verification and deployment bridge (DD-17)

Record: `experiments/cfadv_20260828/results.md`. The small-team lab policy had a REAL
adversarial gap (unlike the large generalist): learned attackers breached frozen
cf_lab at 0.19 (evade gate). One co-training round closed it →
**`aegis_adv_defender_cf_lab`** (win 0.96/breach 0.02 vs mixture; 0.991/0.005 vs
adaptive RL in the grid harness; no scripted regression). **Ship the hardened
model, not raw cf_lab.** Also: **`deploy/cf_bridge.py`** — the CrazyFlie mocap→
policy→setpoint bridge (encoder test-verified against the sim to within 2e-5, --dry-run
flies 2v2 in 4.1 s; cflib path stubbed for first hardware flight). Recurring
exit-137 = transient unified-memory spike, use ≤256 worlds.

*Follow-up:* the cflib path was wired in DD-18 (2026-09-20); the world-count limit
was lifted by the OOM fix (2026-08-29).

*Corrected 2026-10-12:* the original entry called the encoder "bit-identical" to the
sim; the test checks agreement to within 2e-5.

### 2026-08-28 — Night 2: CrazyFlie-ready policy and convoy escort (DD-17)

Record: `experiments/night2_20260828/results.md`. **`aegis_cf_lab`** = validated lab
policy via pure unit-mapping (L=125, T=√2 → 0.34 m/s / 3 m/s² / 8.49 Hz, minimal
sensor obs): gate 1.00 on every N≥M small-team cell vs control's 0.82–0.98,
mean breach 7× lower. **Convoy escort** shipped (moving target, terrain-aware
wander): the ego-centric obs made escort free — static champion already 1.00 on
10 m/s targets; convoy fine-tune matches with attrition 15.3%→10.7%. Next: cflib
hardware bridge (first real CrazyFlie flight), n_max=20 OOM profiling (use ≤288
worlds until fixed).

*Follow-up:* the bridge was built on 2026-08-28 (entry above) and wired to cflib in
DD-18 (2026-09-20); the first flight awaits hardware. The OOM was root-caused and
fixed on 2026-08-29.

### 2026-08-27 — Round 2: attrition pricing; champion `aegis_largeP`

Record: `experiments/overnight_20260827/results.md` (Round 2, 03:43–03:58).
Champion = `aegis_largeP_stage3_realism_best` — attrition-priced retrain (bad_death
−60, safety −0.10): urban attrition 22.7%→**13.3%** at 1.00 win, training-time
mixed-terrain attrition →0.047. Scripted raids now **route through maze gates**
(aegis_maze_routed_10v12.gif). Sensor knockouts: PIP + opportunistic channels are NOT
load-bearing (win 1.00/0.99 without them) — **minimal CrazyFlie sensor suite = own
state + assigned-target track + terrain + boundary**; teammate sensing is a
survivability aid (+10pts attrition without it).

### 2026-08-27 — Large swarms, terrain, and coordinated attackers

Record: `experiments/overnight_20260827/results.md`. New champion
**`aegis_large_stage3_realism_best`**: kamikaze=False generalist trained
N∈[4,20]×M∈[1,20] on randomized hills/**urban**/**maze** terrain (box-obstacle
system, obs contract unchanged) vs new **saturation/waves/decoy** coordinated
raids. Results: urban+maze grids **1.00 win** to 20v20; hills 0.98; open-field
scale **50v40 = 100% win, 0 leakers, ≤0.6% losses**; adversarial check found
**no gap on the first pass** (true breach 0.012 — hardening transfers through
warm-starts). Attacker checkpoints now select on TRUE breach (stalls excluded).
**Dashboard**: `python -m eval.dashboard` → `models/eval_output/dashboard.html`.
Soft spot: ~23% defender terrain attrition in
obstacle worlds (open field ≤1.7%) — next lever per-kill survival bonus; also
attacker route-planning through maze gates.

*Follow-up:* gate-routed raids shipped the same night, and attrition was priced into
the reward instead (Round 2, entry above: urban attrition 22.7% → 13.3%).

### 2026-08-21 — Phase 3: population loop and reuse track

Record: `experiments/adv_pop_20260821/results.md`. The loop CONVERGED in 2 rounds:
fresh-attacker exploit sizes 0.75 → 0.46 → 0.094 ("no gap" → stop). Kamikaze
champion `aegis_adv_defender_r2_best`: 0.55–0.73 win vs every attacker
generation (old defender: 0.01 vs gen-1), baiting defused (bad-deaths 94%→5.5%),
zero scripted regression (0.75). **Best overall model:
`aegis_adv_defender_reuse_best`** (kamikaze=False track, ONE round): 0.93 win vs
its adaptive attacker AND retains 100% N<M wins (3v5/5v8/10v20) — the
reusable-interceptor hypothesis held; residual cost is ~20% terrain attrition
(shaping problem, not strategic). Audit-patch constants are now AegisConfig
defaults; evade mode has an ar_stall=-10 penalty; 'rl' strategy supports
attacker POOLS (attacker_model_paths).

### 2026-08-20 — Phase 3: adversarial co-training, round 1

Record: `experiments/adv_20260820/results.md`. Env gained `mode="evade"` (RL
attackers, 41-dim ego obs) and an `'rl'` attacker strategy (frozen model) for mixture
retraining; driver `train/train_aegis_adv.py`. Findings: learned attackers
crush the scripted-era champion (win 0.78 → **0.01**, 94% defender bad-deaths via
**kamikaze-baiting** into terrain); one co-training round recovers win 0.01 →
0.24, bad-deaths 94% → 20%, leakers 4.2 → 1.8, with **zero scripted regression**
(0.75 vs 0.76 full grid — the mixture prevented forgetting). Next: iterate the
population loop (A2/B2), and co-train the kamikaze=False defender (structurally
less baitable). Models: `models/aegis_adv_{attacker,defender}_best/`.

*Follow-up:* both done on 2026-08-21 (entry above).

### 2026-08-19 — Overnight experiments: scale, reward audit, reusable interceptor

Records: `docs/REWARDS.md`, `experiments/overnight_20260819/results.md`. Highlights:
scale extrapolation perfect to **40v30** (5× trained size); reward-audit patch
adopted (cleaner training, exploit closed); **kamikaze=False reusable-interceptor
mode added to the env** and it removes the N<M ceiling (10v20: 100% win, XR ~9);
self-healing survives 2-defender kill injection at 0.99 win; sensor range is the
binding constraint (0.70 win at 75 m); curriculum confirmed necessary. New models:
`models/aegis_{patch,reuse,scratch3}_stage3_realism_best/`.

---

## 6. Runbook and known traps

```bash
python -m train.train_aegis                      # full staged run (stages 0-3)
python -m train.train_aegis --start-stage 2 --resume models/aegis_stage1_core_nvm_best/best_model.zip
tail -50 logs/train_aegis.log
nvidia-smi dmon -s u -c 3
python -m eval.eval_nvm_grid --model models/aegis_stage3_realism_best/best_model.zip
```

Later models were trained with `--tag <name>` and repeated `--set KEY=VAL` overrides
of `AegisConfig`; the champions use `kamikaze=False`, `r_bad_death=-60.0` and
`r_safety_coeff=-0.10`. The exact command for each model is in
`experiments/<run>/run_queue*.sh`. Reusable-interceptor models must also be evaluated
with `--set kamikaze=False`, because the `AegisConfig` default is still
`kamikaze=True`.

Known trap list (do not reintroduce): eval_freq/ckpt_freq must be divided by n_envs;
`reset_num_timesteps=True` per stage; one-time bad-death elimination; scale-aware
waypoint radii; SAC is banned for this reward regime (Q-divergence, commit f95a86a).
