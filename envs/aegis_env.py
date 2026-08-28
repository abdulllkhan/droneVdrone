"""AEGIS swarm-defense environment (see docs/AEGIS_PLAN.md).

Fork of newton_swarm_env.py implementing the DD-16 redesign:
  - Per-world target positions (kills the global target-teleport bug)
  - Env-side greedy target allocation (envs/allocator.py) — coordination + self-healing
  - Per-agent, owner-based rewards with per-agent done at death (no dead-agent shield)
  - Swept-segment intercept/breach checks (tunneling-proof at 110 m/s closing)
  - Breach despawns the attacker; the episode continues (partial defense measurable)
  - 12 Hz policy rate, 240 Hz physics; gamma=0.995 pairs with this in training
  - Batched numpy for obs/rewards/guidance — one GPU sync per control step
  - Per-world (N_w, M_w) sampled each episode; slots padded to n_max/m_max
  - Exogenous defender culling (self-healing augmentation), zero-reward deaths
  - Per-drone mass actually passed to the force kernel; wind once per control step

Observation: flat Box(56,), ego-centric, fixed for ANY (N, M) — see plan §3.2.
Action: Box(-1, 1, (3,)) desired velocity, scaled by defender_speed_limit.
SB3 sees num_envs = num_worlds * n_max streams (dead/unused slots emit an all-zero
sentinel obs with zero reward and no episode info).
"""

from dataclasses import dataclass, field

import numpy as np
import warp as wp
from gymnasium import spaces
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from envs.allocator import allocate, assigned_tgo, guard_stations, GUARD

OBS_DIM = 56
ATT_OBS_DIM = 41   # evade-mode (RL attacker) observation


@dataclass
class AegisConfig:
    """Ukraine-scale defaults; stage scripts override fields per docs/AEGIS_PLAN.md §3.6."""
    # Team sizes: per-world N ~ U[n_range], M ~ U[m_range] at each world reset.
    n_max: int = 8
    m_max: int = 8
    n_range: tuple = (3, 8)
    m_range: tuple = (1, 8)
    holdout_cells: tuple = ((6, 4),)   # never sampled in training; eval-only
    fixed_nm: tuple = None             # (N, M) exact — for eval; overrides ranges

    # Arena (Ukraine preset)
    arena_bounds: float = 500.0
    altitude_min: float = 5.0
    altitude_max: float = 200.0
    target_alt_range: tuple = (20.0, 60.0)

    # Drones
    defender_speed_limit: float = 60.0
    attacker_speed_limit: float = 50.0
    attacker_speed_scale: float = 1.0
    attacker_speed_variance: float = 0.0
    drone_mass: float = 0.200
    drone_mass_variance: float = 0.0

    # Interception / breach
    intercept_dist: float = 3.0
    target_reached_dist: float = 5.0
    kamikaze: bool = True
    # kamikaze=False: defender survives intercepts (gun/net interceptor) but must
    # rearm for this many steps before it can intercept again (2 s at 12 Hz).
    intercept_cooldown_steps: int = 24

    # Timing: policy 12 Hz, physics 240 Hz
    ctrl_freq: int = 12
    phys_freq: int = 240
    episode_secs: float = 60.0

    # Force model
    force_kp: float = 50.0
    force_kd: float = 15.0
    max_force: float = 150.0
    angular_damping: float = 5.0

    # Terrain: "hills" (random cylinders, count=num_obstacles), "urban" (grid of
    # tower blocks), "maze" (concentric broken-ring labyrinth), "none".
    # `terrains` non-empty -> sampled per world each episode (terrain generalist).
    terrain: str = "hills"
    terrains: tuple = ()

    # Hills (analytic — no physics bodies; distance-based kill/shaping)
    num_obstacles: int = 0
    obstacle_radius: float = 50.0
    obstacle_height: float = 60.0      # half-height; top of hill = 2*obstacle_height
    obstacle_placement_range: float = 350.0
    obstacle_min_spacing: float = 120.0
    obstacle_collision_radius: float = 55.0   # horizontal kill radius
    obstacle_danger_radius: float = 90.0      # safety shaping starts here (audit: was 70)

    # Sensing (global shared detection, latched once seen)
    sensor_range: float = 350.0        # 0 = unlimited
    obs_noise_std: float = 0.0
    obs_opportunistic: bool = True     # 8-dim nearest-other-attacker block
    obs_zero_pip: bool = False         # zero PIP hint [22:25] (train-time knockout
                                       # for minimal-sensor hardware builds)

    # Wind
    wind_enabled: bool = False
    wind_max_force: float = 0.5
    wind_change_freq: int = 12         # control steps between wind re-rolls
    wind_gust_prob: float = 0.0
    wind_gust_force: float = 1.5

    # Attackers
    attacker_strategies: tuple = ("direct",)   # sampled per world per episode
                                               # ('rl' = frozen attacker_model_path)

    # Adversarial co-training (Phase 3)
    mode: str = "defend"               # "defend" (RL=defenders) | "evade" (RL=attackers)
    defender_model_path: str = ""      # frozen defender policy driving defenders in evade mode
    attacker_model_path: str = ""      # frozen RL attacker used by the 'rl' strategy
    attacker_model_paths: tuple = ()   # POOL of frozen attackers ('rl' worlds sample one)
    # attacker reward table (evade mode) — mirrors the defender philosophy
    ar_close_coeff: float = 0.03       # per meter closed toward the target
    ar_breach: float = 30.0            # reached the target (attacker's win)
    ar_intercepted: float = -20.0      # killed by a defender
    ar_bad_death: float = -15.0        # terrain / out-of-bounds
    ar_time: float = -0.01
    ar_safety_coeff: float = -0.02
    ar_stall: float = -10.0            # still airborne at episode timeout — stalling
                                       # is mission failure, not evasion success

    # Self-healing augmentation: exogenous zero-reward defender culls
    cull_prob: float = 0.0             # per-episode probability
    cull_step_range: tuple = (120, 480)
    cull_count: int = 1                # defenders killed at the cull step (kill-injection battery)

    # Allocation
    realloc_every: int = 6             # control steps (0.5 s at 12 Hz)
    guard_ring_radius: float = 100.0

    # Rewards (plan §3.4) — ONE table for all stages
    r_close_coeff: float = 0.03        # per meter toward assigned attacker
    r_guard_close_coeff: float = 0.01  # per meter toward ring station
    r_guard_hold: float = 0.005        # per step within 150 m of target
    # 2026-08-19 audit patch is now the default (docs/REWARDS.md):
    # bad_death -15->-30 (closes the dead-owner suicide escape), unowned -10->-5,
    # safety -0.02->-0.05 (hazard signal was 14:1 outgunned by chase shaping)
    r_intercept: float = 30.0
    r_bad_death: float = -30.0
    r_breach_owner: float = -25.0
    r_breach_other: float = -5.0
    r_breach_unowned: float = -5.0
    r_team_win: float = 10.0
    r_time: float = -0.01
    r_safety_coeff: float = -0.05
    boundary_margin: float = 25.0

    # Moving protected target ("convoy escort"). (0,0) = static, bit-identical
    # to prior behavior. Speed sampled per world per episode; the target wanders
    # waypoint-to-waypoint inside target_wander_radius (capped per terrain).
    target_speed_range: tuple = (0.0, 0.0)
    target_wander_radius: float = 250.0
    target_waypoint_tol: float = 5.0

    # Spawns
    defender_dist_range: tuple = (20.0, 120.0)
    defender_alt_range: tuple = (30.0, 150.0)
    attacker_dist_range: tuple = (200.0, 450.0)
    attacker_alt_range: tuple = (30.0, 90.0)

    seed: int = 0


@wp.kernel
def aegis_force_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_vel: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
    num_drones: int,
    mass: wp.array(dtype=float),
    gravity: float,
    kp: float,
    kd: float,
    max_force: float,
    angular_damping: float,
    wind: wp.array(dtype=wp.vec3),
):
    """PD velocity tracking + gravity compensation + per-world wind, per-drone mass."""
    tid = wp.tid()
    world_id = tid // num_drones

    vel = wp.vec3(body_qd[tid][0], body_qd[tid][1], body_qd[tid][2])
    desired = target_vel[tid]
    m = mass[tid]

    vel_error = desired - vel
    ctrl = wp.vec3(
        vel_error[0] * kp * m - vel[0] * kd * m,
        vel_error[1] * kp * m - vel[1] * kd * m,
        vel_error[2] * kp * m - vel[2] * kd * m,
    )
    mag = wp.length(ctrl)
    if mag > max_force:
        ctrl = ctrl * (max_force / mag)

    w = wind[world_id]
    force = wp.vec3(ctrl[0] + w[0], ctrl[1] + w[1], ctrl[2] + m * gravity + w[2])

    ang_vel = wp.vec3(body_qd[tid][3], body_qd[tid][4], body_qd[tid][5])
    torque = ang_vel * (-angular_damping)

    wp.atomic_add(body_f, tid, wp.spatial_vector(force, torque))


@wp.kernel
def aegis_reset_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    init_positions: wp.array(dtype=wp.vec3),
    reset_mask: wp.array(dtype=int),
    num_drones: int,
):
    tid = wp.tid()
    world_id = tid // num_drones
    if reset_mask[world_id] == 0:
        return
    body_q[tid] = wp.transform(init_positions[tid], wp.quat_identity())
    body_qd[tid] = wp.spatial_vector(wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0))


def _build_drone(builder, arm_length, collision_group, label):
    import newton as nt
    body = builder.add_body(label=label)
    bar = arm_length * 0.05
    cfg = nt.ModelBuilder.ShapeConfig(density=781.5, collision_group=collision_group)
    builder.add_shape_box(body, hx=bar, hy=arm_length, hz=bar, cfg=cfg)
    builder.add_shape_box(body, hx=arm_length, hy=bar, hz=bar, cfg=cfg)
    return body


def segment_min_dist(p0, p1):
    """Min |p(t)| over the segment p(t) = p0 + t*(p1-p0), t in [0,1], batched (...,3)."""
    d = p1 - p0
    dd = np.sum(d * d, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -np.sum(p0 * d, axis=-1) / np.where(dd > 1e-12, dd, 1.0)
    t = np.clip(np.where(dd > 1e-12, t, 0.0), 0.0, 1.0)
    closest = p0 + t[..., None] * d
    return np.linalg.norm(closest, axis=-1)


class AegisSwarmEnv(VecEnv):
    """N-defenders-vs-M-attackers GPU env with env-side target allocation."""

    metadata = {}

    def __init__(self, num_worlds, config=None):
        import newton as nt

        self.cfg = config or AegisConfig()
        c = self.cfg
        self.num_worlds = num_worlds
        self.n_max = c.n_max
        self.m_max = c.m_max
        self._num_drones = self.n_max + self.m_max
        self.mode = c.mode
        self.n_envs = num_worlds * (self.m_max if self.mode == "evade" else self.n_max)
        self.rl_obs_dim = ATT_OBS_DIM if self.mode == "evade" else OBS_DIM
        self.defender_model = None
        self.attacker_model = None

        self.dt = 1.0 / c.ctrl_freq
        self.substeps = max(1, c.phys_freq // c.ctrl_freq)
        self.sub_dt = self.dt / self.substeps
        self.max_steps = int(c.episode_secs * c.ctrl_freq)
        self.attacker_speed = c.attacker_speed_scale * c.attacker_speed_limit

        self._rng = np.random.default_rng(c.seed)
        self.device = "cuda:0"

        # --- physics scene: n_max + m_max drones per world, no obstacle bodies ---
        builder = nt.ModelBuilder(gravity=-9.81)
        arm = 0.2
        for w in range(num_worlds):
            base = w * (self._num_drones + 1)
            for d in range(self.n_max):
                _build_drone(builder, arm, base + d, f"def_{w}_{d}")
            for a in range(self.m_max):
                _build_drone(builder, arm, base + self.n_max + a, f"att_{w}_{a}")
        self.model = builder.finalize(device=self.device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.solver = nt.solvers.SolverSemiImplicit(self.model)

        K = num_worlds * self._num_drones
        self._vel_np = np.zeros((K, 3), dtype=np.float32)
        self.drone_vel = wp.zeros(K, dtype=wp.vec3, device=self.device)
        self._mass_np = np.full(K, c.drone_mass, dtype=np.float32)
        self.mass_wp = wp.array(self._mass_np, dtype=float, device=self.device)
        self._wind_np = np.zeros((num_worlds, 3), dtype=np.float32)
        self.wind_wp = wp.array(self._wind_np, dtype=wp.vec3, device=self.device)
        self._init_pos_np = np.zeros((K, 3), dtype=np.float32)

        W, N, M = num_worlds, self.n_max, self.m_max
        # --- per-world episode state ---
        self.target_pos = np.zeros((W, 3), dtype=np.float32)     # PER-WORLD target
        self.n_w = np.full(W, N, dtype=np.int64)
        self.m_w = np.full(W, M, dtype=np.int64)
        self.step_count = np.zeros(W, dtype=np.int64)
        self.def_alive = np.zeros((W, N), dtype=bool)
        self.att_active = np.zeros((W, M), dtype=bool)
        self.att_detected = np.zeros((W, M), dtype=bool)         # latched
        self.att_breached = np.zeros((W, M), dtype=bool)
        self.att_terrain = np.zeros((W, M), dtype=bool)
        self.slot_real = np.zeros((W, N), dtype=bool)            # slot < n_w at spawn
        self.stream_dead = np.ones((W, N), dtype=bool)           # emits sentinel
        self.per_att_speed = np.full((W, M), self.attacker_speed, dtype=np.float32)
        self.world_strategy = np.zeros(W, dtype=np.int64)
        self.jink_phase = np.zeros((W, M), dtype=np.float32)
        self.cull_step = np.full(W, -1, dtype=np.int64)
        self.obstacle_pos = np.zeros((W, max(c.num_obstacles, 1), 3), dtype=np.float32)
        # box obstacles (walls/buildings): cx, cy, hx, hy, top_z per slot
        self.MAXB = 24
        self.box_obs = np.zeros((W, self.MAXB, 5), dtype=np.float32)
        self.box_count = np.zeros(W, dtype=np.int64)
        self.world_terrain = np.array(["none"] * W, dtype=object)
        # maze gate midpoints for scripted route-planning: (x, y, ring_radius)
        self.gate_pos = np.zeros((W, 12, 3), dtype=np.float32)
        self.gate_count = np.zeros(W, dtype=np.int64)
        self.is_maze = np.zeros(W, dtype=bool)
        # moving-target (convoy) state
        self.target_pos_prev = np.zeros((W, 3), dtype=np.float32)
        self.target_vel = np.zeros((W, 3), dtype=np.float32)
        self.target_speed_w = np.zeros(W, dtype=np.float32)
        self.target_wp = np.zeros((W, 2), dtype=np.float32)

        self.hill_count = np.zeros(W, dtype=np.int64)

        # --- allocation state ---
        self.assign = np.full((W, N), GUARD, dtype=np.int64)
        self.stations = np.zeros((W, N, 3), dtype=np.float32)
        self.rearm = np.zeros((W, N), dtype=np.int64)   # intercept cooldown (kamikaze=False)
        self.prev_pot_dist = np.zeros((W, N), dtype=np.float32)  # shaping anchor
        self.steps_since_reassign = np.zeros((W, N), dtype=np.int64)
        self.realloc_needed = np.ones(W, dtype=bool)

        # --- episode bookkeeping ---
        self._ep_rew = np.zeros((W, N), dtype=np.float64)
        self._ep_len = np.zeros((W, N), dtype=np.int64)
        # attacker-side RL bookkeeping (evade mode)
        self.att_stream_dead = np.ones((W, M), dtype=bool)
        self.att_prev_dist = np.zeros((W, M), dtype=np.float32)
        self._att_ep_rew = np.zeros((W, M), dtype=np.float64)
        self._att_ep_len = np.zeros((W, M), dtype=np.int64)
        self.ep_intercepts = np.zeros(W, dtype=np.int64)
        self.ep_bad_deaths = np.zeros(W, dtype=np.int64)
        self.ep_culled = np.zeros(W, dtype=np.int64)
        self.ep_first_intercept = np.full(W, -1, dtype=np.int64)
        self.episode_outcomes = []   # dicts appended at world end; pop via pop_outcomes()

        self.def_pos = np.zeros((W, N, 3), dtype=np.float32)
        self.def_vel = np.zeros((W, N, 3), dtype=np.float32)
        self.att_pos = np.zeros((W, M, 3), dtype=np.float32)
        self.att_vel = np.zeros((W, M, 3), dtype=np.float32)
        self.def_pos_prev = np.zeros((W, N, 3), dtype=np.float32)
        self.att_pos_prev = np.zeros((W, M, 3), dtype=np.float32)

        self.def_obs_buf = np.zeros((W * N, OBS_DIM), dtype=np.float32)
        self.att_obs_buf = np.zeros((W * M, ATT_OBS_DIM), dtype=np.float32)
        self.obs_buf = np.zeros((self.n_envs, self.rl_obs_dim), dtype=np.float32)

        if c.defender_model_path:
            self.defender_model = self._load_frozen(c.defender_model_path)
        self.attacker_models = []
        pool = list(c.attacker_model_paths) or (
            [c.attacker_model_path] if c.attacker_model_path else [])
        for p in pool:
            self.attacker_models.append(self._load_frozen(p))
        self.attacker_model = self.attacker_models[0] if self.attacker_models else None
        self.rl_model_idx = np.zeros(num_worlds, dtype=np.int64)

        obs_space = spaces.Box(-np.inf, np.inf, shape=(self.rl_obs_dim,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        super().__init__(self.n_envs, obs_space, act_space)

    @staticmethod
    def _load_frozen(path):
        from stable_baselines3 import PPO
        return PPO.load(path, device="cuda")

    # ------------------------------------------------------------------ #
    # Reset / spawn
    # ------------------------------------------------------------------ #

    def _sample_nm(self):
        c = self.cfg
        if c.fixed_nm is not None:
            return int(c.fixed_nm[0]), int(c.fixed_nm[1])
        for _ in range(50):
            n = int(self._rng.integers(c.n_range[0], c.n_range[1] + 1))
            m = int(self._rng.integers(c.m_range[0], c.m_range[1] + 1))
            if (n, m) not in tuple(map(tuple, c.holdout_cells or ())):
                return n, m
        return n, m

    def _spawn_world(self, w):
        """Sample (N,M), target, positions, strategy, culling for world w (CPU only)."""
        c = self.cfg
        rng = self._rng
        n, m = self._sample_nm()
        self.n_w[w], self.m_w[w] = n, m
        self.target_pos[w] = (0.0, 0.0, rng.uniform(*c.target_alt_range))

        base = w * self._num_drones
        # Defenders: ring around target; unused slots parked high in a corner.
        for d in range(self.n_max):
            if d < n:
                ang = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(*c.defender_dist_range)
                self._init_pos_np[base + d] = (
                    self.target_pos[w, 0] + dist * np.cos(ang),
                    self.target_pos[w, 1] + dist * np.sin(ang),
                    rng.uniform(*c.defender_alt_range),
                )
            else:
                self._init_pos_np[base + d] = (
                    c.arena_bounds - 20.0 - 8.0 * d, c.arena_bounds - 20.0,
                    c.altitude_max - 20.0,
                )
        # Attackers: far ring; unused parked in opposite corner.
        for a in range(self.m_max):
            if a < m:
                ang = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(*c.attacker_dist_range)
                self._init_pos_np[base + self.n_max + a] = (
                    self.target_pos[w, 0] + dist * np.cos(ang),
                    self.target_pos[w, 1] + dist * np.sin(ang),
                    rng.uniform(*c.attacker_alt_range),
                )
            else:
                self._init_pos_np[base + self.n_max + a] = (
                    -(c.arena_bounds - 20.0) + 8.0 * a, -(c.arena_bounds - 20.0),
                    c.altitude_max - 20.0,
                )

        self.def_alive[w] = False
        self.def_alive[w, :n] = True
        self.slot_real[w] = False
        self.slot_real[w, :n] = True
        self.stream_dead[w] = ~self.slot_real[w]
        self.att_active[w] = False
        self.att_active[w, :m] = True
        self.att_detected[w] = False
        self.att_breached[w] = False
        self.att_terrain[w] = False
        self.step_count[w] = 0
        self.assign[w] = GUARD
        self.rearm[w] = 0
        self.prev_pot_dist[w] = 0.0
        self.steps_since_reassign[w] = 0
        self.realloc_needed[w] = True
        self._ep_rew[w] = 0.0
        self._ep_len[w] = 0
        self.ep_intercepts[w] = 0
        self.ep_bad_deaths[w] = 0
        self.ep_culled[w] = 0
        self.ep_first_intercept[w] = -1

        self.per_att_speed[w] = self.attacker_speed
        if c.attacker_speed_variance > 0:
            self.per_att_speed[w] *= 1.0 + rng.uniform(
                -c.attacker_speed_variance, c.attacker_speed_variance, self.m_max
            ).astype(np.float32)
        if c.drone_mass_variance > 0:
            self._mass_np[base:base + self._num_drones] = c.drone_mass * (
                1.0 + rng.uniform(-c.drone_mass_variance, c.drone_mass_variance,
                                  self._num_drones).astype(np.float32))

        strategies = c.attacker_strategies or ("direct",)
        self.world_strategy[w] = list(_STRATEGIES).index(
            strategies[int(rng.integers(len(strategies)))])
        if self.attacker_models:
            self.rl_model_idx[w] = int(rng.integers(len(self.attacker_models)))
        self.jink_phase[w] = rng.uniform(0, 2 * np.pi, self.m_max).astype(np.float32)

        self.cull_step[w] = -1
        if c.cull_prob > 0 and rng.random() < c.cull_prob:
            self.cull_step[w] = int(rng.integers(*c.cull_step_range))

        # attacker-side RL bookkeeping
        self.att_stream_dead[w] = True
        self.att_stream_dead[w, :m] = False
        flat_att = self._init_pos_np.reshape(self.num_worlds, self._num_drones, 3)[w, self.n_max:]
        self.att_prev_dist[w] = np.linalg.norm(flat_att - self.target_pos[w], axis=-1)
        self._att_ep_rew[w] = 0.0
        self._att_ep_len[w] = 0

        # --- terrain generation ---
        t = c.terrain
        if c.terrains:
            t = c.terrains[int(rng.integers(len(c.terrains)))]
        self.world_terrain[w] = t
        self.box_count[w] = 0
        self.hill_count[w] = 0
        self.is_maze[w] = False
        self.gate_count[w] = 0
        if t == "hills" and c.num_obstacles > 0:
            placed = []
            for _ in range(c.num_obstacles):
                for _try in range(50):
                    ang = rng.uniform(0, 2 * np.pi)
                    dist = rng.uniform(120.0, c.obstacle_placement_range)
                    p = (dist * np.cos(ang), dist * np.sin(ang), c.obstacle_height)
                    if all(np.hypot(p[0] - q[0], p[1] - q[1]) >= c.obstacle_min_spacing
                           for q in placed):
                        break
                placed.append(p)
            self.obstacle_pos[w, :len(placed)] = placed
            self.hill_count[w] = len(placed)
        elif t == "urban":
            # grid of tower blocks; streets ~80-100 m wide; clear zone at target
            k = 0
            for gx in range(-2, 3):
                for gy in range(-2, 3):
                    if k >= self.MAXB:
                        break
                    cx = gx * 145.0 + rng.uniform(-18, 18)
                    cy = gy * 145.0 + rng.uniform(-18, 18)
                    if np.hypot(cx - self.target_pos[w, 0],
                                cy - self.target_pos[w, 1]) < 120.0:
                        continue
                    self.box_obs[w, k] = (cx, cy, rng.uniform(22, 34),
                                          rng.uniform(22, 34), rng.uniform(60, 110))
                    k += 1
            self.box_count[w] = k
        elif t == "maze":
            # labyrinth: 3 concentric square walls, one random gate per side
            k = 0
            g = 0
            self.is_maze[w] = True
            for r in (150.0, 250.0, 350.0):
                top, th, gap = 100.0, 8.0, 45.0
                for side in range(4):
                    gc = rng.uniform(-r + 70, r - 70)
                    gate_xy = ((r, gc), (-r, gc), (gc, r), (gc, -r))[side]
                    self.gate_pos[w, g] = (gate_xy[0], gate_xy[1], r)
                    g += 1
                    for lo, hi in ((-r, gc - gap), (gc + gap, r)):
                        if hi - lo < 20 or k >= self.MAXB:
                            continue
                        mid, half = (lo + hi) / 2, (hi - lo) / 2
                        if side == 0:
                            self.box_obs[w, k] = (r, mid, th, half, top)
                        elif side == 1:
                            self.box_obs[w, k] = (-r, mid, th, half, top)
                        elif side == 2:
                            self.box_obs[w, k] = (mid, r, half, th, top)
                        else:
                            self.box_obs[w, k] = (mid, -r, half, th, top)
                        k += 1
            self.box_count[w] = k
            self.gate_count[w] = g

        if c.wind_enabled:
            ang = rng.uniform(0, 2 * np.pi)
            mag = rng.uniform(0, c.wind_max_force)
            self._wind_np[w] = (mag * np.cos(ang), mag * np.sin(ang), 0.0)

        # moving-target (convoy) init — AFTER terrain, the sampler needs obstacles
        self.target_speed_w[w] = rng.uniform(*c.target_speed_range)
        self.target_vel[w] = 0.0
        self.target_pos_prev[w] = self.target_pos[w]
        self.target_wp[w] = (self._sample_target_wp(w)
                             if self.target_speed_w[w] > 0
                             else self.target_pos[w, :2])

    def _effective_wander_radius(self, w):
        cap = _TARGET_TERRAIN_CAP.get(self.world_terrain[w], np.inf)
        return min(self.cfg.target_wander_radius, cap)

    def _sample_target_wp(self, w):
        """Waypoint whose approach segment stays clear of terrain kill zones."""
        rng = self._rng
        radius = self._effective_wander_radius(w)
        cur = self.target_pos[w, :2]
        if radius <= 0:
            return cur.copy()
        for _ in range(20):
            ang = rng.uniform(0, 2 * np.pi)
            r = radius * np.sqrt(rng.uniform())
            cand = np.array([r * np.cos(ang), r * np.sin(ang)], dtype=np.float32)
            pts = cur[None, :] + np.linspace(0, 1, 16)[:, None] * (cand - cur)[None, :]
            ok = True
            H = int(self.hill_count[w])
            if H and np.min(np.linalg.norm(
                    pts[:, None, :] - self.obstacle_pos[w, :H, :2][None], axis=-1)) < 100.0:
                ok = False
            B = int(self.box_count[w])
            if ok and B:
                bx = self.box_obs[w, :B]
                dx = np.abs(pts[:, None, 0] - bx[None, :, 0]) - bx[None, :, 2]
                dy = np.abs(pts[:, None, 1] - bx[None, :, 1]) - bx[None, :, 3]
                if np.min(np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))) < 15.0:
                    ok = False
            if ok:
                return cand
        return (0.5 * cur).astype(np.float32)   # pull toward the clear center

    def _upload_resets(self, mask):
        """Push spawn positions/masses/wind for masked worlds to the GPU."""
        mask_wp = wp.array(mask.astype(np.int32), dtype=int, device=self.device)
        init_wp = wp.array(self._init_pos_np, dtype=wp.vec3, device=self.device)
        wp.launch(
            aegis_reset_kernel,
            dim=self.num_worlds * self._num_drones,
            inputs=(self.state_0.body_q, self.state_0.body_qd, init_wp, mask_wp,
                    self._num_drones),
            device=self.device,
        )
        self.mass_wp = wp.array(self._mass_np, dtype=float, device=self.device)
        wp.synchronize()
        # refresh CPU mirrors for the reset worlds
        idx = np.nonzero(mask)[0]
        flat = self._init_pos_np.reshape(self.num_worlds, self._num_drones, 3)
        self.def_pos[idx] = flat[idx, :self.n_max]
        self.att_pos[idx] = flat[idx, self.n_max:]
        self.def_pos_prev[idx] = self.def_pos[idx]
        self.att_pos_prev[idx] = self.att_pos[idx]
        self.target_pos_prev[idx] = self.target_pos[idx]
        self.def_vel[idx] = 0.0
        self.att_vel[idx] = 0.0

    def reset(self):
        for w in range(self.num_worlds):
            self._spawn_world(w)
        self._upload_resets(np.ones(self.num_worlds, dtype=bool))
        self._update_detection()
        self._reallocate(np.ones(self.num_worlds, dtype=bool))
        self._anchor_potentials(np.ones((self.num_worlds, self.n_max), dtype=bool))
        self._compute_all_obs()
        return self.obs_buf.copy()

    # ------------------------------------------------------------------ #
    # Allocation / detection
    # ------------------------------------------------------------------ #

    def _update_detection(self):
        """Latched global shared detection: any alive defender within sensor range."""
        c = self.cfg
        if c.sensor_range <= 0:
            newly = self.att_active & ~self.att_detected
        else:
            diff = self.att_pos[:, None, :, :] - self.def_pos[:, :, None, :]
            dist = np.linalg.norm(diff, axis=-1)                       # (W, N, M)
            dist[~self.def_alive] = np.inf
            seen = (dist <= c.sensor_range).any(axis=1)                # (W, M)
            newly = seen & self.att_active & ~self.att_detected
        if newly.any():
            self.att_detected |= newly
            self.realloc_needed |= newly.any(axis=1)

    def _reallocate(self, world_mask):
        """Re-run the allocator for masked worlds only; re-anchor changed assignments."""
        if not world_mask.any():
            return
        idx = np.nonzero(world_mask)[0]
        sub_assign = allocate(
            self.def_pos[idx], self.def_alive[idx], self.att_pos[idx],
            self.att_vel[idx], self.att_active[idx], self.att_detected[idx],
            self.target_pos[idx], self.cfg.defender_speed_limit,
            prev_assign=self.assign[idx], target_vel=self.target_vel[idx],
        )
        new_assign = self.assign.copy()
        new_assign[idx] = sub_assign
        changed = (new_assign != self.assign) & world_mask[:, None]
        self.assign = new_assign
        old_stations = self.stations
        self.stations = guard_stations(self.assign, self.def_alive, self.target_pos,
                                       self.cfg.guard_ring_radius)
        # a guard whose ring station moved (teammate died / role churn) must also
        # re-anchor, or the station jump pays a spurious shaping spike
        station_moved = ((self.assign == GUARD) & self.def_alive &
                         (np.linalg.norm(self.stations - old_stations, axis=-1) > 1e-6))
        changed = changed | (station_moved & world_mask[:, None])
        self.steps_since_reassign[changed] = 0
        self._anchor_potentials(changed)
        self.realloc_needed[world_mask] = False

    def _terrain_shell(self, pos):
        """Unified obstacle proximity for hills AND box walls/buildings.

        pos (W,K,3) -> (shell (W,K,O), below (W,K,O), centers (W,O,3)), or
        (None,None,None) if no world has obstacles. Shell = horizontal distance
        to the obstacle SURFACE. Kill at shell<5 m while below the top; danger
        shaping ramps inside shell<40 m (matches the old hill constants:
        collision 55 m center / danger 90 m center at radius 50).
        """
        c = self.cfg
        if not self.hill_count.any() and not self.box_count.any():
            return None, None, None
        W, K = pos.shape[0], pos.shape[1]
        H, B = self.obstacle_pos.shape[1], self.MAXB
        hd = np.linalg.norm(
            pos[:, :, None, :2] - self.obstacle_pos[:, None, :, :2], axis=-1)
        h_shell = np.where(
            np.arange(H)[None, None, :] < self.hill_count[:, None, None],
            hd - c.obstacle_radius, np.inf)
        h_below = np.broadcast_to(
            pos[:, :, None, 2] < 2 * c.obstacle_height, h_shell.shape)
        dx = np.abs(pos[:, :, None, 0] - self.box_obs[:, None, :, 0]) \
            - self.box_obs[:, None, :, 2]
        dy = np.abs(pos[:, :, None, 1] - self.box_obs[:, None, :, 1]) \
            - self.box_obs[:, None, :, 3]
        b_shell = np.where(
            np.arange(B)[None, None, :] < self.box_count[:, None, None],
            np.hypot(np.maximum(dx, 0), np.maximum(dy, 0)), np.inf)
        b_below = pos[:, :, None, 2] < self.box_obs[:, None, :, 4]
        shell = np.concatenate([h_shell, b_shell], axis=-1)
        below = np.concatenate([h_below, b_below], axis=-1)
        centers = np.concatenate([
            np.concatenate([self.obstacle_pos[:, :, :2],
                            np.full((W, H, 1), c.obstacle_height, np.float32)], -1),
            np.concatenate([self.box_obs[:, :, :2],
                            self.box_obs[:, :, 4:5] * 0.5], -1),
        ], axis=1).astype(np.float32)
        return shell, below, centers

    def _pot_dist(self):
        """Distance to my potential goal: assigned attacker, or guard station. (W,N)"""
        goal = self.stations.copy()
        has_target = self.assign >= 0
        gathered = np.take_along_axis(
            self.att_pos, np.clip(self.assign, 0, None)[..., None], axis=1)
        goal[has_target] = gathered[has_target]
        return np.linalg.norm(goal - self.def_pos, axis=-1)

    def _anchor_potentials(self, mask):
        """Re-anchor shaping distance (reassignment is reward-neutral by construction)."""
        if mask.any():
            self.prev_pot_dist = np.where(mask, self._pot_dist(), self.prev_pot_dist)

    # ------------------------------------------------------------------ #
    # Attacker guidance (vectorized, scripted)
    # ------------------------------------------------------------------ #

    def _attacker_velocities(self):
        """Desired velocities (W, M, 3) for all active attackers."""
        c = self.cfg
        W, M = self.num_worlds, self.m_max
        to_target = self.target_pos[:, None, :] - self.att_pos          # (W, M, 3)
        dist = np.linalg.norm(to_target, axis=-1, keepdims=True)
        dist = np.maximum(dist, 0.01)
        heading = to_target / dist

        strat = self.world_strategy[:, None]                            # (W, 1)
        a_idx = np.arange(M)[None, :]

        # flanking: offset waypoint until close; mixed: even slots flank
        converge = max(3.0 * c.intercept_dist, 50.0)
        want_flank = ((strat == _STRATEGIES.index("flanking")) |
                      ((strat == _STRATEGIES.index("mixed")) & (a_idx % 2 == 0)))
        want_flank = want_flank & (dist[..., 0] > converge)
        if want_flank.any():
            ang = 2 * np.pi * a_idx / max(M, 1) + np.pi / 4
            radius = dist[..., 0] * 0.15
            wpt = self.target_pos[:, None, :] + np.stack(
                [radius * np.cos(ang), radius * np.sin(ang), np.zeros_like(radius)],
                axis=-1)
            fdir = wpt - self.att_pos
            fdir /= np.maximum(np.linalg.norm(fdir, axis=-1, keepdims=True), 0.01)
            heading = np.where(want_flank[..., None], fdir, heading)

        # jinking: sinusoidal lateral weave on top of direct heading
        if (strat == _STRATEGIES.index("jink")).any():
            t = self.step_count[:, None] * self.dt
            lat = np.cross(heading, np.array([0.0, 0.0, 1.0]))
            lat /= np.maximum(np.linalg.norm(lat, axis=-1, keepdims=True), 0.01)
            weave = 0.4 * np.sin(2 * np.pi * t / 4.0 + self.jink_phase)[..., None]
            jdir = heading + weave * lat
            jdir /= np.maximum(np.linalg.norm(jdir, axis=-1, keepdims=True), 0.01)
            heading = np.where((strat == _STRATEGIES.index("jink"))[..., None],
                               jdir, heading)

        # --- maze route-planning: thread the ring gates, don't bump walls ---
        commit = np.zeros((W, M), dtype=bool)   # near a gate: fly through, no dodge
        if self.is_maze.any():
            r_att = np.linalg.norm(
                self.att_pos[..., :2] - self.target_pos[:, None, :2], axis=-1)
            routed = np.zeros((W, M), dtype=bool)
            for ri, ring in ((2, 350.0), (1, 250.0), (0, 150.0)):
                gates = self.gate_pos[:, ri * 4:(ri + 1) * 4, :2]        # (W,4,2)
                need = (self.is_maze[:, None] & self.att_active &
                        (r_att > ring + 25.0) & ~routed)
                if not need.any():
                    continue
                d = np.linalg.norm(
                    gates[:, None, :, :] - self.att_pos[:, :, None, :2], axis=-1)
                gi = np.argmin(d, axis=-1)
                gxy = np.take_along_axis(
                    gates[:, None, :, :].repeat(M, 1),
                    gi[..., None, None].repeat(2, -1), axis=2)[:, :, 0, :]
                vec = np.concatenate([gxy - self.att_pos[..., :2],
                                      40.0 - self.att_pos[..., 2:3]], axis=-1)
                vn = vec / np.maximum(np.linalg.norm(vec, axis=-1, keepdims=True), 0.01)
                heading = np.where(need[..., None], vn, heading)
                gate_d = np.take_along_axis(d, gi[..., None], axis=-1)[..., 0]
                commit |= need & (gate_d < 70.0)
                routed |= need

        # --- coordinated strategies: saturation / waves / decoy ---
        speed_mult = np.ones((W, M), dtype=np.float32)
        t_now = self.step_count[:, None] * self.dt                      # (W,1)
        d0 = dist[..., 0]                                               # (W,M)
        sat = strat == _STRATEGIES.index("saturation")
        if sat.any():
            # throttle so the whole raid arrives simultaneously
            dmax = np.where(self.att_active, d0, 0.0).max(axis=1, keepdims=True)
            speed_mult = np.where(sat, np.clip(d0 / np.maximum(dmax, 1.0), 0.35, 1.0),
                                  speed_mult)
        wav = strat == _STRATEGIES.index("waves")
        if wav.any():
            wave = (a_idx % 3).astype(np.float32)
            speed_mult = np.where(wav & (t_now < wave * 8.0), 0.05, speed_mult)
        dec = strat == _STRATEGIES.index("decoy")
        if dec.any():
            u = to_target / dist
            is_decoy = (a_idx % 2 == 0) & dec & (t_now < 45.0)
            tang = np.stack([-u[..., 1], u[..., 0], np.zeros_like(u[..., 0])], -1)
            corr = np.clip((d0 - 250.0) / 60.0, -1.0, 1.0)[..., None]
            orbit = tang * 0.85 + u * corr * 0.5
            orbit /= np.maximum(np.linalg.norm(orbit, axis=-1, keepdims=True), 0.01)
            heading = np.where(is_decoy[..., None], orbit, heading)
            strike_hold = (a_idx % 2 == 1) & dec & (t_now < 4.0)
            speed_mult = np.where(strike_hold, 0.05, speed_mult)

        # obstacle avoidance: repulsion + climb near hills AND walls
        a_shell, a_below, centers = self._terrain_shell(self.att_pos)
        if a_shell is not None:
            near_idx = np.argmin(a_shell, axis=-1)                      # (W,M)
            min_shell = np.take_along_axis(a_shell, near_idx[..., None], -1)[..., 0]
            below_n = np.take_along_axis(a_below, near_idx[..., None], -1)[..., 0]
            sel = (min_shell < 45.0) & below_n & ~commit
            if sel.any():
                cen = np.take_along_axis(
                    centers[:, None, :, :].repeat(M, 1),
                    near_idx[..., None, None].repeat(3, -1), axis=2)[:, :, 0, :]
                away = self.att_pos[..., :2] - cen[..., :2]
                away /= np.maximum(np.linalg.norm(away, axis=-1, keepdims=True), 0.01)
                push = np.clip(1.0 - min_shell / 45.0, 0, 1)[..., None]
                avoid = np.concatenate([away * push * 0.9, 0.35 * push], axis=-1)
                adir = heading + np.where(sel[..., None], avoid, 0.0)
                adir /= np.maximum(np.linalg.norm(adir, axis=-1, keepdims=True), 0.01)
                heading = np.where(sel[..., None], adir, heading)

        vel = heading * (self.per_att_speed * speed_mult)[..., None]
        vel[~self.att_active] = 0.0
        return vel

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        c = self.cfg
        W, N, M = self.num_worlds, self.n_max, self.m_max

        # --- 1. commanded velocities ---
        if self.mode == "evade":
            # RL controls attackers; frozen model flies the defenders
            aacts = self._actions.reshape(W, M, 3)
            avel_cmd = np.clip(aacts, -1, 1) * c.attacker_speed_limit
            avel_cmd[~self.att_active] = 0.0
            if self.defender_model is not None:
                dacts, _ = self.defender_model.predict(self.def_obs_buf, deterministic=True)
                dvel = np.clip(dacts.reshape(W, N, 3), -1, 1) * c.defender_speed_limit
            else:
                dvel = np.zeros((W, N, 3), dtype=np.float32)
            dvel[~self.def_alive] = 0.0
        else:
            acts = self._actions.reshape(W, N, 3)
            dvel = np.clip(acts, -1, 1) * c.defender_speed_limit
            dvel[~self.def_alive] = 0.0
            avel_cmd = self._attacker_velocities()
            # 'rl' strategy worlds: frozen attacker model(s) override scripted guidance
            rl_worlds = self.world_strategy == _STRATEGIES.index("rl")
            if rl_worlds.any() and self.attacker_models:
                for mi, mdl in enumerate(self.attacker_models):
                    sel = rl_worlds & (self.rl_model_idx == mi)
                    if not sel.any():
                        continue
                    racts, _ = mdl.predict(self.att_obs_buf, deterministic=True)
                    rvel = np.clip(racts.reshape(W, M, 3), -1, 1) * c.attacker_speed_limit
                    rvel[~self.att_active] = 0.0
                    avel_cmd = np.where(sel[:, None, None], rvel, avel_cmd)

        vel = self._vel_np.reshape(W, self._num_drones, 3)
        vel[:, :N] = dvel
        vel[:, N:] = avel_cmd
        self.drone_vel = wp.array(self._vel_np, dtype=wp.vec3, device=self.device)

        # --- 2. wind (once per control step; steady component re-rolled per episode) ---
        if c.wind_enabled:
            gust = self._rng.random(W) < c.wind_gust_prob
            wind = self._wind_np.copy()
            if gust.any():
                ang = self._rng.uniform(0, 2 * np.pi, gust.sum())
                mag = self._rng.uniform(0.5, 1.0, gust.sum()) * c.wind_gust_force
                wind[gust, 0] += mag * np.cos(ang)
                wind[gust, 1] += mag * np.sin(ang)
                wind[gust, 2] += self._rng.uniform(-0.3, 0.3, gust.sum()) * c.wind_gust_force
            self.wind_wp = wp.array(wind, dtype=wp.vec3, device=self.device)

        # --- 3. physics substeps ---
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            wp.launch(
                aegis_force_kernel,
                dim=W * self._num_drones,
                inputs=(self.state_0.body_q, self.state_0.body_qd, self.drone_vel,
                        self.state_0.body_f, self._num_drones, self.mass_wp,
                        9.81, c.force_kp, c.force_kd, c.max_force,
                        c.angular_damping, self.wind_wp),
                device=self.device,
            )
            self.solver.step(self.state_0, self.state_1, None, None, self.sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # --- 4. one sync; refresh CPU mirrors ---
        wp.synchronize()
        body_q = self.state_0.body_q.numpy().reshape(W, self._num_drones, 7)
        body_qd = self.state_0.body_qd.numpy().reshape(W, self._num_drones, 6)
        self.def_pos_prev[:] = self.def_pos
        self.att_pos_prev[:] = self.att_pos
        self.def_pos = body_q[:, :N, :3].astype(np.float32)
        self.def_vel = body_qd[:, :N, :3].astype(np.float32)
        self.att_pos = body_q[:, N:, :3].astype(np.float32)
        self.att_vel = body_qd[:, N:, :3].astype(np.float32)
        self.step_count += 1

        # --- moving target (convoy): advance toward waypoint ---
        self.target_pos_prev[:] = self.target_pos
        moving = self.target_speed_w > 0
        if moving.any():
            to_wp = self.target_wp - self.target_pos[:, :2]
            d_wp = np.linalg.norm(to_wp, axis=-1)
            go = moving & (d_wp > 1e-6)
            stp = np.minimum(self.target_speed_w * self.dt, d_wp)
            dirv = to_wp / np.maximum(d_wp, 1e-6)[:, None]
            self.target_pos[:, :2] += np.where(go[:, None], dirv * stp[:, None], 0.0)
            self.target_vel[:, :2] = np.where(
                go[:, None], dirv * self.target_speed_w[:, None], 0.0)
            self.target_vel[:, 2] = 0.0
            for w in np.nonzero(moving & (d_wp < c.target_waypoint_tol))[0]:
                self.target_wp[w] = self._sample_target_wp(int(w))

        rewards = np.zeros((W, N), dtype=np.float32)
        dones = np.zeros((W, N), dtype=bool)
        infos = [{} for _ in range(self.n_envs)]
        alive_at_start = self.def_alive.copy()
        assign_old = self.assign.copy()

        # --- 5. swept intercepts (defender x attacker pairs) ---
        p0 = self.def_pos_prev[:, :, None, :] - self.att_pos_prev[:, None, :, :]
        p1 = self.def_pos[:, :, None, :] - self.att_pos[:, None, :, :]
        pair_min = segment_min_dist(p0, p1)                              # (W, N, M)
        att_intercepted_now = np.zeros((W, M), dtype=bool)
        att_breached_now = np.zeros((W, M), dtype=bool)
        att_terrain_now = np.zeros((W, M), dtype=bool)

        armed = self.rearm <= 0
        pair_hit = (pair_min < c.intercept_dist) & \
            (self.def_alive & armed)[:, :, None] & self.att_active[:, None, :]
        intercept_events = []                                            # (w, d, a)
        if pair_hit.any():
            ws, ds, as_ = np.nonzero(pair_hit)
            order = np.argsort(pair_min[ws, ds, as_])
            for k in order:
                w, d, a = int(ws[k]), int(ds[k]), int(as_[k])
                if self.def_alive[w, d] and self.att_active[w, a] and self.rearm[w, d] <= 0:
                    self.att_active[w, a] = False
                    if c.kamikaze:
                        self.def_alive[w, d] = False
                    else:
                        self.rearm[w, d] = c.intercept_cooldown_steps
                    att_intercepted_now[w, a] = True
                    intercept_events.append((w, d, a))
                    self.ep_intercepts[w] += 1
                    if self.ep_first_intercept[w] < 0:
                        self.ep_first_intercept[w] = int(self.step_count[w])
                    self.realloc_needed[w] = True

        # --- 6. swept breaches (attacker vs its world's target, target-relative
        #        frame so the check stays exact for a MOVING target) ---
        b0 = self.att_pos_prev - self.target_pos_prev[:, None, :]
        b1 = self.att_pos - self.target_pos[:, None, :]
        breach_min = segment_min_dist(b0, b1)                            # (W, M)
        breach_hit = (breach_min < c.target_reached_dist) & self.att_active
        breach_events = []
        if breach_hit.any():
            for w, a in zip(*np.nonzero(breach_hit)):
                self.att_active[w, a] = False
                self.att_breached[w, a] = True
                att_breached_now[w, a] = True
                breach_events.append((int(w), int(a)))
                self.realloc_needed[w] = True

        # --- 7. defender bad deaths + attacker terrain/OOB elimination ---
        oob = ((np.abs(self.def_pos[..., 0]) > c.arena_bounds) |
               (np.abs(self.def_pos[..., 1]) > c.arena_bounds) |
               (self.def_pos[..., 2] < c.altitude_min) |
               (self.def_pos[..., 2] > c.altitude_max))
        bad = oob & self.def_alive
        hill_danger = np.zeros((W, N), dtype=np.float32)   # for safety shaping
        d_shell, d_below, _ = self._terrain_shell(self.def_pos)
        if d_shell is not None:
            t_hit = ((d_shell < 5.0) & d_below).any(axis=-1)
            bad |= t_hit & self.def_alive
            danger = np.clip(1.0 - d_shell / 40.0, 0.0, 1.0)
            hill_danger = np.where(d_below, danger, 0.0).max(axis=-1)

            a_shell2, a_below2, _ = self._terrain_shell(self.att_pos)
            a_kill = ((a_shell2 < 5.0) & a_below2).any(axis=-1) & self.att_active
            if a_kill.any():
                self.att_terrain |= a_kill
                att_terrain_now |= a_kill
                self.att_active &= ~a_kill
                self.realloc_needed |= a_kill.any(axis=1)
        a_oob = ((np.abs(self.att_pos[..., 0]) > c.arena_bounds) |
                 (np.abs(self.att_pos[..., 1]) > c.arena_bounds) |
                 (self.att_pos[..., 2] < 0.5) |
                 (self.att_pos[..., 2] > c.altitude_max)) & self.att_active
        if a_oob.any():
            att_terrain_now |= a_oob
            self.att_active &= ~a_oob
            self.realloc_needed |= a_oob.any(axis=1)

        bad_events = list(zip(*np.nonzero(bad))) if bad.any() else []
        for w, d in bad_events:
            self.def_alive[w, d] = False
            self.ep_bad_deaths[w] += 1
            self.realloc_needed[w] = True

        # --- 8. exogenous culling (self-healing augmentation, zero reward) ---
        cull_events = []
        cull_now = (self.cull_step >= 0) & (self.step_count == self.cull_step)
        for w in np.nonzero(cull_now)[0]:
            for _ in range(max(1, c.cull_count)):
                alive = np.nonzero(self.def_alive[w])[0]
                if len(alive) <= 1:
                    break
                d = int(self._rng.choice(alive))
                self.def_alive[w, d] = False
                self.ep_culled[w] += 1
                cull_events.append((int(w), d))
                self.realloc_needed[w] = True

        # --- 9. detection + reallocation (self-healing: orphans re-owned NOW) ---
        self._update_detection()
        cadence = (self.step_count % max(c.realloc_every, 1) == 0)
        self._reallocate(self.realloc_needed | cadence)
        self.steps_since_reassign += 1

        if self.mode == "evade":
            return self._finish_evade(infos, att_intercepted_now, att_breached_now,
                                      att_terrain_now)

        # --- 10. rewards ---
        living = self.def_alive  # post-deaths
        # (a) potential shaping toward assigned target / guard station
        pot = self._pot_dist()
        delta = self.prev_pot_dist - pot
        interceptor = self.assign >= 0
        shaping = np.where(interceptor, c.r_close_coeff * delta,
                           c.r_guard_close_coeff * delta)
        near_target = np.linalg.norm(
            self.def_pos - self.target_pos[:, None, :], axis=-1) < 150.0
        shaping += np.where(~interceptor & near_target, c.r_guard_hold, 0.0)
        rewards += np.where(living, shaping, 0.0).astype(np.float32)
        self.prev_pot_dist = pot

        # (b) time penalty + safety shaping
        rewards += np.where(living, c.r_time, 0.0)
        bmargin = c.boundary_margin
        near_bound = np.clip(1.0 - (c.arena_bounds - np.abs(self.def_pos[..., 0])) / bmargin, 0, 1)
        near_bound = np.maximum(near_bound, np.clip(
            1.0 - (c.arena_bounds - np.abs(self.def_pos[..., 1])) / bmargin, 0, 1))
        near_bound = np.maximum(near_bound, np.clip(
            1.0 - (self.def_pos[..., 2] - c.altitude_min) / bmargin, 0, 1))
        near_bound = np.maximum(near_bound, np.clip(
            1.0 - (c.altitude_max - self.def_pos[..., 2]) / bmargin, 0, 1))
        rewards += np.where(living, c.r_safety_coeff * (hill_danger + near_bound), 0.0)

        self.rearm = np.maximum(self.rearm - 1, 0)

        # (c) death events -> terminal rewards + per-agent done
        # (kamikaze=False interceptors survive: paid but NOT done)
        for w, d, a in intercept_events:
            rewards[w, d] += c.r_intercept
            if c.kamikaze:
                dones[w, d] = True
        for w, d in bad_events:
            rewards[w, d] += c.r_bad_death
            dones[w, d] = True
        for w, d in cull_events:
            dones[w, d] = True
            infos[w * N + d]["culled"] = True

        # (d) breaches: owner-based penalty
        culled_set = set(cull_events)
        for w, a in breach_events:
            owners = np.nonzero((assign_old[w] == a) & alive_at_start[w])[0]
            if len(owners) > 0:
                for d in np.nonzero(living[w])[0]:
                    rewards[w, d] += (c.r_breach_owner if d in owners
                                      else c.r_breach_other)
                for d in owners:
                    # owner died this step: still fined — EXCEPT exogenous culls,
                    # which must pay exactly 0 (augmentation adds no penalty noise)
                    if not living[w, d] and dones[w, d] and (w, d) not in culled_set:
                        rewards[w, d] += c.r_breach_owner
            else:
                rewards[w, living[w]] += c.r_breach_unowned

        # --- 11. world termination ---
        resolved = ~self.att_active
        all_resolved = np.array([resolved[w, :self.m_w[w]].all() for w in range(W)])
        all_dead = np.array([~self.def_alive[w, :self.n_w[w]].any() for w in range(W)])
        timeout = self.step_count >= self.max_steps
        world_done = all_resolved | all_dead | timeout

        win = all_resolved & np.array(
            [not self.att_breached[w, :self.m_w[w]].any() for w in range(W)])
        for w in np.nonzero(win)[0]:
            rewards[w, living[w]] += c.r_team_win

        # --- 12. post-step observations (terminal-obs source), episode accounting ---
        self._compute_all_obs()
        self._ep_rew += np.where(~self.stream_dead, rewards, 0.0)
        self._ep_len += (~self.stream_dead).astype(np.int64)

        died_this_step = dones.copy()
        for w in range(W):
            wd = world_done[w]
            for d in range(N):
                i = w * N + d
                if self.stream_dead[w, d]:
                    # sentinel stream: ends silently when the world resets
                    if wd:
                        dones[w, d] = True
                    continue
                if died_this_step[w, d] or wd:
                    dones[w, d] = True
                    infos[i]["terminal_observation"] = self._agent_obs(w, d)
                    if wd and not died_this_step[w, d] and timeout[w] and not all_resolved[w]:
                        infos[i]["TimeLimit.truncated"] = True
                    infos[i]["episode"] = {"r": float(self._ep_rew[w, d]),
                                           "l": int(self._ep_len[w, d])}
                    if died_this_step[w, d] and not wd:
                        self.stream_dead[w, d] = True
                        self._ep_rew[w, d] = 0.0
                        self._ep_len[w, d] = 0

        # --- 13. record outcomes + auto-reset done worlds ---
        for w in np.nonzero(world_done)[0]:
            m = int(self.m_w[w])
            self.episode_outcomes.append({
                "n": int(self.n_w[w]), "m": m,
                "intercepts": int(self.ep_intercepts[w]),
                "breaches": int(self.att_breached[w, :m].sum()),
                "terrain_kills": int(self.att_terrain[w, :m].sum()),
                "bad_deaths": int(self.ep_bad_deaths[w]),
                "culled": int(self.ep_culled[w]),
                "win": bool(win[w]),
                # attackers still airborne when the episode ended (all defenders
                # dead or timeout) — the target is undefended, so these are
                # de-facto leakers for metrics even though no breach registered
                "stranded": int(self.att_active[w, :m].sum()),
                "timeout": bool(timeout[w] and not all_resolved[w]),
                "length": int(self.step_count[w]),
                "first_intercept": int(self.ep_first_intercept[w]),
                "defenders_left": int(self.def_alive[w, :self.n_w[w]].sum()),
            })
            self._spawn_world(w)
        if world_done.any():
            self._upload_resets(world_done)
            self._update_detection()
            self._reallocate(world_done)
            self._anchor_potentials(world_done[:, None] & np.ones((W, N), dtype=bool))
            # recompute so done worlds return their fresh-spawn initial obs
            self._compute_all_obs()

        return (self.obs_buf.copy(), rewards.reshape(-1).astype(np.float32),
                dones.reshape(-1), infos)

    def _finish_evade(self, infos, att_intercepted_now, att_breached_now,
                      att_terrain_now):
        """Evade-mode tail of step_wait: attacker rewards/dones/infos + resets.

        World termination differs from defend mode: all-defenders-dead does NOT
        end the episode — surviving attackers fly on to breach (that is the point
        of killing the defenders).
        """
        c = self.cfg
        W, M = self.num_worlds, self.m_max
        rewards = np.zeros((W, M), dtype=np.float32)
        dones = np.zeros((W, M), dtype=bool)
        active = self.att_active

        # potential shaping toward the target
        dist_t = np.linalg.norm(self.att_pos - self.target_pos[:, None, :], axis=-1)
        rewards += np.where(active, c.ar_close_coeff * (self.att_prev_dist - dist_t), 0.0)
        self.att_prev_dist = dist_t.astype(np.float32)

        rewards += np.where(active, c.ar_time, 0.0)
        a_shell, a_below, _ = self._terrain_shell(self.att_pos)
        if a_shell is not None:
            danger = np.clip(1.0 - a_shell / 40.0, 0.0, 1.0)
            rewards += np.where(active, c.ar_safety_coeff *
                                np.where(a_below, danger, 0.0).max(axis=-1), 0.0)

        rewards += np.where(att_breached_now, c.ar_breach, 0.0)
        rewards += np.where(att_intercepted_now, c.ar_intercepted, 0.0)
        rewards += np.where(att_terrain_now, c.ar_bad_death, 0.0)
        dones |= att_breached_now | att_intercepted_now | att_terrain_now

        resolved = ~self.att_active
        all_resolved = np.array([resolved[w, :self.m_w[w]].all() for w in range(W)])
        timeout = self.step_count >= self.max_steps
        world_done = all_resolved | timeout
        # stall penalty: attackers still airborne at timeout failed the mission
        stalled = timeout[:, None] & self.att_active
        rewards += np.where(stalled, c.ar_stall, 0.0)
        win = all_resolved & np.array(
            [not self.att_breached[w, :self.m_w[w]].any() for w in range(W)])

        # --- post-step obs, episode accounting, infos ---
        self._compute_all_obs()
        self._att_ep_rew += np.where(~self.att_stream_dead, rewards, 0.0)
        self._att_ep_len += (~self.att_stream_dead).astype(np.int64)
        died = dones.copy()
        for w in range(W):
            wd = world_done[w]
            for a in range(M):
                i = w * M + a
                if self.att_stream_dead[w, a]:
                    if wd:
                        dones[w, a] = True
                    continue
                if died[w, a] or wd:
                    dones[w, a] = True
                    infos[i]["terminal_observation"] = self.att_obs_buf[i].copy()
                    if wd and not died[w, a] and timeout[w] and not all_resolved[w]:
                        infos[i]["TimeLimit.truncated"] = True
                    infos[i]["episode"] = {"r": float(self._att_ep_rew[w, a]),
                                           "l": int(self._att_ep_len[w, a])}
                    if died[w, a] and not wd:
                        self.att_stream_dead[w, a] = True
                        self._att_ep_rew[w, a] = 0.0
                        self._att_ep_len[w, a] = 0

        # --- outcomes + auto-reset (defender-perspective record, same schema) ---
        for w in np.nonzero(world_done)[0]:
            m = int(self.m_w[w])
            self.episode_outcomes.append({
                "n": int(self.n_w[w]), "m": m,
                "intercepts": int(self.ep_intercepts[w]),
                "breaches": int(self.att_breached[w, :m].sum()),
                "terrain_kills": int(self.att_terrain[w, :m].sum()),
                "bad_deaths": int(self.ep_bad_deaths[w]),
                "culled": int(self.ep_culled[w]),
                "win": bool(win[w]),
                "stranded": int(self.att_active[w, :m].sum()),
                "timeout": bool(timeout[w] and not all_resolved[w]),
                "length": int(self.step_count[w]),
                "first_intercept": int(self.ep_first_intercept[w]),
                "defenders_left": int(self.def_alive[w, :self.n_w[w]].sum()),
            })
            self._spawn_world(w)
        if world_done.any():
            self._upload_resets(world_done)
            self._update_detection()
            self._reallocate(world_done)
            self._anchor_potentials(
                world_done[:, None] & np.ones((W, self.n_max), dtype=bool))
            self._compute_all_obs()

        return (self.obs_buf.copy(), rewards.reshape(-1).astype(np.float32),
                dones.reshape(-1), infos)

    # ------------------------------------------------------------------ #
    # Observations (plan §3.2) — batched
    # ------------------------------------------------------------------ #

    def _compute_obs(self):
        c = self.cfg
        W, N, M = self.num_worlds, self.n_max, self.m_max
        obs = np.zeros((W, N, OBS_DIM), dtype=np.float32)

        alive = self.def_alive
        n_alive = alive.sum(axis=1, keepdims=True).astype(np.float32)
        m_det = (self.att_active & self.att_detected).sum(axis=1, keepdims=True)

        rel_t = self.target_pos[:, None, :] - self.def_pos               # (W,N,3)
        dist_t = np.linalg.norm(rel_t, axis=-1, keepdims=True)

        obs[..., 0:3] = self.def_vel / 60.0
        obs[..., 3] = self.def_pos[..., 2] / 200.0
        obs[..., 4:7] = rel_t / 500.0
        obs[..., 7] = dist_t[..., 0] / 500.0
        obs[..., 8] = (n_alive / np.maximum(self.n_w[:, None], 1))
        obs[..., 9] = (m_det / np.maximum(self.m_w[:, None], 1))

        # --- assigned block [10:26] ---
        has_t = self.assign >= 0
        a_idx = np.clip(self.assign, 0, None)[..., None]
        goal_pos = np.where(has_t[..., None],
                            np.take_along_axis(self.att_pos, a_idx, axis=1),
                            self.stations)
        goal_vel = np.where(has_t[..., None],
                            np.take_along_axis(self.att_vel, a_idx, axis=1),
                            0.0)
        rel = goal_pos - self.def_pos
        d = np.linalg.norm(rel, axis=-1)
        relv = goal_vel - self.def_vel
        safe_d = np.maximum(d, 0.01)
        los = rel / safe_d[..., None]
        closing = -np.sum(rel * relv, axis=-1) / safe_d
        goal_dist_t = np.linalg.norm(goal_pos - self.target_pos[:, None, :], axis=-1)
        tgo = assigned_tgo(self.def_pos, c.defender_speed_limit,
                           self.att_pos, self.att_vel, self.assign)
        pip = goal_pos + goal_vel * tgo[..., None]

        obs[..., 10:13] = rel / 500.0
        obs[..., 13:16] = relv / 120.0
        obs[..., 16] = d / 500.0
        obs[..., 17] = closing / 120.0
        obs[..., 18:21] = los
        obs[..., 21] = goal_dist_t / 500.0
        obs[..., 22:25] = 0.0 if c.obs_zero_pip else (pip - self.def_pos) / 500.0
        obs[..., 25] = 1.0

        # --- assignment meta [26:29] ---
        obs[..., 26] = has_t.astype(np.float32)
        obs[..., 27] = np.clip(self.steps_since_reassign * self.dt / 5.0, 0, 1)
        share = np.zeros((W, N), dtype=np.float32)
        for a in range(M):
            cnt = ((self.assign == a) & alive).sum(axis=1, keepdims=True)
            share = np.where(self.assign == a, np.maximum(cnt - 1, 0), share)
        obs[..., 28] = share / 4.0

        # default sentinels: absent teammate/hill slots read as "far away"
        obs[..., 32] = 1.0
        obs[..., 36] = 1.0
        obs[..., 40] = 1.0
        obs[..., 44] = 1.0

        # --- 2 nearest living teammates [29:37] ---
        dd = self.def_pos[:, :, None, :] - self.def_pos[:, None, :, :]
        ddist = np.linalg.norm(dd, axis=-1)                              # (W,N,N)
        eye = np.eye(N, dtype=bool)[None]
        invalid = eye | ~alive[:, None, :]
        ddist = np.where(invalid, np.inf, ddist)
        nearest2 = np.argsort(ddist, axis=-1)[..., :2]                   # (W,N,<=2)
        for k in range(min(2, max(N - 1, 0))):
            idx = nearest2[..., k]
            tdist = np.take_along_axis(ddist, idx[..., None], axis=-1)[..., 0]
            trel = np.take_along_axis(
                self.def_pos[:, None, :, :].repeat(N, 1), idx[..., None, None]
                .repeat(3, -1), axis=2)[:, :, 0, :] - self.def_pos
            valid = np.isfinite(tdist)
            base = 29 + k * 4
            obs[..., base:base + 3] = np.where(
                valid[..., None], np.clip(trel / 100.0, -1, 1), 0.0)
            obs[..., base + 3] = np.where(valid, np.clip(tdist / 100.0, 0, 1), 1.0)

        # --- 2 nearest obstacles (hills or walls) [37:45] ---
        t_shell, _, t_cen = self._terrain_shell(self.def_pos)
        if t_shell is not None:
            near = np.argsort(t_shell, axis=-1)[..., :2]
            for k in range(min(2, t_shell.shape[-1])):
                idx = near[..., k]
                sh = np.take_along_axis(t_shell, idx[..., None], axis=-1)[..., 0]
                ok = np.isfinite(sh)
                cen = np.take_along_axis(
                    t_cen[:, None, :, :].repeat(N, 1),
                    idx[..., None, None].repeat(3, -1), axis=2)[:, :, 0, :]
                r = cen - self.def_pos
                base = 37 + k * 4
                obs[..., base:base + 3] = np.where(
                    ok[..., None], np.clip(r / 200.0, -1, 1), 0.0)
                obs[..., base + 3] = np.where(ok, np.clip(sh / 200.0, 0, 1), 1.0)

        # --- boundary [45:48] ---
        obs[..., 45] = (c.arena_bounds - np.abs(self.def_pos[..., 0])) / 500.0
        obs[..., 46] = (c.arena_bounds - np.abs(self.def_pos[..., 1])) / 500.0
        obs[..., 47] = np.minimum(self.def_pos[..., 2] - c.altitude_min,
                                  c.altitude_max - self.def_pos[..., 2]) / 200.0

        # --- nearest other attacker [48:56] ---
        if c.obs_opportunistic:
            adiff = self.att_pos[:, None, :, :] - self.def_pos[:, :, None, :]
            adist = np.linalg.norm(adiff, axis=-1)                       # (W,N,M)
            mask = (self.att_active & self.att_detected)[:, None, :].repeat(N, 1)
            own = np.zeros((W, N, M), dtype=bool)
            np.put_along_axis(own, np.clip(self.assign, 0, None)[..., None],
                              (self.assign >= 0)[..., None], axis=-1)
            adist = np.where(mask & ~own, adist, np.inf)
            ai = np.argmin(adist, axis=-1)                               # (W,N)
            avalid = np.isfinite(np.take_along_axis(adist, ai[..., None], axis=-1)[..., 0])
            apos = np.take_along_axis(self.att_pos, ai[..., None], axis=1)
            avel = np.take_along_axis(self.att_vel, ai[..., None], axis=1)
            arel = apos - self.def_pos
            arelv = avel - self.def_vel
            obs[..., 48:51] = np.where(avalid[..., None], arel / 500.0, 0.0)
            obs[..., 51:54] = np.where(avalid[..., None], arelv / 120.0, 0.0)
            obs[..., 54] = np.where(avalid,
                                    np.linalg.norm(arel, axis=-1) / 500.0, 0.0)
            obs[..., 55] = avalid.astype(np.float32)

        # undetected assigned target -> zero the assigned block (flag 0). Assigned
        # implies detected under latched detection, so only guards w/o station... keep
        # the invariant explicit anyway:
        undet = has_t & ~np.take_along_axis(self.att_detected, np.clip(self.assign, 0, None), axis=1)
        if undet.any():
            blk = obs[..., 10:26]
            blk[undet] = 0.0

        if c.obs_noise_std > 0:
            obs += self._rng.normal(0, c.obs_noise_std, obs.shape).astype(np.float32)

        # dead/unused slots: all-zero sentinel
        obs[~alive] = 0.0
        self.def_obs_buf = obs.reshape(-1, OBS_DIM)

    def _compute_att_obs(self):
        """Attacker-centric obs (evade mode / 'rl' strategy), batched. ATT_OBS_DIM=41."""
        c = self.cfg
        W, N, M = self.num_worlds, self.n_max, self.m_max
        obs = np.zeros((W, M, ATT_OBS_DIM), dtype=np.float32)
        active = self.att_active

        rel_t = self.target_pos[:, None, :] - self.att_pos
        dist_t = np.linalg.norm(rel_t, axis=-1)
        obs[..., 0:3] = self.att_vel / 50.0
        obs[..., 3] = self.att_pos[..., 2] / 200.0
        obs[..., 4:7] = rel_t / 500.0
        obs[..., 7] = dist_t / 500.0
        obs[..., 8] = active.sum(axis=1, keepdims=True) / np.maximum(self.m_w[:, None], 1)
        obs[..., 9] = self.def_alive.sum(axis=1, keepdims=True) / np.maximum(self.n_w[:, None], 1)

        # 3 nearest ALIVE defenders: rel pos /500 + closing rate /120
        dd = self.def_pos[:, None, :, :] - self.att_pos[:, :, None, :]   # (W,M,N,3)
        ddist = np.linalg.norm(dd, axis=-1)
        ddist = np.where(self.def_alive[:, None, :], ddist, np.inf)
        order = np.argsort(ddist, axis=-1)[..., :3]
        for k in range(min(3, N)):
            idx = order[..., k]
            dist_k = np.take_along_axis(ddist, idx[..., None], axis=-1)[..., 0]
            valid = np.isfinite(dist_k)
            rel = np.take_along_axis(
                self.def_pos[:, None, :, :].repeat(M, 1),
                idx[..., None, None].repeat(3, -1), axis=2)[:, :, 0, :] - self.att_pos
            dvel = np.take_along_axis(
                self.def_vel[:, None, :, :].repeat(M, 1),
                idx[..., None, None].repeat(3, -1), axis=2)[:, :, 0, :]
            relv = dvel - self.att_vel
            safe = np.maximum(dist_k, 0.01)
            closing = -np.sum(rel * relv, axis=-1) / safe
            base = 10 + k * 4
            obs[..., base:base + 3] = np.where(valid[..., None], rel / 500.0, 0.0)
            obs[..., base + 3] = np.where(valid, closing / 120.0, 0.0)

        # 2 nearest active teammates: rel /100 clip + dist /100 (sentinel dist 1.0)
        obs[..., 25] = 1.0
        obs[..., 29] = 1.0
        aa = self.att_pos[:, :, None, :] - self.att_pos[:, None, :, :]
        adist = np.linalg.norm(aa, axis=-1)
        eye = np.eye(M, dtype=bool)[None]
        adist = np.where(eye | ~active[:, None, :], np.inf, adist)
        near2 = np.argsort(adist, axis=-1)[..., :2]
        for k in range(min(2, max(M - 1, 0))):
            idx = near2[..., k]
            dist_k = np.take_along_axis(adist, idx[..., None], axis=-1)[..., 0]
            valid = np.isfinite(dist_k)
            rel = np.take_along_axis(
                self.att_pos[:, None, :, :].repeat(M, 1),
                idx[..., None, None].repeat(3, -1), axis=2)[:, :, 0, :] - self.att_pos
            base = 22 + k * 4
            obs[..., base:base + 3] = np.where(valid[..., None],
                                               np.clip(rel / 100.0, -1, 1), 0.0)
            obs[..., base + 3] = np.where(valid, np.clip(dist_k / 100.0, 0, 1), 1.0)

        # 2 nearest obstacles (hills or walls; sentinel shell dist 1.0) + boundary
        obs[..., 33] = 1.0
        obs[..., 37] = 1.0
        t_shell, _, t_cen = self._terrain_shell(self.att_pos)
        if t_shell is not None:
            near = np.argsort(t_shell, axis=-1)[..., :2]
            for k in range(min(2, t_shell.shape[-1])):
                idx = near[..., k]
                sh = np.take_along_axis(t_shell, idx[..., None], axis=-1)[..., 0]
                ok = np.isfinite(sh)
                cen = np.take_along_axis(
                    t_cen[:, None, :, :].repeat(M, 1),
                    idx[..., None, None].repeat(3, -1), axis=2)[:, :, 0, :]
                r = cen - self.att_pos
                base = 30 + k * 4
                obs[..., base:base + 3] = np.where(
                    ok[..., None], np.clip(r / 200.0, -1, 1), 0.0)
                obs[..., base + 3] = np.where(ok, np.clip(sh / 200.0, 0, 1), 1.0)
        obs[..., 38] = (c.arena_bounds - np.abs(self.att_pos[..., 0])) / 500.0
        obs[..., 39] = (c.arena_bounds - np.abs(self.att_pos[..., 1])) / 500.0
        obs[..., 40] = np.minimum(self.att_pos[..., 2] - 0.5,
                                  c.altitude_max - self.att_pos[..., 2]) / 200.0

        if c.obs_noise_std > 0:
            obs += self._rng.normal(0, c.obs_noise_std, obs.shape).astype(np.float32)
        obs[~active] = 0.0
        self.att_obs_buf = obs.reshape(-1, ATT_OBS_DIM)

    def _compute_all_obs(self):
        """Refresh def/att obs buffers and point obs_buf at the RL-facing one."""
        self._compute_obs()
        if self.mode == "evade" or self.attacker_models:
            self._compute_att_obs()
        self.obs_buf = (self.att_obs_buf if self.mode == "evade"
                        else self.def_obs_buf)

    def _agent_obs(self, w, d):
        """Terminal observation for one defender (post-step def obs)."""
        return self.def_obs_buf[w * self.n_max + d].copy()

    # ------------------------------------------------------------------ #
    # Utilities / VecEnv API
    # ------------------------------------------------------------------ #

    def pop_outcomes(self):
        out = self.episode_outcomes
        self.episode_outcomes = []
        return out

    def effective_sample_fraction(self):
        if self.mode == "evade":
            return float(self.att_active.mean())
        return float(self.def_alive.mean())

    def seed(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        return [seed] * self.n_envs

    def close(self):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.n_envs

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs)]

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.n_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)


_STRATEGIES = ("direct", "flanking", "mixed", "jink", "rl",
               "saturation", "waves", "decoy")
_TARGET_TERRAIN_CAP = {"urban": 60.0, "maze": 0.0}
