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

    # Timing: policy 12 Hz, physics 240 Hz
    ctrl_freq: int = 12
    phys_freq: int = 240
    episode_secs: float = 60.0

    # Force model
    force_kp: float = 50.0
    force_kd: float = 15.0
    max_force: float = 150.0
    angular_damping: float = 5.0

    # Hills (analytic — no physics bodies; distance-based kill/shaping)
    num_obstacles: int = 0
    obstacle_radius: float = 50.0
    obstacle_height: float = 60.0      # half-height; top of hill = 2*obstacle_height
    obstacle_placement_range: float = 350.0
    obstacle_min_spacing: float = 120.0
    obstacle_collision_radius: float = 55.0   # horizontal kill radius
    obstacle_danger_radius: float = 70.0      # safety shaping starts here

    # Sensing (global shared detection, latched once seen)
    sensor_range: float = 350.0        # 0 = unlimited
    obs_noise_std: float = 0.0
    obs_opportunistic: bool = True     # 8-dim nearest-other-attacker block

    # Wind
    wind_enabled: bool = False
    wind_max_force: float = 0.5
    wind_change_freq: int = 12         # control steps between wind re-rolls
    wind_gust_prob: float = 0.0
    wind_gust_force: float = 1.5

    # Attackers
    attacker_strategies: tuple = ("direct",)   # sampled per world per episode

    # Self-healing augmentation: exogenous zero-reward defender culls
    cull_prob: float = 0.0             # per-episode probability
    cull_step_range: tuple = (120, 480)

    # Allocation
    realloc_every: int = 6             # control steps (0.5 s at 12 Hz)
    guard_ring_radius: float = 100.0

    # Rewards (plan §3.4) — ONE table for all stages
    r_close_coeff: float = 0.03        # per meter toward assigned attacker
    r_guard_close_coeff: float = 0.01  # per meter toward ring station
    r_guard_hold: float = 0.005        # per step within 150 m of target
    r_intercept: float = 30.0
    r_bad_death: float = -15.0
    r_breach_owner: float = -25.0
    r_breach_other: float = -5.0
    r_breach_unowned: float = -10.0
    r_team_win: float = 10.0
    r_time: float = -0.01
    r_safety_coeff: float = -0.02
    boundary_margin: float = 25.0

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
        self.n_envs = num_worlds * self.n_max

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

        # --- allocation state ---
        self.assign = np.full((W, N), GUARD, dtype=np.int64)
        self.stations = np.zeros((W, N, 3), dtype=np.float32)
        self.prev_pot_dist = np.zeros((W, N), dtype=np.float32)  # shaping anchor
        self.steps_since_reassign = np.zeros((W, N), dtype=np.int64)
        self.realloc_needed = np.ones(W, dtype=bool)

        # --- episode bookkeeping ---
        self._ep_rew = np.zeros((W, N), dtype=np.float64)
        self._ep_len = np.zeros((W, N), dtype=np.int64)
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

        self.obs_buf = np.zeros((self.n_envs, OBS_DIM), dtype=np.float32)

        obs_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        super().__init__(self.n_envs, obs_space, act_space)

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
        self.jink_phase[w] = rng.uniform(0, 2 * np.pi, self.m_max).astype(np.float32)

        self.cull_step[w] = -1
        if c.cull_prob > 0 and rng.random() < c.cull_prob:
            self.cull_step[w] = int(rng.integers(*c.cull_step_range))

        if c.num_obstacles > 0:
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

        if c.wind_enabled:
            ang = rng.uniform(0, 2 * np.pi)
            mag = rng.uniform(0, c.wind_max_force)
            self._wind_np[w] = (mag * np.cos(ang), mag * np.sin(ang), 0.0)

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
        self.def_vel[idx] = 0.0
        self.att_vel[idx] = 0.0

    def reset(self):
        for w in range(self.num_worlds):
            self._spawn_world(w)
        self._upload_resets(np.ones(self.num_worlds, dtype=bool))
        self._update_detection()
        self._reallocate(np.ones(self.num_worlds, dtype=bool))
        self._anchor_potentials(np.ones((self.num_worlds, self.n_max), dtype=bool))
        self._compute_obs()
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
            prev_assign=self.assign[idx],
        )
        new_assign = self.assign.copy()
        new_assign[idx] = sub_assign
        changed = (new_assign != self.assign) & world_mask[:, None]
        self.assign = new_assign
        self.stations = guard_stations(self.assign, self.def_alive, self.target_pos,
                                       self.cfg.guard_ring_radius)
        self.steps_since_reassign[changed] = 0
        self._anchor_potentials(changed)
        self.realloc_needed[world_mask] = False

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

        # hill avoidance: repulsion + climb near hills
        if c.num_obstacles > 0:
            hill = self.obstacle_pos[:, None, :, :]                     # (W,1,O,3)
            d_h = self.att_pos[:, :, None, :2] - hill[..., :2]
            hd = np.linalg.norm(d_h, axis=-1)                           # (W,M,O)
            near = (hd < 80.0) & (self.att_pos[:, :, None, 2] < 2 * c.obstacle_height + 10)
            if near.any():
                o = np.argmin(np.where(near, hd, np.inf), axis=-1)      # (W,M)
                sel = near.any(axis=-1)
                away = np.take_along_axis(d_h, o[..., None, None], axis=2)[:, :, 0, :]
                away /= np.maximum(np.linalg.norm(away, axis=-1, keepdims=True), 0.01)
                hd_min = np.take_along_axis(hd, o[..., None], axis=2)[..., 0]
                push = (1.0 - hd_min / 80.0)[..., None]
                avoid = np.concatenate([away * push * 0.8,
                                        np.full((*push.shape[:2], 1), 0.3) * push], axis=-1)
                adir = heading + np.where(sel[..., None], avoid, 0.0)
                adir /= np.maximum(np.linalg.norm(adir, axis=-1, keepdims=True), 0.01)
                heading = np.where(sel[..., None], adir, heading)

        vel = heading * self.per_att_speed[..., None]
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
        acts = self._actions.reshape(W, N, 3)
        dvel = np.clip(acts, -1, 1) * c.defender_speed_limit
        dvel[~self.def_alive] = 0.0
        avel_cmd = self._attacker_velocities()

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

        rewards = np.zeros((W, N), dtype=np.float32)
        dones = np.zeros((W, N), dtype=bool)
        infos = [{} for _ in range(self.n_envs)]
        alive_at_start = self.def_alive.copy()
        assign_old = self.assign.copy()

        # --- 5. swept intercepts (defender x attacker pairs) ---
        p0 = self.def_pos_prev[:, :, None, :] - self.att_pos_prev[:, None, :, :]
        p1 = self.def_pos[:, :, None, :] - self.att_pos[:, None, :, :]
        pair_min = segment_min_dist(p0, p1)                              # (W, N, M)
        pair_hit = (pair_min < c.intercept_dist) & \
            self.def_alive[:, :, None] & self.att_active[:, None, :]
        intercept_events = []                                            # (w, d, a)
        if pair_hit.any():
            ws, ds, as_ = np.nonzero(pair_hit)
            order = np.argsort(pair_min[ws, ds, as_])
            for k in order:
                w, d, a = int(ws[k]), int(ds[k]), int(as_[k])
                if self.def_alive[w, d] and self.att_active[w, a]:
                    self.att_active[w, a] = False
                    if c.kamikaze:
                        self.def_alive[w, d] = False
                    intercept_events.append((w, d, a))
                    self.ep_intercepts[w] += 1
                    if self.ep_first_intercept[w] < 0:
                        self.ep_first_intercept[w] = int(self.step_count[w])
                    self.realloc_needed[w] = True

        # --- 6. swept breaches (attacker vs its world's target) ---
        b0 = self.att_pos_prev - self.target_pos[:, None, :]
        b1 = self.att_pos - self.target_pos[:, None, :]
        breach_min = segment_min_dist(b0, b1)                            # (W, M)
        breach_hit = (breach_min < c.target_reached_dist) & self.att_active
        breach_events = []
        if breach_hit.any():
            for w, a in zip(*np.nonzero(breach_hit)):
                self.att_active[w, a] = False
                self.att_breached[w, a] = True
                breach_events.append((int(w), int(a)))
                self.realloc_needed[w] = True

        # --- 7. defender bad deaths + attacker terrain/OOB elimination ---
        oob = ((np.abs(self.def_pos[..., 0]) > c.arena_bounds) |
               (np.abs(self.def_pos[..., 1]) > c.arena_bounds) |
               (self.def_pos[..., 2] < c.altitude_min) |
               (self.def_pos[..., 2] > c.altitude_max))
        bad = oob & self.def_alive
        hill_danger = np.zeros((W, N), dtype=np.float32)   # for safety shaping
        if c.num_obstacles > 0:
            hd = np.linalg.norm(
                self.def_pos[:, :, None, :2] - self.obstacle_pos[:, None, :, :2],
                axis=-1)                                                 # (W,N,O)
            below_top = self.def_pos[..., 2, None] < 2 * c.obstacle_height
            hill_hit = ((hd < c.obstacle_collision_radius) & below_top).any(axis=-1)
            bad |= hill_hit & self.def_alive
            danger = np.clip(1.0 - hd / c.obstacle_danger_radius, 0.0, 1.0)
            hill_danger = np.where(below_top, danger, 0.0).max(axis=-1)

            ahd = np.linalg.norm(
                self.att_pos[:, :, None, :2] - self.obstacle_pos[:, None, :, :2],
                axis=-1)
            a_below = self.att_pos[..., 2, None] < 2 * c.obstacle_height
            a_hill = ((ahd < c.obstacle_collision_radius) & a_below).any(axis=-1)
            a_kill = a_hill & self.att_active
            if a_kill.any():
                self.att_terrain |= a_kill
                self.att_active &= ~a_kill
                self.realloc_needed |= a_kill.any(axis=1)
        a_oob = ((np.abs(self.att_pos[..., 0]) > c.arena_bounds) |
                 (np.abs(self.att_pos[..., 1]) > c.arena_bounds) |
                 (self.att_pos[..., 2] < 0.5) |
                 (self.att_pos[..., 2] > c.altitude_max)) & self.att_active
        if a_oob.any():
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
            alive = np.nonzero(self.def_alive[w])[0]
            if len(alive) > 1:
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

        # (c) death events -> terminal rewards + per-agent done
        for w, d, a in intercept_events:
            rewards[w, d] += c.r_intercept
            dones[w, d] = True
        for w, d in bad_events:
            rewards[w, d] += c.r_bad_death
            dones[w, d] = True
        for w, d in cull_events:
            dones[w, d] = True
            infos[w * N + d]["culled"] = True

        # (d) breaches: owner-based penalty
        for w, a in breach_events:
            owners = np.nonzero((assign_old[w] == a) & alive_at_start[w])[0]
            if len(owners) > 0:
                for d in np.nonzero(living[w])[0]:
                    rewards[w, d] += (c.r_breach_owner if d in owners
                                      else c.r_breach_other)
                for d in owners:
                    if not living[w, d] and dones[w, d]:
                        rewards[w, d] += c.r_breach_owner  # owner died this step
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
        self._compute_obs()
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
            self._compute_obs()

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
        obs[..., 22:25] = (pip - self.def_pos) / 500.0
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

        # --- 2 nearest hills [37:45] ---
        if c.num_obstacles > 0:
            hrel = self.obstacle_pos[:, None, :, :] - self.def_pos[:, :, None, :]
            hd = np.linalg.norm(hrel[..., :2], axis=-1)                  # (W,N,O)
            near = np.argsort(hd, axis=-1)[..., :2]                      # (W,N,<=2)
            for k in range(min(2, hd.shape[-1])):
                idx = near[..., k]
                r = np.take_along_axis(hrel, idx[..., None, None].repeat(3, -1),
                                       axis=2)[:, :, 0, :]
                shell = np.take_along_axis(hd, idx[..., None], axis=-1)[..., 0] \
                    - c.obstacle_radius
                base = 37 + k * 4
                obs[..., base:base + 3] = np.clip(r / 200.0, -1, 1)
                obs[..., base + 3] = np.clip(shell / 200.0, 0, 1)

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
        self.obs_buf = obs.reshape(self.n_envs, OBS_DIM)

    def _agent_obs(self, w, d):
        """Terminal observation for one agent (already computed this step's obs)."""
        return self.obs_buf[w * self.n_max + d].copy()

    # ------------------------------------------------------------------ #
    # Utilities / VecEnv API
    # ------------------------------------------------------------------ #

    def pop_outcomes(self):
        out = self.episode_outcomes
        self.episode_outcomes = []
        return out

    def effective_sample_fraction(self):
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


_STRATEGIES = ("direct", "flanking", "mixed", "jink")
