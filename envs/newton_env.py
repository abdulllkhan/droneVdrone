"""
Unified GPU-Accelerated Drone Environment (Newton Physics)

Supports multiple training modes with a consistent 19-dim observation
and 3-dim action space, enabling progressive training with weight transfer:

  fly_to_target — Single drone flies to random static targets
  chase         — Single drone chases a moving target
  intercept     — Defender vs scripted/frozen attacker (original scenario)
  adversarial   — Both drones are RL-controlled (future)

Force model: Direct world-frame forces (no propeller simulation).
Compatible with Stable Baselines3 via VecEnv interface.
"""

import numpy as np
import warp as wp
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from train.config import EnvConfig


# ------------------------------------------------------------------ #
# Warp Kernels (run on GPU)
# ------------------------------------------------------------------ #

@wp.kernel
def apply_velocity_force(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_vel: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
    bodies_per_world: int,
    body_offset: int,
    mass: float,
    gravity: float,
    kp: float,
    kd: float,
    max_force: float,
    angular_damping: float,
):
    """Apply world-frame PD forces to track a desired velocity."""
    world_id = wp.tid()
    body_idx = world_id * bodies_per_world + body_offset

    vel = wp.vec3(
        body_qd[body_idx][0],
        body_qd[body_idx][1],
        body_qd[body_idx][2],
    )
    desired = target_vel[world_id]

    vel_error = desired - vel

    # Control force (PD tracking) — this gets clamped
    ctrl_x = vel_error[0] * kp * mass - vel[0] * kd * mass
    ctrl_y = vel_error[1] * kp * mass - vel[1] * kd * mass
    ctrl_z = vel_error[2] * kp * mass - vel[2] * kd * mass

    ctrl = wp.vec3(ctrl_x, ctrl_y, ctrl_z)
    mag = wp.length(ctrl)
    if mag > max_force:
        ctrl = ctrl * (max_force / mag)

    # Gravity compensation — ALWAYS applied in full, never clamped
    force = wp.vec3(ctrl[0], ctrl[1], ctrl[2] + mass * gravity)

    ang_vel = wp.vec3(
        body_qd[body_idx][3],
        body_qd[body_idx][4],
        body_qd[body_idx][5],
    )
    torque = ang_vel * (-angular_damping)

    wp.atomic_add(body_f, body_idx, wp.spatial_vector(force, torque))


@wp.kernel
def attacker_pursuit(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_pos: wp.vec3,
    attacker_vel_out: wp.array(dtype=wp.vec3),
    bodies_per_world: int,
    attacker_body_offset: int,
    speed: float,
):
    """Scripted attacker: fly straight toward target."""
    world_id = wp.tid()
    body_idx = world_id * bodies_per_world + attacker_body_offset

    pos = wp.transform_get_translation(body_q[body_idx])
    to_target = target_pos - pos
    dist = wp.length(to_target)

    vel = wp.vec3(0.0, 0.0, 0.0)
    if dist > 0.01:
        vel = to_target * (speed / dist)
    attacker_vel_out[world_id] = vel


@wp.kernel
def compute_obs_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_pos: wp.vec3,
    obs_out: wp.array(dtype=float, ndim=2),
    bodies_per_world: int,
    defender_offset: int,
    attacker_offset: int,
):
    """Compute 19-dim observation for each world (intercept mode).
    All values manually normalized to roughly [-1, 1]."""
    world_id = wp.tid()
    def_idx = world_id * bodies_per_world + defender_offset
    att_idx = world_id * bodies_per_world + attacker_offset

    def_pos = wp.transform_get_translation(body_q[def_idx])
    att_pos = wp.transform_get_translation(body_q[att_idx])
    def_vel = wp.vec3(body_qd[def_idx][0], body_qd[def_idx][1], body_qd[def_idx][2])
    att_vel = wp.vec3(body_qd[att_idx][0], body_qd[att_idx][1], body_qd[att_idx][2])
    rel_pos = att_pos - def_pos
    dist = wp.length(rel_pos)

    pos_scale = 0.25    # 1/4.0 (arena bounds)
    vel_scale = 0.2     # 1/5.0 (max velocity)
    rel_scale = 1.0 / 6.0  # max distance ~6m

    obs_out[world_id, 0] = def_pos[0] * pos_scale
    obs_out[world_id, 1] = def_pos[1] * pos_scale
    obs_out[world_id, 2] = def_pos[2] * pos_scale
    obs_out[world_id, 3] = def_vel[0] * vel_scale
    obs_out[world_id, 4] = def_vel[1] * vel_scale
    obs_out[world_id, 5] = def_vel[2] * vel_scale
    obs_out[world_id, 6] = att_pos[0] * pos_scale
    obs_out[world_id, 7] = att_pos[1] * pos_scale
    obs_out[world_id, 8] = att_pos[2] * pos_scale
    obs_out[world_id, 9] = att_vel[0] * vel_scale
    obs_out[world_id, 10] = att_vel[1] * vel_scale
    obs_out[world_id, 11] = att_vel[2] * vel_scale
    obs_out[world_id, 12] = rel_pos[0] * rel_scale
    obs_out[world_id, 13] = rel_pos[1] * rel_scale
    obs_out[world_id, 14] = rel_pos[2] * rel_scale
    obs_out[world_id, 15] = dist * rel_scale
    obs_out[world_id, 16] = target_pos[0] * pos_scale
    obs_out[world_id, 17] = target_pos[1] * pos_scale
    obs_out[world_id, 18] = target_pos[2] * pos_scale


@wp.kernel
def compute_obs_single_drone_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_positions: wp.array(dtype=wp.vec3),
    target_velocities: wp.array(dtype=wp.vec3),
    obs_out: wp.array(dtype=float, ndim=2),
    bodies_per_world: int,
    drone_offset: int,
):
    """
    Compute 19-dim observation for fly_to_target / chase modes.

    Maps target info into the same slots as attacker info so weights
    transfer seamlessly to intercept mode:
      [0:3]   drone pos
      [3:6]   drone vel
      [6:9]   target pos (= attacker pos slot)
      [9:12]  target vel (= attacker vel slot, zero for static target)
      [12:15] relative pos (target - drone)
      [15]    distance to target
      [16:19] zeros (no defended target in this mode)
    """
    world_id = wp.tid()
    body_idx = world_id * bodies_per_world + drone_offset

    pos = wp.transform_get_translation(body_q[body_idx])
    vel = wp.vec3(body_qd[body_idx][0], body_qd[body_idx][1], body_qd[body_idx][2])
    tgt = target_positions[world_id]
    tgt_vel = target_velocities[world_id]
    rel = tgt - pos
    dist = wp.length(rel)

    pos_scale = 0.25    # 1/4.0 (arena bounds)
    vel_scale = 0.2     # 1/5.0 (max velocity)
    rel_scale = 1.0 / 6.0  # max distance ~6m

    obs_out[world_id, 0] = pos[0] * pos_scale
    obs_out[world_id, 1] = pos[1] * pos_scale
    obs_out[world_id, 2] = pos[2] * pos_scale
    obs_out[world_id, 3] = vel[0] * vel_scale
    obs_out[world_id, 4] = vel[1] * vel_scale
    obs_out[world_id, 5] = vel[2] * vel_scale
    obs_out[world_id, 6] = tgt[0] * pos_scale
    obs_out[world_id, 7] = tgt[1] * pos_scale
    obs_out[world_id, 8] = tgt[2] * pos_scale
    obs_out[world_id, 9] = tgt_vel[0] * vel_scale
    obs_out[world_id, 10] = tgt_vel[1] * vel_scale
    obs_out[world_id, 11] = tgt_vel[2] * vel_scale
    obs_out[world_id, 12] = rel[0] * rel_scale
    obs_out[world_id, 13] = rel[1] * rel_scale
    obs_out[world_id, 14] = rel[2] * rel_scale
    obs_out[world_id, 15] = dist * rel_scale
    obs_out[world_id, 16] = 0.0
    obs_out[world_id, 17] = 0.0
    obs_out[world_id, 18] = 0.0


@wp.kernel
def compute_obs_evade_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_pos: wp.vec3,
    obs_out: wp.array(dtype=float, ndim=2),
    bodies_per_world: int,
    attacker_offset: int,
    defender_offset: int,
):
    """Compute 19-dim observation from attacker perspective (evade mode).
    Attacker is the RL agent, trying to reach target while avoiding defender."""
    world_id = wp.tid()
    att_idx = world_id * bodies_per_world + attacker_offset
    def_idx = world_id * bodies_per_world + defender_offset

    att_pos = wp.transform_get_translation(body_q[att_idx])
    def_pos = wp.transform_get_translation(body_q[def_idx])
    att_vel = wp.vec3(body_qd[att_idx][0], body_qd[att_idx][1], body_qd[att_idx][2])
    def_vel = wp.vec3(body_qd[def_idx][0], body_qd[def_idx][1], body_qd[def_idx][2])
    rel_to_target = target_pos - att_pos
    dist_to_target = wp.length(rel_to_target)

    pos_scale = 0.25
    vel_scale = 0.2
    rel_scale = 1.0 / 6.0

    # Attacker pos/vel (self)
    obs_out[world_id, 0] = att_pos[0] * pos_scale
    obs_out[world_id, 1] = att_pos[1] * pos_scale
    obs_out[world_id, 2] = att_pos[2] * pos_scale
    obs_out[world_id, 3] = att_vel[0] * vel_scale
    obs_out[world_id, 4] = att_vel[1] * vel_scale
    obs_out[world_id, 5] = att_vel[2] * vel_scale
    # Target position (goal)
    obs_out[world_id, 6] = target_pos[0] * pos_scale
    obs_out[world_id, 7] = target_pos[1] * pos_scale
    obs_out[world_id, 8] = target_pos[2] * pos_scale
    # Defender velocity (threat)
    obs_out[world_id, 9] = def_vel[0] * vel_scale
    obs_out[world_id, 10] = def_vel[1] * vel_scale
    obs_out[world_id, 11] = def_vel[2] * vel_scale
    # Relative to target
    obs_out[world_id, 12] = rel_to_target[0] * rel_scale
    obs_out[world_id, 13] = rel_to_target[1] * rel_scale
    obs_out[world_id, 14] = rel_to_target[2] * rel_scale
    obs_out[world_id, 15] = dist_to_target * rel_scale
    # Defender position
    obs_out[world_id, 16] = def_pos[0] * pos_scale
    obs_out[world_id, 17] = def_pos[1] * pos_scale
    obs_out[world_id, 18] = def_pos[2] * pos_scale


@wp.kernel
def compute_rewards_evade_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    prev_dist: wp.array(dtype=float),
    target_pos: wp.vec3,
    intercept_dist: float,
    target_reached_dist: float,
    bodies_per_world: int,
    attacker_offset: int,
    defender_offset: int,
    rewards_out: wp.array(dtype=float),
    dones_out: wp.array(dtype=int),
    info_target_reached: wp.array(dtype=int),
    info_intercepted: wp.array(dtype=int),
):
    """Reward for evade mode: reach target while avoiding defender."""
    world_id = wp.tid()
    att_idx = world_id * bodies_per_world + attacker_offset
    def_idx = world_id * bodies_per_world + defender_offset

    att_pos = wp.transform_get_translation(body_q[att_idx])
    def_pos = wp.transform_get_translation(body_q[def_idx])
    dist_to_target = wp.length(att_pos - target_pos)
    dist_to_defender = wp.length(att_pos - def_pos)

    reward = 0.0
    done = 0

    if dist_to_target < target_reached_dist:
        # Attacker wins: reached the target
        reward = 200.0
        done = 1
        info_target_reached[world_id] = 1
    elif dist_to_defender < intercept_dist:
        # Attacker loses: got intercepted
        reward = -200.0
        done = 1
        info_intercepted[world_id] = 1
    else:
        pd = prev_dist[world_id]
        # Approach target (primary objective)
        if pd > 0.0:
            reward += (pd - dist_to_target) * 50.0
        # Proximity bonus for being near target
        reward += 1.0 / (dist_to_target + 0.1) - 1.0 / (4.0 + 0.1)
        # Evasion bonus: reward for keeping distance from defender
        if dist_to_defender < 1.5:
            reward -= (1.5 - dist_to_defender) * 2.0
        # Time penalty
        reward -= 0.05

    rewards_out[world_id] = reward
    prev_dist[world_id] = dist_to_target
    dones_out[world_id] = done


@wp.kernel
def compute_rewards_intercept_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    prev_dist: wp.array(dtype=float),
    target_pos: wp.vec3,
    intercept_dist: float,
    target_reached_dist: float,
    bodies_per_world: int,
    defender_offset: int,
    attacker_offset: int,
    rewards_out: wp.array(dtype=float),
    dones_out: wp.array(dtype=int),
    info_intercepted: wp.array(dtype=int),
    info_target_reached: wp.array(dtype=int),
):
    """Reward for intercept mode v3: approach + closing speed + urgency + prediction."""
    world_id = wp.tid()
    def_idx = world_id * bodies_per_world + defender_offset
    att_idx = world_id * bodies_per_world + attacker_offset

    def_pos = wp.transform_get_translation(body_q[def_idx])
    att_pos = wp.transform_get_translation(body_q[att_idx])
    def_vel = wp.vec3(body_qd[def_idx][0], body_qd[def_idx][1], body_qd[def_idx][2])
    att_vel = wp.vec3(body_qd[att_idx][0], body_qd[att_idx][1], body_qd[att_idx][2])
    to_attacker = att_pos - def_pos
    dist = wp.length(to_attacker)
    att_to_target = wp.length(att_pos - target_pos)

    reward = 0.0
    done = 0

    if dist < intercept_dist:
        reward = 200.0
        done = 1
        info_intercepted[world_id] = 1
    elif att_to_target < target_reached_dist:
        reward = -200.0
        done = 1
        info_target_reached[world_id] = 1
    else:
        pd = prev_dist[world_id]
        # Strong approach reward
        if pd > 0.0:
            reward += (pd - dist) * 50.0

        # Closing speed bonus: reward velocity pointing toward attacker
        # relative_vel = def_vel - att_vel (closing velocity)
        # closing_speed = dot(relative_vel, direction_to_attacker)
        if dist > 0.01:
            direction = to_attacker * (1.0 / dist)
            rel_vel = def_vel - att_vel
            closing_speed = wp.dot(rel_vel, direction)
            # Reward positive closing speed (approaching), penalize opening
            reward += closing_speed * 2.0

        # Proximity bonus — sharp increase as defender gets close
        reward += 1.0 / (dist + 0.1) - 1.0 / (4.0 + 0.1)
        # Quadratic distance penalty — penalize being far
        reward -= dist * dist * 0.3
        # Urgency: as attacker nears target, increase time pressure
        urgency = 1.0 / (att_to_target + 0.5)
        reward -= urgency * 0.5
        # Time penalty
        reward -= 0.05

    rewards_out[world_id] = reward
    prev_dist[world_id] = dist
    dones_out[world_id] = done


@wp.kernel
def compute_rewards_navigate_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    prev_dist: wp.array(dtype=float),
    target_positions: wp.array(dtype=wp.vec3),
    arrival_dist: float,
    arrival_bonus: float,
    bodies_per_world: int,
    drone_offset: int,
    rewards_out: wp.array(dtype=float),
    dones_out: wp.array(dtype=int),
    info_arrived: wp.array(dtype=int),
):
    """Reward for fly_to_target / chase: dense approach + arrival bonus."""
    world_id = wp.tid()
    body_idx = world_id * bodies_per_world + drone_offset

    pos = wp.transform_get_translation(body_q[body_idx])
    tgt = target_positions[world_id]
    dist = wp.length(tgt - pos)

    vel = wp.vec3(
        body_qd[body_idx][0],
        body_qd[body_idx][1],
        body_qd[body_idx][2],
    )
    speed = wp.length(vel)

    reward = 0.0
    done = 0

    if dist < arrival_dist:
        reward = arrival_bonus
        done = 1
        info_arrived[world_id] = 1
    else:
        # Dense approach reward — large multiplier so signal >> noise
        pd = prev_dist[world_id]
        if pd > 0.0:
            reward += (pd - dist) * 200.0

        # Distance penalty — quadratic so being far is much worse
        reward -= dist * dist * 0.5

        # Exponential proximity bonus — gets VERY strong near target
        # At dist=0.5: bonus = 1/(0.5+0.1) - 1/(3+0.1) ≈ 1.37
        # At dist=0.3: bonus = 1/(0.3+0.1) - 0.32 ≈ 2.18
        # Pulls drone through the last 0.5m to arrival
        reward += 1.0 / (dist + 0.1) - 1.0 / (3.0 + 0.1)

        # Small time penalty
        reward -= 0.05

    rewards_out[world_id] = reward
    prev_dist[world_id] = dist
    dones_out[world_id] = done


@wp.kernel
def check_bounds_kernel(
    body_q: wp.array(dtype=wp.transform),
    step_count: wp.array(dtype=int),
    max_steps: int,
    arena_bounds: float,
    alt_min: float,
    alt_max: float,
    bodies_per_world: int,
    num_bodies_to_check: int,
    dones: wp.array(dtype=int),
):
    """Check truncation conditions (bounds + timeout)."""
    world_id = wp.tid()

    if step_count[world_id] >= max_steps:
        dones[world_id] = 1

    for i in range(num_bodies_to_check):
        body_idx = world_id * bodies_per_world + i
        pos = wp.transform_get_translation(body_q[body_idx])
        if wp.abs(pos[0]) > arena_bounds or wp.abs(pos[1]) > arena_bounds:
            dones[world_id] = 1
        if pos[2] < alt_min or pos[2] > alt_max:
            dones[world_id] = 1


@wp.kernel
def reset_bodies_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    init_positions: wp.array(dtype=wp.vec3, ndim=2),
    reset_mask: wp.array(dtype=int),
    bodies_per_world: int,
    num_bodies: int,
):
    """Reset bodies in worlds that are done."""
    world_id = wp.tid()
    if reset_mask[world_id] == 0:
        return

    for i in range(num_bodies):
        body_idx = world_id * bodies_per_world + i
        body_q[body_idx] = wp.transform(init_positions[world_id, i], wp.quat_identity())
        body_qd[body_idx] = wp.spatial_vector(
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)
        )


# ------------------------------------------------------------------ #
# Helper: build a single drone body
# ------------------------------------------------------------------ #

def _build_drone(builder, arm_length, collision_group, label="drone"):
    """Add a drone body with crossbar shapes. Returns body index."""
    import newton as nt

    body = builder.add_body(label=label)
    bar_h = arm_length * 0.05
    bar_w = arm_length * 0.05
    density = 84.4  # tuned so drone mass ≈ 0.027 kg (CrazyFlie 2.1)

    cfg = nt.ModelBuilder.ShapeConfig(
        density=density, collision_group=collision_group
    )
    builder.add_shape_box(body, hx=bar_w, hy=arm_length, hz=bar_h, cfg=cfg)
    builder.add_shape_box(body, hx=arm_length, hy=bar_w, hz=bar_h, cfg=cfg)

    return body


# ------------------------------------------------------------------ #
# Unified multi-mode environment
# ------------------------------------------------------------------ #

VALID_MODES = ("fly_to_target", "chase", "intercept", "evade", "adversarial")


class NewtonDroneEnv(VecEnv):
    """
    GPU-accelerated vectorized drone environment with multiple training modes.

    All modes share the same 19-dim observation and 3-dim action space
    to enable weight transfer across progressive training phases.
    """

    OBS_DIM = 19
    ACT_DIM = 3

    def __init__(self, num_envs=256, config=None, device="cuda:0"):
        import newton as nt
        import gymnasium as gym

        self.cfg = config or EnvConfig()
        self.mode = self.cfg.mode
        assert self.mode in VALID_MODES, f"Unknown mode: {self.mode}. Must be one of {VALID_MODES}"

        self.device = device
        self.n_envs = num_envs

        # How many drone bodies per world
        self._use_two_drones = self.mode in ("intercept", "evade", "adversarial")

        # Physics params
        self.fps = self.cfg.ctrl_freq
        self.dt = 1.0 / self.fps
        self.substeps = max(1, self.cfg.pyb_freq // self.cfg.ctrl_freq)
        self.sub_dt = self.dt / self.substeps
        self.arm_length = 0.2
        self.max_steps = int(self.cfg.episode_secs * self.cfg.ctrl_freq)
        self.target_pos_wp = wp.vec3(*self.cfg.target_pos)

        self.drone_mass = 0.027  # placeholder
        self.gravity = 9.81

        # Speed limits
        self.defender_speed_limit = 0.15 * 30.0 * (1000.0 / 3600.0)  # ~1.25 m/s
        self.attacker_speed = self.cfg.attacker_speed_scale * 0.8

        # Force model
        self.force_kp = self.cfg.force_kp
        self.force_kd = self.cfg.force_kd
        self.max_force = self.cfg.max_force
        self.angular_damping = self.cfg.angular_damping

        # Build scene
        builder = nt.ModelBuilder(gravity=-self.gravity)
        self.bodies_per_world = 2 if self._use_two_drones else 1

        for env_i in range(num_envs):
            builder.begin_world(label=f"world_{env_i}")
            _build_drone(builder, self.arm_length,
                         collision_group=env_i * 2, label=f"drone0_{env_i}")
            if self._use_two_drones:
                _build_drone(builder, self.arm_length,
                             collision_group=env_i * 2 + 1, label=f"drone1_{env_i}")
            builder.end_world()

        self.model = builder.finalize()
        body_masses = self.model.body_mass.numpy()
        self.drone_mass = float(body_masses[0])

        # States
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.solver = nt.solvers.SolverSemiImplicit(self.model)

        # RL state arrays
        self.obs_buf = wp.zeros((num_envs, self.OBS_DIM), dtype=float, device=device)
        self.rewards_buf = wp.zeros(num_envs, dtype=float, device=device)
        self.dones_buf = wp.zeros(num_envs, dtype=int, device=device)
        self.term_buf = wp.zeros(num_envs, dtype=int, device=device)
        self.prev_dist_buf = wp.zeros(num_envs, dtype=float, device=device)
        self.step_count_buf = wp.zeros(num_envs, dtype=int, device=device)

        # Info buffers (mode-dependent meaning)
        self.info_buf_a = wp.zeros(num_envs, dtype=int, device=device)  # intercepted / arrived
        self.info_buf_b = wp.zeros(num_envs, dtype=int, device=device)  # target_reached (intercept only)

        # Velocity command buffers
        self.drone0_vel = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        if self._use_two_drones:
            self.drone1_vel = wp.zeros(num_envs, dtype=wp.vec3, device=device)

        # Target positions for fly_to_target / chase modes
        self.target_positions = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        self.target_velocities = wp.zeros(num_envs, dtype=wp.vec3, device=device)

        # Initial positions buffer
        self.init_pos_buf = wp.zeros(
            (num_envs, self.bodies_per_world), dtype=wp.vec3, device=device
        )

        # Frozen attacker model(s) for intercept mode
        self.attacker_model = None
        self.attacker_models = []
        self._attacker_assignments = None  # per-env model index for mixture training
        if self.mode == "intercept":
            from stable_baselines3 import SAC
            # Mixture of opponents: load multiple models
            if self.cfg.attacker_model_paths:
                for path in self.cfg.attacker_model_paths:
                    print(f"  Loading attacker model: {path}")
                    self.attacker_models.append(SAC.load(path))
                # Also include scripted pursuit as an "opponent" (None entry)
                self.attacker_models.append(None)
                self._attacker_assignments = np.zeros(num_envs, dtype=int)
                self._randomize_attacker_assignments()
            elif self.cfg.attacker_model_path:
                self.attacker_model = SAC.load(self.cfg.attacker_model_path)

        # Frozen defender model (for evade mode — RL controls attacker)
        self.defender_model = None
        if self.mode == "evade" and self.cfg.defender_model_path:
            from stable_baselines3 import SAC
            self.defender_model = SAC.load(self.cfg.defender_model_path)

        # Gymnasium spaces
        obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.OBS_DIM,), dtype=np.float32
        )
        act_space = gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.ACT_DIM,), dtype=np.float32
        )

        super().__init__(num_envs, obs_space, act_space)

        # Initial reset
        self._randomize_all()
        self._apply_init_positions(mask=None)

    # ------------------------------------------------------------------ #
    # VecEnv interface
    # ------------------------------------------------------------------ #

    def reset(self):
        self._randomize_all()
        self._apply_init_positions(mask=None)
        self.prev_dist_buf.zero_()
        self.step_count_buf.zero_()
        self._randomize_attacker_assignments()
        self._compute_obs()
        return self.obs_buf.numpy().astype(np.float32)

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        actions = self._actions

        if self.mode == "evade":
            # RL agent controls drone1 (attacker), frozen defender controls drone0
            vel_np = actions * self.defender_speed_limit
            self.drone1_vel = wp.array(
                [wp.vec3(v[0], v[1], v[2]) for v in vel_np],
                dtype=wp.vec3, device=self.device
            )
            self._compute_drone0_velocity_frozen()
        else:
            # RL agent controls drone0, compute drone1 velocity
            vel_np = actions * self.defender_speed_limit
            self.drone0_vel = wp.array(
                [wp.vec3(v[0], v[1], v[2]) for v in vel_np],
                dtype=wp.vec3, device=self.device
            )
            if self._use_two_drones:
                self._compute_drone1_velocity()

        # Update moving targets (chase mode)
        if self.mode == "chase":
            self._update_moving_targets()

        # Physics substeps
        for _ in range(self.substeps):
            self.state_0.clear_forces()

            # Forces on drone 0 (RL agent)
            wp.launch(
                apply_velocity_force,
                dim=self.n_envs,
                inputs=(
                    self.state_0.body_q, self.state_0.body_qd,
                    self.drone0_vel, self.state_0.body_f,
                    self.bodies_per_world, 0,
                    self.drone_mass, self.gravity,
                    self.force_kp, self.force_kd,
                    self.max_force, self.angular_damping,
                ),
                device=self.device,
            )

            # Forces on drone 1 (if present)
            if self._use_two_drones:
                wp.launch(
                    apply_velocity_force,
                    dim=self.n_envs,
                    inputs=(
                        self.state_0.body_q, self.state_0.body_qd,
                        self.drone1_vel, self.state_0.body_f,
                        self.bodies_per_world, 1,
                        self.drone_mass, self.gravity,
                        self.force_kp, self.force_kd,
                        self.max_force, self.angular_damping,
                    ),
                    device=self.device,
                )

            self.solver.step(self.state_0, self.state_1, None, None, self.sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # Increment step counters
        step_np = self.step_count_buf.numpy()
        step_np += 1
        self.step_count_buf = wp.array(step_np, dtype=int, device=self.device)

        # Compute rewards and termination
        self.info_buf_a.zero_()
        self.info_buf_b.zero_()
        self.term_buf.zero_()
        self.dones_buf.zero_()

        self._compute_rewards()

        # Check bounds/timeout
        wp.launch(
            check_bounds_kernel,
            dim=self.n_envs,
            inputs=(
                self.state_0.body_q, self.step_count_buf,
                self.max_steps, self.cfg.arena_bounds,
                self.cfg.altitude_min, self.cfg.altitude_max,
                self.bodies_per_world,
                self.bodies_per_world,  # check all bodies
                self.dones_buf,
            ),
            device=self.device,
        )

        wp.synchronize()

        # Read results to CPU
        rewards = self.rewards_buf.numpy().astype(np.float32)
        terms = self.term_buf.numpy()
        truncs = self.dones_buf.numpy()
        dones = np.logical_or(terms > 0, truncs > 0)
        info_a = self.info_buf_a.numpy()
        info_b = self.info_buf_b.numpy()

        infos = []
        for i in range(self.n_envs):
            if self.mode in ("fly_to_target", "chase"):
                infos.append({"arrived": bool(info_a[i])})
            elif self.mode == "evade":
                # In evade mode: info_a = target_reached (win), info_b = intercepted (lose)
                infos.append({
                    "target_reached": bool(info_a[i]),
                    "intercepted": bool(info_b[i]),
                })
            else:
                infos.append({
                    "intercepted": bool(info_a[i]),
                    "target_reached": bool(info_b[i]),
                })

        # Auto-reset done environments
        if np.any(dones):
            done_mask = dones.astype(np.int32)
            self._randomize_masked(done_mask)
            reset_mask_wp = wp.array(done_mask, dtype=int, device=self.device)
            wp.launch(
                reset_bodies_kernel,
                dim=self.n_envs,
                inputs=(
                    self.state_0.body_q, self.state_0.body_qd,
                    self.init_pos_buf, reset_mask_wp,
                    self.bodies_per_world, self.bodies_per_world,
                ),
                device=self.device,
            )
            step_np = self.step_count_buf.numpy()
            prev_np = self.prev_dist_buf.numpy()
            step_np[dones] = 0
            prev_np[dones] = 0.0
            self.step_count_buf = wp.array(step_np, dtype=int, device=self.device)
            self.prev_dist_buf = wp.array(prev_np, dtype=float, device=self.device)
            # Re-randomize attacker assignments for reset envs
            if self._attacker_assignments is not None:
                done_indices = np.where(dones)[0]
                n_models = len(self.attacker_models)
                self._attacker_assignments[done_indices] = np.random.randint(0, n_models, size=len(done_indices))

        self._compute_obs()
        obs = self.obs_buf.numpy().astype(np.float32)
        return obs, rewards, dones, infos

    def close(self):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.n_envs

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return [None] * self.n_envs

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None] * self.n_envs
        return [None] * self.n_envs

    def set_attr(self, attr_name, value, indices=None):
        pass

    def seed(self, seed=None):
        pass

    # ------------------------------------------------------------------ #
    # Mode-specific logic
    # ------------------------------------------------------------------ #

    def _randomize_attacker_assignments(self):
        """Randomly assign attacker models to environments for mixture training."""
        if self._attacker_assignments is not None:
            n_models = len(self.attacker_models)
            self._attacker_assignments = np.random.randint(0, n_models, size=self.n_envs)

    def _compute_drone1_velocity(self):
        """Compute velocity for the second drone (attacker)."""
        if self.attacker_models:
            # Mixture of opponents: each env may use a different attacker
            obs = self._get_attacker_obs()
            vel_np = np.zeros((self.n_envs, 3), dtype=np.float32)

            for model_idx, model in enumerate(self.attacker_models):
                mask = self._attacker_assignments == model_idx
                if not np.any(mask):
                    continue
                if model is None:
                    # Scripted pursuit — compute on GPU then read back
                    wp.launch(
                        attacker_pursuit,
                        dim=self.n_envs,
                        inputs=(
                            self.state_0.body_q, self.state_0.body_qd,
                            self.target_pos_wp, self.drone1_vel,
                            self.bodies_per_world, 1,
                            self.attacker_speed,
                        ),
                        device=self.device,
                    )
                    wp.synchronize()
                    scripted_vel = np.array([[v[0], v[1], v[2]] for v in self.drone1_vel.numpy()])
                    vel_np[mask] = scripted_vel[mask]
                else:
                    env_indices = np.where(mask)[0]
                    subset_obs = obs[env_indices]
                    actions, _ = model.predict(subset_obs, deterministic=True)
                    vel_np[env_indices] = actions * self.defender_speed_limit

            self.drone1_vel = wp.array(
                [wp.vec3(v[0], v[1], v[2]) for v in vel_np],
                dtype=wp.vec3, device=self.device
            )
        elif self.attacker_model is not None:
            # Single frozen RL attacker
            obs = self._get_attacker_obs()
            actions, _ = self.attacker_model.predict(obs, deterministic=True)
            vel_np = actions * self.defender_speed_limit
            self.drone1_vel = wp.array(
                [wp.vec3(v[0], v[1], v[2]) for v in vel_np],
                dtype=wp.vec3, device=self.device
            )
        else:
            # Scripted pursuit
            wp.launch(
                attacker_pursuit,
                dim=self.n_envs,
                inputs=(
                    self.state_0.body_q, self.state_0.body_qd,
                    self.target_pos_wp, self.drone1_vel,
                    self.bodies_per_world, 1,
                    self.attacker_speed,
                ),
                device=self.device,
            )

    def _get_attacker_obs(self):
        """Build observations from the attacker's perspective (swap drone 0/1).
        Uses same manual normalization as kernels for consistency."""
        wp.synchronize()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        target = np.array(self.cfg.target_pos)

        pos_scale = 0.25     # 1/4.0
        vel_scale = 0.2      # 1/5.0
        rel_scale = 1.0 / 6.0

        obs = np.zeros((self.n_envs, self.OBS_DIM), dtype=np.float32)
        for i in range(self.n_envs):
            att_idx = i * self.bodies_per_world + 1
            def_idx = i * self.bodies_per_world + 0

            att_pos = body_q[att_idx][:3]
            def_pos = body_q[def_idx][:3]
            att_vel = body_qd[att_idx][:3]
            def_vel = body_qd[def_idx][:3]
            rel = target - att_pos
            dist = np.linalg.norm(rel)

            obs[i, 0:3] = att_pos * pos_scale
            obs[i, 3:6] = att_vel * vel_scale
            obs[i, 6:9] = target * pos_scale
            obs[i, 9:12] = 0.0  # target doesn't move
            obs[i, 12:15] = rel * rel_scale
            obs[i, 15] = dist * rel_scale
            obs[i, 16:19] = def_pos * pos_scale
        return obs

    def _compute_drone0_velocity_frozen(self):
        """Compute velocity for drone0 (frozen defender) in evade mode."""
        if self.defender_model is not None:
            obs = self._get_defender_obs()
            actions, _ = self.defender_model.predict(obs, deterministic=True)
            vel_np = actions * self.defender_speed_limit
            self.drone0_vel = wp.array(
                [wp.vec3(v[0], v[1], v[2]) for v in vel_np],
                dtype=wp.vec3, device=self.device
            )

    def _get_defender_obs(self):
        """Build observations from defender's perspective for frozen defender.
        Same format as compute_obs_kernel: defender=drone0, attacker=drone1."""
        wp.synchronize()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        target = np.array(self.cfg.target_pos)

        pos_scale = 0.25
        vel_scale = 0.2
        rel_scale = 1.0 / 6.0

        obs = np.zeros((self.n_envs, self.OBS_DIM), dtype=np.float32)
        for i in range(self.n_envs):
            def_idx = i * self.bodies_per_world + 0
            att_idx = i * self.bodies_per_world + 1

            def_pos = body_q[def_idx][:3]
            att_pos = body_q[att_idx][:3]
            def_vel = body_qd[def_idx][:3]
            att_vel = body_qd[att_idx][:3]
            rel = att_pos - def_pos
            dist = np.linalg.norm(rel)

            obs[i, 0:3] = def_pos * pos_scale
            obs[i, 3:6] = def_vel * vel_scale
            obs[i, 6:9] = att_pos * pos_scale
            obs[i, 9:12] = att_vel * vel_scale
            obs[i, 12:15] = rel * rel_scale
            obs[i, 15] = dist * rel_scale
            obs[i, 16:19] = target * pos_scale
        return obs

    def _update_moving_targets(self):
        """Move targets along their velocity vectors (chase mode)."""
        tgt_np = self.target_positions.numpy()
        vel_np = self.target_velocities.numpy()
        tgt_np += vel_np * self.dt

        # Bounce off arena bounds
        bounds = self.cfg.arena_bounds * 0.8  # stay inside
        alt_lo, alt_hi = self.cfg.target_alt_range
        for i in range(self.n_envs):
            for ax in range(2):  # x, y
                if abs(tgt_np[i][ax]) > bounds:
                    vel_np[i][ax] *= -1.0
                    tgt_np[i][ax] = np.clip(tgt_np[i][ax], -bounds, bounds)
            if tgt_np[i][2] < alt_lo or tgt_np[i][2] > alt_hi:
                vel_np[i][2] *= -1.0
                tgt_np[i][2] = np.clip(tgt_np[i][2], alt_lo, alt_hi)

        self.target_positions = wp.array(
            [wp.vec3(float(t[0]), float(t[1]), float(t[2])) for t in tgt_np],
            dtype=wp.vec3, device=self.device
        )
        self.target_velocities = wp.array(
            [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in vel_np],
            dtype=wp.vec3, device=self.device
        )

    def _compute_obs(self):
        """Compute observations based on current mode."""
        if self.mode in ("fly_to_target", "chase"):
            wp.launch(
                compute_obs_single_drone_kernel,
                dim=self.n_envs,
                inputs=(
                    self.state_0.body_q, self.state_0.body_qd,
                    self.target_positions, self.target_velocities,
                    self.obs_buf,
                    self.bodies_per_world, 0,
                ),
                device=self.device,
            )
        elif self.mode == "evade":
            wp.launch(
                compute_obs_evade_kernel,
                dim=self.n_envs,
                inputs=(
                    self.state_0.body_q, self.state_0.body_qd,
                    self.target_pos_wp, self.obs_buf,
                    self.bodies_per_world, 1, 0,  # attacker=1, defender=0
                ),
                device=self.device,
            )
        else:
            wp.launch(
                compute_obs_kernel,
                dim=self.n_envs,
                inputs=(
                    self.state_0.body_q, self.state_0.body_qd,
                    self.target_pos_wp, self.obs_buf,
                    self.bodies_per_world, 0, 1,
                ),
                device=self.device,
            )

    def _compute_rewards(self):
        """Compute rewards based on current mode."""
        if self.mode in ("fly_to_target", "chase"):
            wp.launch(
                compute_rewards_navigate_kernel,
                dim=self.n_envs,
                inputs=(
                    self.state_0.body_q, self.state_0.body_qd,
                    self.prev_dist_buf,
                    self.target_positions,
                    self.cfg.arrival_dist,
                    self.cfg.arrival_bonus,
                    self.bodies_per_world, 0,
                    self.rewards_buf, self.term_buf,
                    self.info_buf_a,
                ),
                device=self.device,
            )
        elif self.mode == "evade":
            wp.launch(
                compute_rewards_evade_kernel,
                dim=self.n_envs,
                inputs=(
                    self.state_0.body_q, self.state_0.body_qd,
                    self.prev_dist_buf,
                    self.target_pos_wp,
                    self.cfg.intercept_dist,
                    self.cfg.target_reached_dist,
                    self.bodies_per_world, 1, 0,  # attacker=1, defender=0
                    self.rewards_buf, self.term_buf,
                    self.info_buf_a, self.info_buf_b,
                ),
                device=self.device,
            )
        else:
            wp.launch(
                compute_rewards_intercept_kernel,
                dim=self.n_envs,
                inputs=(
                    self.state_0.body_q, self.state_0.body_qd,
                    self.prev_dist_buf,
                    self.target_pos_wp,
                    self.cfg.intercept_dist,
                    self.cfg.target_reached_dist,
                    self.bodies_per_world, 0, 1,
                    self.rewards_buf, self.term_buf,
                    self.info_buf_a, self.info_buf_b,
                ),
                device=self.device,
            )

    # ------------------------------------------------------------------ #
    # Position randomization
    # ------------------------------------------------------------------ #

    def _randomize_all(self):
        """Randomize initial positions and targets for all environments."""
        rng = np.random.default_rng()

        if self.mode in ("fly_to_target", "chase"):
            self._randomize_single_drone_positions(rng, range(self.n_envs))
        else:
            self._randomize_intercept_positions(rng, range(self.n_envs))

    def _randomize_masked(self, mask):
        """Randomize positions only for done environments."""
        rng = np.random.default_rng()
        indices = [i for i in range(self.n_envs) if mask[i]]

        if self.mode in ("fly_to_target", "chase"):
            self._randomize_single_drone_positions(rng, indices)
        else:
            self._randomize_intercept_positions(rng, indices)

    def _randomize_single_drone_positions(self, rng, indices):
        """Randomize drone + target positions for fly_to_target / chase."""
        positions = self.init_pos_buf.numpy()
        tgt_np = self.target_positions.numpy()
        vel_np = self.target_velocities.numpy()

        for i in indices:
            # Drone starts near origin at random altitude
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.3, 1.0)
            positions[i, 0] = [
                dist * np.cos(angle),
                dist * np.sin(angle),
                rng.uniform(*self.cfg.defender_alt_range),
            ]

            # Target at random position in arena
            t_angle = rng.uniform(0, 2 * np.pi)
            t_dist = rng.uniform(1.0, self.cfg.target_randomize_range)
            tgt_np[i] = [
                t_dist * np.cos(t_angle),
                t_dist * np.sin(t_angle),
                rng.uniform(*self.cfg.target_alt_range),
            ]

            # Moving target velocity (chase mode only)
            if self.mode == "chase":
                v_angle = rng.uniform(0, 2 * np.pi)
                v_alt = rng.uniform(-0.3, 0.3)
                speed = self.cfg.moving_target_speed
                vel_np[i] = [
                    speed * np.cos(v_angle),
                    speed * np.sin(v_angle),
                    v_alt * speed,
                ]
            else:
                vel_np[i] = [0.0, 0.0, 0.0]

        self.init_pos_buf = wp.array(
            [[wp.vec3(float(positions[i, j, 0]),
                      float(positions[i, j, 1]),
                      float(positions[i, j, 2]))
              for j in range(self.bodies_per_world)]
             for i in range(self.n_envs)],
            dtype=wp.vec3, device=self.device
        )
        self.target_positions = wp.array(
            [wp.vec3(float(tgt_np[i][0]), float(tgt_np[i][1]), float(tgt_np[i][2]))
             for i in range(self.n_envs)],
            dtype=wp.vec3, device=self.device
        )
        self.target_velocities = wp.array(
            [wp.vec3(float(vel_np[i][0]), float(vel_np[i][1]), float(vel_np[i][2]))
             for i in range(self.n_envs)],
            dtype=wp.vec3, device=self.device
        )

    def _randomize_intercept_positions(self, rng, indices):
        """Randomize defender + attacker positions for intercept mode."""
        positions = self.init_pos_buf.numpy()
        target = np.array(self.cfg.target_pos)

        for i in indices:
            # Defender near target
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(*self.cfg.defender_dist_range)
            positions[i, 0] = [
                target[0] + dist * np.cos(angle),
                target[1] + dist * np.sin(angle),
                rng.uniform(*self.cfg.defender_alt_range),
            ]
            # Attacker far from target
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(*self.cfg.attacker_dist_range)
            positions[i, 1] = [
                target[0] + dist * np.cos(angle),
                target[1] + dist * np.sin(angle),
                rng.uniform(*self.cfg.attacker_alt_range),
            ]

        self.init_pos_buf = wp.array(
            [[wp.vec3(float(positions[i, j, 0]),
                      float(positions[i, j, 1]),
                      float(positions[i, j, 2]))
              for j in range(self.bodies_per_world)]
             for i in range(self.n_envs)],
            dtype=wp.vec3, device=self.device
        )

    def _apply_init_positions(self, mask=None):
        if mask is None:
            mask_wp = wp.ones(self.n_envs, dtype=int, device=self.device)
        else:
            mask_wp = wp.array(mask, dtype=int, device=self.device)

        wp.launch(
            reset_bodies_kernel,
            dim=self.n_envs,
            inputs=(
                self.state_0.body_q, self.state_0.body_qd,
                self.init_pos_buf, mask_wp,
                self.bodies_per_world, self.bodies_per_world,
            ),
            device=self.device,
        )
