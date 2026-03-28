"""
Unified GPU-Accelerated Drone Environment (Newton Physics)

Supports multiple training modes with a consistent observation space
and 3-dim action space, enabling progressive training with weight transfer:

  fly_to_target — Single drone flies to random static targets
  chase         — Single drone chases a moving target
  intercept     — Defender vs scripted/frozen attacker (original scenario)
  evade         — Attacker (RL) vs frozen defender
  adversarial   — Both drones are RL-controlled (future)

Force model: Direct world-frame forces (no propeller simulation).
Compatible with Stable Baselines3 via VecEnv interface.

Observation (35-dim):
  [0:3]   self pos          [3:6]   self vel
  [6:9]   opponent/target   [9:12]  opponent/target vel
  [12:15] relative pos      [15]    distance
  [16:19] target/defender pos
  [19]    closing rate       [20:23] self acceleration
  [23:27] obstacle 0 (rel_x, rel_y, rel_z, dist)
  [27:31] obstacle 1
  [31:35] obstacle 2
"""

import numpy as np
import warp as wp
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from train.config import EnvConfig

MAX_OBSTACLES = 3


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
    prev_vel: wp.array(dtype=wp.vec3),
    obstacle_positions: wp.array(dtype=wp.vec3, ndim=2),
    num_obstacles: int,
    obs_out: wp.array(dtype=float, ndim=2),
    bodies_per_world: int,
    defender_offset: int,
    attacker_offset: int,
):
    """Compute 35-dim observation for each world (intercept mode)."""
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

    # Base 19 dims (unchanged layout)
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

    # Closing rate (scalar)
    closing_rate = 0.0
    if dist > 0.01:
        direction = rel_pos * (1.0 / dist)
        rel_vel = def_vel - att_vel
        closing_rate = wp.dot(rel_vel, direction)
    obs_out[world_id, 19] = closing_rate * vel_scale

    # Acceleration (change in velocity)
    pv = prev_vel[world_id]
    acc = def_vel - pv
    obs_out[world_id, 20] = acc[0] * vel_scale * 5.0
    obs_out[world_id, 21] = acc[1] * vel_scale * 5.0
    obs_out[world_id, 22] = acc[2] * vel_scale * 5.0

    # Obstacle relative positions (up to MAX_OBSTACLES=3, 4 dims each)
    for o in range(3):
        base = 23 + o * 4
        if o < num_obstacles:
            obs_pos = obstacle_positions[world_id, o]
            obs_rel = obs_pos - def_pos
            obs_dist = wp.length(obs_rel)
            obs_out[world_id, base + 0] = obs_rel[0] * rel_scale
            obs_out[world_id, base + 1] = obs_rel[1] * rel_scale
            obs_out[world_id, base + 2] = obs_rel[2] * rel_scale
            obs_out[world_id, base + 3] = obs_dist * rel_scale
        else:
            obs_out[world_id, base + 0] = 0.0
            obs_out[world_id, base + 1] = 0.0
            obs_out[world_id, base + 2] = 0.0
            obs_out[world_id, base + 3] = 0.0

    # Store current velocity for next step's acceleration
    prev_vel[world_id] = def_vel


@wp.kernel
def compute_obs_single_drone_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_positions: wp.array(dtype=wp.vec3),
    target_velocities: wp.array(dtype=wp.vec3),
    prev_vel: wp.array(dtype=wp.vec3),
    obstacle_positions: wp.array(dtype=wp.vec3, ndim=2),
    num_obstacles: int,
    obs_out: wp.array(dtype=float, ndim=2),
    bodies_per_world: int,
    drone_offset: int,
):
    """Compute 35-dim observation for fly_to_target / chase modes."""
    world_id = wp.tid()
    body_idx = world_id * bodies_per_world + drone_offset

    pos = wp.transform_get_translation(body_q[body_idx])
    vel = wp.vec3(body_qd[body_idx][0], body_qd[body_idx][1], body_qd[body_idx][2])
    tgt = target_positions[world_id]
    tgt_vel = target_velocities[world_id]
    rel = tgt - pos
    dist = wp.length(rel)

    pos_scale = 0.25
    vel_scale = 0.2
    rel_scale = 1.0 / 6.0

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

    # Closing rate toward target
    closing_rate = 0.0
    if dist > 0.01:
        direction = rel * (1.0 / dist)
        closing_rate = wp.dot(vel - tgt_vel, direction)
    obs_out[world_id, 19] = closing_rate * vel_scale

    # Acceleration
    pv = prev_vel[world_id]
    acc = vel - pv
    obs_out[world_id, 20] = acc[0] * vel_scale * 5.0
    obs_out[world_id, 21] = acc[1] * vel_scale * 5.0
    obs_out[world_id, 22] = acc[2] * vel_scale * 5.0

    # Obstacles
    for o in range(3):
        base = 23 + o * 4
        if o < num_obstacles:
            obs_pos = obstacle_positions[world_id, o]
            obs_rel = obs_pos - pos
            obs_dist = wp.length(obs_rel)
            obs_out[world_id, base + 0] = obs_rel[0] * rel_scale
            obs_out[world_id, base + 1] = obs_rel[1] * rel_scale
            obs_out[world_id, base + 2] = obs_rel[2] * rel_scale
            obs_out[world_id, base + 3] = obs_dist * rel_scale
        else:
            obs_out[world_id, base + 0] = 0.0
            obs_out[world_id, base + 1] = 0.0
            obs_out[world_id, base + 2] = 0.0
            obs_out[world_id, base + 3] = 0.0

    prev_vel[world_id] = vel


@wp.kernel
def compute_obs_evade_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_pos: wp.vec3,
    prev_vel: wp.array(dtype=wp.vec3),
    obstacle_positions: wp.array(dtype=wp.vec3, ndim=2),
    num_obstacles: int,
    obs_out: wp.array(dtype=float, ndim=2),
    bodies_per_world: int,
    attacker_offset: int,
    defender_offset: int,
):
    """Compute 35-dim observation from attacker perspective (evade mode)."""
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

    obs_out[world_id, 0] = att_pos[0] * pos_scale
    obs_out[world_id, 1] = att_pos[1] * pos_scale
    obs_out[world_id, 2] = att_pos[2] * pos_scale
    obs_out[world_id, 3] = att_vel[0] * vel_scale
    obs_out[world_id, 4] = att_vel[1] * vel_scale
    obs_out[world_id, 5] = att_vel[2] * vel_scale
    obs_out[world_id, 6] = target_pos[0] * pos_scale
    obs_out[world_id, 7] = target_pos[1] * pos_scale
    obs_out[world_id, 8] = target_pos[2] * pos_scale
    obs_out[world_id, 9] = def_vel[0] * vel_scale
    obs_out[world_id, 10] = def_vel[1] * vel_scale
    obs_out[world_id, 11] = def_vel[2] * vel_scale
    obs_out[world_id, 12] = rel_to_target[0] * rel_scale
    obs_out[world_id, 13] = rel_to_target[1] * rel_scale
    obs_out[world_id, 14] = rel_to_target[2] * rel_scale
    obs_out[world_id, 15] = dist_to_target * rel_scale
    obs_out[world_id, 16] = def_pos[0] * pos_scale
    obs_out[world_id, 17] = def_pos[1] * pos_scale
    obs_out[world_id, 18] = def_pos[2] * pos_scale

    # Closing rate (defender approaching attacker)
    rel_att_def = def_pos - att_pos
    dist_att_def = wp.length(rel_att_def)
    closing_rate = 0.0
    if dist_att_def > 0.01:
        direction = rel_att_def * (1.0 / dist_att_def)
        closing_rate = wp.dot(def_vel - att_vel, direction)
    obs_out[world_id, 19] = closing_rate * vel_scale

    # Acceleration
    pv = prev_vel[world_id]
    acc = att_vel - pv
    obs_out[world_id, 20] = acc[0] * vel_scale * 5.0
    obs_out[world_id, 21] = acc[1] * vel_scale * 5.0
    obs_out[world_id, 22] = acc[2] * vel_scale * 5.0

    # Obstacles (relative to attacker)
    for o in range(3):
        base = 23 + o * 4
        if o < num_obstacles:
            obs_pos = obstacle_positions[world_id, o]
            obs_rel = obs_pos - att_pos
            obs_dist = wp.length(obs_rel)
            obs_out[world_id, base + 0] = obs_rel[0] * rel_scale
            obs_out[world_id, base + 1] = obs_rel[1] * rel_scale
            obs_out[world_id, base + 2] = obs_rel[2] * rel_scale
            obs_out[world_id, base + 3] = obs_dist * rel_scale
        else:
            obs_out[world_id, base + 0] = 0.0
            obs_out[world_id, base + 1] = 0.0
            obs_out[world_id, base + 2] = 0.0
            obs_out[world_id, base + 3] = 0.0

    prev_vel[world_id] = att_vel


@wp.kernel
def compute_rewards_evade_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    prev_dist: wp.array(dtype=float),
    target_pos: wp.vec3,
    intercept_dist: float,
    target_reached_dist: float,
    obstacle_positions: wp.array(dtype=wp.vec3, ndim=2),
    num_obstacles: int,
    obstacle_danger_radius: float,
    obstacle_proximity_penalty: float,
    bodies_per_world: int,
    attacker_offset: int,
    defender_offset: int,
    rewards_out: wp.array(dtype=float),
    dones_out: wp.array(dtype=int),
    info_target_reached: wp.array(dtype=int),
    info_intercepted: wp.array(dtype=int),
):
    """Reward for evade mode: reach target while avoiding defender and obstacles."""
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
        reward = 200.0
        done = 1
        info_target_reached[world_id] = 1
    elif dist_to_defender < intercept_dist:
        reward = -200.0
        done = 1
        info_intercepted[world_id] = 1
    else:
        pd = prev_dist[world_id]
        if pd > 0.0:
            reward += (pd - dist_to_target) * 50.0
        reward += 1.0 / (dist_to_target + 0.1) - 1.0 / (4.0 + 0.1)
        if dist_to_defender < 1.5:
            reward -= (1.5 - dist_to_defender) * 2.0
        reward -= 0.05

        # Obstacle proximity penalty for attacker
        for o in range(3):
            if o < num_obstacles:
                obs_pos = obstacle_positions[world_id, o]
                d = wp.length(att_pos - obs_pos)
                if d < obstacle_danger_radius:
                    reward -= (obstacle_danger_radius - d) * obstacle_proximity_penalty

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
    obstacle_positions: wp.array(dtype=wp.vec3, ndim=2),
    num_obstacles: int,
    obstacle_danger_radius: float,
    obstacle_proximity_penalty: float,
    bodies_per_world: int,
    defender_offset: int,
    attacker_offset: int,
    rewards_out: wp.array(dtype=float),
    dones_out: wp.array(dtype=int),
    info_intercepted: wp.array(dtype=int),
    info_target_reached: wp.array(dtype=int),
):
    """Reward for intercept mode v3: approach + closing speed + urgency + obstacles."""
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
        if pd > 0.0:
            reward += (pd - dist) * 50.0

        # Closing speed bonus
        if dist > 0.01:
            direction = to_attacker * (1.0 / dist)
            rel_vel = def_vel - att_vel
            closing_speed = wp.dot(rel_vel, direction)
            reward += closing_speed * 2.0

        reward += 1.0 / (dist + 0.1) - 1.0 / (4.0 + 0.1)
        reward -= dist * dist * 0.3
        urgency = 1.0 / (att_to_target + 0.5)
        reward -= urgency * 0.5
        reward -= 0.05

        # Obstacle proximity penalty for defender
        for o in range(3):
            if o < num_obstacles:
                obs_pos = obstacle_positions[world_id, o]
                d = wp.length(def_pos - obs_pos)
                if d < obstacle_danger_radius:
                    reward -= (obstacle_danger_radius - d) * obstacle_proximity_penalty

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
    obstacle_positions: wp.array(dtype=wp.vec3, ndim=2),
    num_obstacles: int,
    obstacle_danger_radius: float,
    obstacle_proximity_penalty: float,
    bodies_per_world: int,
    drone_offset: int,
    rewards_out: wp.array(dtype=float),
    dones_out: wp.array(dtype=int),
    info_arrived: wp.array(dtype=int),
):
    """Reward for fly_to_target / chase: dense approach + arrival bonus + obstacle avoidance."""
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

    reward = 0.0
    done = 0

    if dist < arrival_dist:
        reward = arrival_bonus
        done = 1
        info_arrived[world_id] = 1
    else:
        pd = prev_dist[world_id]
        if pd > 0.0:
            reward += (pd - dist) * 200.0
        reward -= dist * dist * 0.5
        reward += 1.0 / (dist + 0.1) - 1.0 / (3.0 + 0.1)
        reward -= 0.05

        # Obstacle proximity penalty
        for o in range(3):
            if o < num_obstacles:
                obs_pos = obstacle_positions[world_id, o]
                d = wp.length(pos - obs_pos)
                if d < obstacle_danger_radius:
                    reward -= (obstacle_danger_radius - d) * obstacle_proximity_penalty

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
    num_drone_bodies: int,
    dones: wp.array(dtype=int),
):
    """Check truncation conditions (bounds + timeout). Only checks drone bodies."""
    world_id = wp.tid()

    if step_count[world_id] >= max_steps:
        dones[world_id] = 1

    for i in range(num_drone_bodies):
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


def _build_obstacle(builder, collision_group, radius, height, label="obstacle"):
    """Add a static obstacle pillar. Returns body index."""
    import newton as nt

    body = builder.add_body(label=label)
    cfg = nt.ModelBuilder.ShapeConfig(
        density=10000.0, collision_group=collision_group
    )
    builder.add_shape_box(body, hx=radius, hy=radius, hz=height, cfg=cfg)
    builder.add_joint_fixed(parent=-1, child=body)
    return body


# ------------------------------------------------------------------ #
# Unified multi-mode environment
# ------------------------------------------------------------------ #

VALID_MODES = ("fly_to_target", "chase", "intercept", "evade", "adversarial")


class NewtonDroneEnv(VecEnv):
    """
    GPU-accelerated vectorized drone environment with multiple training modes.

    35-dim observation: 19 base + 4 dynamics (closing rate, acceleration) + 12 obstacles.
    3-dim action: velocity commands (vx, vy, vz) in [-1, 1].
    """

    OBS_DIM = 35
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
        self._num_drone_bodies = 2 if self._use_two_drones else 1
        self._num_obstacles = min(self.cfg.num_obstacles, MAX_OBSTACLES)

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
        # bodies_per_world includes drones + obstacles
        self.bodies_per_world = self._num_drone_bodies + self._num_obstacles

        for env_i in range(num_envs):
            builder.begin_world(label=f"world_{env_i}")
            _build_drone(builder, self.arm_length,
                         collision_group=env_i * 3, label=f"drone0_{env_i}")
            if self._use_two_drones:
                _build_drone(builder, self.arm_length,
                             collision_group=env_i * 3 + 1, label=f"drone1_{env_i}")
            # Obstacles get a unique collision group per world (collides with both drones)
            for obs_i in range(self._num_obstacles):
                _build_obstacle(
                    builder,
                    collision_group=env_i * 3 + 2,
                    radius=self.cfg.obstacle_radius,
                    height=self.cfg.obstacle_height,
                    label=f"obstacle_{env_i}_{obs_i}",
                )
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

        # Previous velocity buffer for acceleration computation
        self.prev_vel_buf = wp.zeros(num_envs, dtype=wp.vec3, device=device)

        # Obstacle positions buffer (per-env, per-obstacle)
        self.obstacle_pos_buf = wp.zeros(
            (num_envs, MAX_OBSTACLES), dtype=wp.vec3, device=device
        )

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

        # Initial positions buffer (drones only — obstacles positioned separately)
        self.init_pos_buf = wp.zeros(
            (num_envs, self.bodies_per_world), dtype=wp.vec3, device=device
        )

        # Frozen attacker model(s) for intercept mode
        self.attacker_model = None
        self.attacker_models = []
        self._attacker_assignments = None  # per-env model index for mixture training
        if self.mode == "intercept":
            from stable_baselines3 import SAC
            if self.cfg.attacker_model_paths:
                for path in self.cfg.attacker_model_paths:
                    print(f"  Loading attacker model: {path}")
                    self.attacker_models.append(SAC.load(path))
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
        self.prev_vel_buf.zero_()
        self._randomize_attacker_assignments()
        self._compute_obs()
        return self.obs_buf.numpy().astype(np.float32)

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        actions = self._actions

        if self.mode == "evade":
            vel_np = actions * self.defender_speed_limit
            self.drone1_vel = wp.array(
                [wp.vec3(v[0], v[1], v[2]) for v in vel_np],
                dtype=wp.vec3, device=self.device
            )
            self._compute_drone0_velocity_frozen()
        else:
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

            # Forces on drone 0
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

        # Check bounds/timeout (only check drone bodies, not obstacles)
        wp.launch(
            check_bounds_kernel,
            dim=self.n_envs,
            inputs=(
                self.state_0.body_q, self.step_count_buf,
                self.max_steps, self.cfg.arena_bounds,
                self.cfg.altitude_min, self.cfg.altitude_max,
                self.bodies_per_world,
                self._num_drone_bodies,
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
            # Reset previous velocity for done envs
            prev_vel_np = self.prev_vel_buf.numpy()
            for idx in np.where(dones)[0]:
                prev_vel_np[idx] = [0.0, 0.0, 0.0]
            self.prev_vel_buf = wp.array(
                [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in prev_vel_np],
                dtype=wp.vec3, device=self.device
            )
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
            obs = self._get_attacker_obs()
            vel_np = np.zeros((self.n_envs, 3), dtype=np.float32)

            for model_idx, model in enumerate(self.attacker_models):
                mask = self._attacker_assignments == model_idx
                if not np.any(mask):
                    continue
                if model is None:
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
            obs = self._get_attacker_obs()
            actions, _ = self.attacker_model.predict(obs, deterministic=True)
            vel_np = actions * self.defender_speed_limit
            self.drone1_vel = wp.array(
                [wp.vec3(v[0], v[1], v[2]) for v in vel_np],
                dtype=wp.vec3, device=self.device
            )
        else:
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
        Detects frozen model obs dim and returns matching shape."""
        wp.synchronize()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        target = np.array(self.cfg.target_pos)

        pos_scale = 0.25
        vel_scale = 0.2
        rel_scale = 1.0 / 6.0

        # Detect obs dim from frozen model
        frozen_obs_dim = 19
        if self.attacker_model is not None:
            frozen_obs_dim = self.attacker_model.observation_space.shape[0]
        elif self.attacker_models:
            for m in self.attacker_models:
                if m is not None:
                    frozen_obs_dim = m.observation_space.shape[0]
                    break

        obs = np.zeros((self.n_envs, frozen_obs_dim), dtype=np.float32)
        obs_pos_np = self.obstacle_pos_buf.numpy()

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
            obs[i, 9:12] = 0.0
            obs[i, 12:15] = rel * rel_scale
            obs[i, 15] = dist * rel_scale
            obs[i, 16:19] = def_pos * pos_scale

            if frozen_obs_dim >= 35:
                # Closing rate (defender approaching attacker)
                rel_da = def_pos - att_pos
                dist_da = np.linalg.norm(rel_da)
                closing = 0.0
                if dist_da > 0.01:
                    closing = np.dot(def_vel - att_vel, rel_da / dist_da)
                obs[i, 19] = closing * vel_scale
                obs[i, 20:23] = 0.0  # acceleration (not tracked for frozen models)
                for o in range(min(self._num_obstacles, MAX_OBSTACLES)):
                    base = 23 + o * 4
                    opos = obs_pos_np[i][o]
                    orel = opos - att_pos
                    obs[i, base:base+3] = orel * rel_scale
                    obs[i, base+3] = np.linalg.norm(orel) * rel_scale

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
        Detects frozen model obs dim and returns matching shape."""
        wp.synchronize()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()
        target = np.array(self.cfg.target_pos)

        pos_scale = 0.25
        vel_scale = 0.2
        rel_scale = 1.0 / 6.0

        frozen_obs_dim = 19
        if self.defender_model is not None:
            frozen_obs_dim = self.defender_model.observation_space.shape[0]

        obs = np.zeros((self.n_envs, frozen_obs_dim), dtype=np.float32)
        obs_pos_np = self.obstacle_pos_buf.numpy()

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

            if frozen_obs_dim >= 35:
                closing = 0.0
                if dist > 0.01:
                    closing = np.dot(def_vel - att_vel, rel / dist)
                obs[i, 19] = closing * vel_scale
                obs[i, 20:23] = 0.0
                for o in range(min(self._num_obstacles, MAX_OBSTACLES)):
                    base = 23 + o * 4
                    opos = obs_pos_np[i][o]
                    orel = opos - def_pos
                    obs[i, base:base+3] = orel * rel_scale
                    obs[i, base+3] = np.linalg.norm(orel) * rel_scale

        return obs

    def _update_moving_targets(self):
        """Move targets along their velocity vectors (chase mode)."""
        tgt_np = self.target_positions.numpy()
        vel_np = self.target_velocities.numpy()
        tgt_np += vel_np * self.dt

        bounds = self.cfg.arena_bounds * 0.8
        alt_lo, alt_hi = self.cfg.target_alt_range
        for i in range(self.n_envs):
            for ax in range(2):
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
                    self.prev_vel_buf,
                    self.obstacle_pos_buf, self._num_obstacles,
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
                    self.target_pos_wp,
                    self.prev_vel_buf,
                    self.obstacle_pos_buf, self._num_obstacles,
                    self.obs_buf,
                    self.bodies_per_world, 1, 0,
                ),
                device=self.device,
            )
        else:
            wp.launch(
                compute_obs_kernel,
                dim=self.n_envs,
                inputs=(
                    self.state_0.body_q, self.state_0.body_qd,
                    self.target_pos_wp,
                    self.prev_vel_buf,
                    self.obstacle_pos_buf, self._num_obstacles,
                    self.obs_buf,
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
                    self.obstacle_pos_buf, self._num_obstacles,
                    self.cfg.obstacle_danger_radius,
                    self.cfg.obstacle_proximity_penalty,
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
                    self.obstacle_pos_buf, self._num_obstacles,
                    self.cfg.obstacle_danger_radius,
                    self.cfg.obstacle_proximity_penalty,
                    self.bodies_per_world, 1, 0,
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
                    self.obstacle_pos_buf, self._num_obstacles,
                    self.cfg.obstacle_danger_radius,
                    self.cfg.obstacle_proximity_penalty,
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

        if self._num_obstacles > 0:
            self._randomize_obstacle_positions(rng, range(self.n_envs))

    def _randomize_masked(self, mask):
        """Randomize positions only for done environments."""
        rng = np.random.default_rng()
        indices = [i for i in range(self.n_envs) if mask[i]]

        if self.mode in ("fly_to_target", "chase"):
            self._randomize_single_drone_positions(rng, indices)
        else:
            self._randomize_intercept_positions(rng, indices)

        if self._num_obstacles > 0:
            self._randomize_obstacle_positions(rng, indices)

    def _randomize_obstacle_positions(self, rng, indices):
        """Place obstacles at random positions within the arena, respecting min spacing."""
        obs_np = self.obstacle_pos_buf.numpy()
        positions = self.init_pos_buf.numpy()

        for i in indices:
            placed = []
            for o in range(self._num_obstacles):
                for _attempt in range(50):
                    angle = rng.uniform(0, 2 * np.pi)
                    dist = rng.uniform(0.5, self.cfg.obstacle_placement_range)
                    ox = dist * np.cos(angle)
                    oy = dist * np.sin(angle)
                    oz = self.cfg.obstacle_height  # center of pillar

                    # Check spacing from other obstacles
                    ok = True
                    for px, py, _ in placed:
                        if np.sqrt((ox - px) ** 2 + (oy - py) ** 2) < self.cfg.obstacle_min_spacing:
                            ok = False
                            break
                    if ok:
                        placed.append((ox, oy, oz))
                        break
                else:
                    # Fallback: place anyway
                    placed.append((ox, oy, oz))

                obs_np[i][o] = [placed[-1][0], placed[-1][1], placed[-1][2]]
                # Also set in init_pos_buf for obstacle bodies
                obs_body_idx = self._num_drone_bodies + o
                positions[i, obs_body_idx] = [placed[-1][0], placed[-1][1], placed[-1][2]]

        self.obstacle_pos_buf = wp.array(
            [[wp.vec3(float(obs_np[i][o][0]), float(obs_np[i][o][1]), float(obs_np[i][o][2]))
              for o in range(MAX_OBSTACLES)]
             for i in range(self.n_envs)],
            dtype=wp.vec3, device=self.device
        )
        self.init_pos_buf = wp.array(
            [[wp.vec3(float(positions[i, j, 0]),
                      float(positions[i, j, 1]),
                      float(positions[i, j, 2]))
              for j in range(self.bodies_per_world)]
             for i in range(self.n_envs)],
            dtype=wp.vec3, device=self.device
        )

    def _randomize_single_drone_positions(self, rng, indices):
        """Randomize drone + target positions for fly_to_target / chase."""
        positions = self.init_pos_buf.numpy()
        tgt_np = self.target_positions.numpy()
        vel_np = self.target_velocities.numpy()

        for i in indices:
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.3, 1.0)
            positions[i, 0] = [
                dist * np.cos(angle),
                dist * np.sin(angle),
                rng.uniform(*self.cfg.defender_alt_range),
            ]

            t_angle = rng.uniform(0, 2 * np.pi)
            t_dist = rng.uniform(1.0, self.cfg.target_randomize_range)
            tgt_np[i] = [
                t_dist * np.cos(t_angle),
                t_dist * np.sin(t_angle),
                rng.uniform(*self.cfg.target_alt_range),
            ]

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
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(*self.cfg.defender_dist_range)
            positions[i, 0] = [
                target[0] + dist * np.cos(angle),
                target[1] + dist * np.sin(angle),
                rng.uniform(*self.cfg.defender_alt_range),
            ]
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
