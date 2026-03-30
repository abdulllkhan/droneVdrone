"""
GPU-Accelerated Multi-Drone Swarm Environment (Newton Physics)

Supports NvM scenarios: N defenders vs M attackers.
Uses CTDE (Centralized Training, Decentralized Execution) with parameter sharing —
one SAC model controls all defenders, each getting its own observation.

SB3 sees num_worlds * N "environments" (one per defender per world).

Observation (57-dim, fixed for up to 3v3):
  [0:3]   self pos          [3:6]   self vel
  [6:9]   self acceleration  [9]    closing rate to nearest attacker
  [10:13] target pos
  --- Per-attacker block (3 max, 7 dims each) ---
  [13:16] attacker 0 pos    [16:19] attacker 0 vel    [19] dist to attacker 0
  [20:23] attacker 1 pos    [23:26] attacker 1 vel    [26] dist to attacker 1
  [27:30] attacker 2 pos    [30:33] attacker 2 vel    [33] dist to attacker 2
  --- Per-teammate block (2 max, 7 dims each) ---
  [34:37] teammate 0 pos    [37:40] teammate 0 vel    [40] dist to teammate 0
  [41:44] teammate 1 pos    [44:47] teammate 1 vel    [47] dist to teammate 1
  --- Obstacle block (3 max, 3 dims each) ---
  [48:51] obstacle 0 rel    [51:54] obstacle 1 rel    [54:57] obstacle 2 rel

Action: 3-dim velocity (vx, vy, vz) in [-1, 1]
"""

import numpy as np
import warp as wp
from stable_baselines3.common.vec_env.base_vec_env import VecEnv
from gymnasium import spaces

from train.config import EnvConfig

MAX_ATTACKERS = 3
MAX_DEFENDERS = 3
MAX_OBSTACLES = 3
OBS_DIM = 57  # fixed for up to 3v3


# ------------------------------------------------------------------ #
# Warp Kernels
# ------------------------------------------------------------------ #

@wp.kernel
def apply_velocity_force_multi(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_vel: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
    bodies_per_world: int,
    num_drones: int,
    mass: float,
    gravity: float,
    kp: float,
    kd: float,
    max_force: float,
    angular_damping: float,
):
    """Apply PD velocity forces to all drones (defenders + attackers)."""
    tid = wp.tid()
    world_id = tid // num_drones
    drone_id = tid % num_drones
    body_idx = world_id * bodies_per_world + drone_id

    vel = wp.vec3(
        body_qd[body_idx][0],
        body_qd[body_idx][1],
        body_qd[body_idx][2],
    )
    desired = target_vel[tid]

    vel_error = desired - vel
    ctrl_x = vel_error[0] * kp * mass - vel[0] * kd * mass
    ctrl_y = vel_error[1] * kp * mass - vel[1] * kd * mass
    ctrl_z = vel_error[2] * kp * mass - vel[2] * kd * mass

    ctrl = wp.vec3(ctrl_x, ctrl_y, ctrl_z)
    mag = wp.length(ctrl)
    if mag > max_force:
        ctrl = ctrl * (max_force / mag)

    force = wp.vec3(ctrl[0], ctrl[1], ctrl[2] + mass * gravity)

    ang_vel = wp.vec3(
        body_qd[body_idx][3],
        body_qd[body_idx][4],
        body_qd[body_idx][5],
    )
    torque = ang_vel * (-angular_damping)

    wp.atomic_add(body_f, body_idx, wp.spatial_vector(force, torque))


@wp.kernel
def attacker_pursuit_multi(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_pos: wp.vec3,
    out_vel: wp.array(dtype=wp.vec3),
    bodies_per_world: int,
    num_defenders: int,
    attacker_local_id: int,
    speed: float,
):
    """Scripted attacker pursuit — beeline toward target."""
    world_id = wp.tid()
    att_idx = world_id * bodies_per_world + num_defenders + attacker_local_id
    att_pos = wp.transform_get_translation(body_q[att_idx])
    diff = target_pos - att_pos
    d = wp.length(diff)
    if d > 0.01:
        out_vel[world_id] = diff * (speed / d)
    else:
        out_vel[world_id] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def check_bounds_swarm_kernel(
    body_q: wp.array(dtype=wp.transform),
    step_count: wp.array(dtype=int),
    max_steps: int,
    arena_bounds: float,
    alt_min: float,
    alt_max: float,
    bodies_per_world: int,
    num_drones: int,
    dones: wp.array(dtype=int),
):
    """Check bounds and timeout for all drones in a world."""
    world_id = wp.tid()

    if step_count[world_id] >= max_steps:
        dones[world_id] = 1

    for i in range(num_drones):
        body_idx = world_id * bodies_per_world + i
        pos = wp.transform_get_translation(body_q[body_idx])
        if wp.abs(pos[0]) > arena_bounds or wp.abs(pos[1]) > arena_bounds:
            dones[world_id] = 1
        if pos[2] < alt_min or pos[2] > alt_max:
            dones[world_id] = 1


@wp.kernel
def reset_bodies_swarm_kernel(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    init_positions: wp.array(dtype=wp.vec3, ndim=2),
    reset_mask: wp.array(dtype=int),
    bodies_per_world: int,
    num_bodies: int,
):
    """Reset bodies in done worlds."""
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
# Helper functions
# ------------------------------------------------------------------ #

def _build_drone(builder, arm_length, collision_group, label="drone"):
    """Build a drone body with cross-shaped collision geometry."""
    import newton as nt

    body = builder.add_body(label=label)
    bar_h = arm_length * 0.05
    bar_w = arm_length * 0.05
    density = 781.5  # tuned for ~0.250 kg (250g quadrotor)

    cfg = nt.ModelBuilder.ShapeConfig(
        density=density, collision_group=collision_group
    )
    builder.add_shape_box(body, hx=bar_w, hy=arm_length, hz=bar_h, cfg=cfg)
    builder.add_shape_box(body, hx=arm_length, hy=bar_w, hz=bar_h, cfg=cfg)
    return body


def _build_obstacle(builder, collision_group, radius, height, label="obstacle"):
    """Add a static obstacle pillar."""
    import newton as nt

    body = builder.add_body(label=label)
    cfg = nt.ModelBuilder.ShapeConfig(
        density=10000.0, collision_group=collision_group
    )
    builder.add_shape_box(body, hx=radius, hy=radius, hz=height, cfg=cfg)
    builder.add_joint_fixed(parent=-1, child=body)
    return body


# ------------------------------------------------------------------ #
# Main Swarm Environment
# ------------------------------------------------------------------ #

class NewtonSwarmEnv(VecEnv):
    """N defenders vs M attackers on GPU with Newton/Warp physics.

    SB3 interface: num_envs = num_worlds * num_defenders (CTDE with parameter sharing).
    """

    def __init__(self, num_worlds, config=None):
        import newton as nt

        self.cfg = config or EnvConfig()
        self.num_worlds = num_worlds
        self.N = self.cfg.num_defenders  # defenders per world
        self.M = self.cfg.num_attackers  # attackers per world
        self.n_envs = num_worlds * self.N  # SB3 sees this many "envs"
        self.num_envs = self.n_envs

        self.mode = self.cfg.mode
        self._num_obstacles = min(self.cfg.num_obstacles, MAX_OBSTACLES)
        self._num_drones = self.N + self.M
        self.bodies_per_world = self._num_drones + self._num_obstacles

        # Physics params
        self.fps = self.cfg.ctrl_freq
        self.dt = 1.0 / self.fps
        self.substeps = max(1, self.cfg.pyb_freq // self.cfg.ctrl_freq)
        self.sub_dt = self.dt / self.substeps
        self.arm_length = 0.2
        self.max_steps = int(self.cfg.episode_secs * self.cfg.ctrl_freq)
        self.target_pos_wp = wp.vec3(*self.cfg.target_pos)
        self.target_pos_np = np.array(self.cfg.target_pos, dtype=np.float32)

        self.drone_mass = self.cfg.drone_mass
        self.gravity = 9.81

        self.defender_speed_limit = 0.15 * 30.0 * (1000.0 / 3600.0)  # ~1.25 m/s
        self.attacker_speed_limit = self.cfg.attacker_speed_limit
        self.attacker_speed = self.cfg.attacker_speed_scale * 0.8

        # Force model
        self.force_kp = self.cfg.force_kp
        self.force_kd = self.cfg.force_kd
        self.max_force = self.cfg.max_force
        self.angular_damping = self.cfg.angular_damping

        # Domain randomization
        self.wind_enabled = self.cfg.wind_enabled
        self.wind_max = self.cfg.wind_max_force
        self.wind_change_freq = self.cfg.wind_change_freq
        self.obs_noise_std = self.cfg.obs_noise_std

        # Build scene
        builder = nt.ModelBuilder(gravity=-self.gravity)

        for env_i in range(num_worlds):
            groups_per_world = self.N + self.M + 1
            base_group = env_i * groups_per_world

            # Defenders
            for d in range(self.N):
                _build_drone(builder, self.arm_length,
                             collision_group=base_group + d,
                             label=f"defender_{env_i}_{d}")

            # Attackers
            for a in range(self.M):
                _build_drone(builder, self.arm_length,
                             collision_group=base_group + self.N + a,
                             label=f"attacker_{env_i}_{a}")

            # Obstacles
            for o in range(self._num_obstacles):
                _build_obstacle(builder,
                                collision_group=base_group + self.N + self.M,
                                radius=self.cfg.obstacle_radius,
                                height=self.cfg.obstacle_height,
                                label=f"obstacle_{env_i}_{o}")

        self.device = "cuda:0"
        self.model = builder.finalize(device=self.device)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Get actual drone mass from physics
        body_masses = self.model.body_mass.numpy()
        self.drone_mass = float(body_masses[0])

        # Solver
        self.solver = nt.solvers.SolverSemiImplicit(self.model)

        # Buffers
        self.drone_vel = wp.zeros(num_worlds * self._num_drones,
                                  dtype=wp.vec3, device=self.device)
        self.obs_buf = np.zeros((self.n_envs, OBS_DIM), dtype=np.float32)
        self.prev_vel = np.zeros((num_worlds, self.N, 3), dtype=np.float32)

        # Per-world state
        self.step_count_buf = wp.zeros(num_worlds, dtype=int, device=self.device)
        self.prev_dist_buf = np.zeros((num_worlds, self.N), dtype=np.float32)
        self.eliminated_attackers = np.zeros((num_worlds, self.M), dtype=bool)

        # Init positions buffer
        self.init_pos_buf = wp.zeros((num_worlds, self.bodies_per_world),
                                     dtype=wp.vec3, device=self.device)

        # Obstacle position buffer
        self.obstacle_pos_buf = wp.zeros((num_worlds, MAX_OBSTACLES),
                                         dtype=wp.vec3, device=self.device)

        # Wind state
        self.wind_force = np.zeros((num_worlds, 3), dtype=np.float32)
        self.wind_step_counter = 0

        # Load frozen attacker models
        self.attacker_models = []
        if self.cfg.attacker_model_path:
            from stable_baselines3 import SAC
            m = SAC.load(self.cfg.attacker_model_path, device="cpu")
            self.attacker_models = [m]
        elif self.cfg.attacker_model_paths:
            from stable_baselines3 import SAC
            for p in self.cfg.attacker_model_paths:
                self.attacker_models.append(SAC.load(p, device="cpu"))

        # Spaces
        obs_space = spaces.Box(-np.inf, np.inf, shape=(OBS_DIM,), dtype=np.float32)
        act_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

        super().__init__(self.n_envs, obs_space, act_space)

    def reset(self):
        self._randomize_all()
        self._apply_init_positions(mask=None)
        self.prev_dist_buf[:] = 0.0
        self.step_count_buf.zero_()
        self.prev_vel[:] = 0.0
        self.eliminated_attackers[:] = False
        self._randomize_wind()
        self._compute_obs()
        return self.obs_buf.copy()

    def step_async(self, actions):
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        actions = self._actions  # (n_envs, 3) = (num_worlds * N, 3)

        # Reshape actions to (num_worlds, N, 3)
        act_per_world = actions.reshape(self.num_worlds, self.N, 3)

        # Build velocity array for all drones: (num_worlds * (N+M), 3)
        vel_np = np.zeros((self.num_worlds * self._num_drones, 3), dtype=np.float32)

        # Set defender velocities from actions
        for w in range(self.num_worlds):
            for d in range(self.N):
                idx = w * self._num_drones + d
                vel_np[idx] = act_per_world[w, d] * self.defender_speed_limit

        # Compute attacker velocities
        self._compute_attacker_velocities(vel_np)

        # Set velocity buffer
        self.drone_vel = wp.array(
            [wp.vec3(float(v[0]), float(v[1]), float(v[2])) for v in vel_np],
            dtype=wp.vec3, device=self.device
        )

        # Physics substeps
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            wp.launch(
                apply_velocity_force_multi,
                dim=self.num_worlds * self._num_drones,
                inputs=(
                    self.state_0.body_q, self.state_0.body_qd,
                    self.drone_vel, self.state_0.body_f,
                    self.bodies_per_world, self._num_drones,
                    self.drone_mass, self.gravity,
                    self.force_kp, self.force_kd,
                    self.max_force, self.angular_damping,
                ),
                device=self.device,
            )

            # Apply wind if enabled
            if self.wind_enabled:
                self._apply_wind()

            self.solver.step(self.state_0, self.state_1, None, None, self.sub_dt)
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

        # Increment step counter
        step_np = self.step_count_buf.numpy()
        step_np += 1
        self.step_count_buf = wp.array(step_np, dtype=int, device=self.device)

        # Compute rewards and check termination
        rewards, dones, infos = self._compute_rewards_and_dones()

        # Check bounds
        dones_wp = wp.array(dones.astype(int), dtype=int, device=self.device)
        wp.launch(
            check_bounds_swarm_kernel,
            dim=self.num_worlds,
            inputs=(
                self.state_0.body_q, self.step_count_buf,
                self.max_steps, self.cfg.arena_bounds,
                self.cfg.altitude_min, self.cfg.altitude_max,
                self.bodies_per_world, self._num_drones,
                dones_wp,
            ),
            device=self.device,
        )
        wp.synchronize()
        world_dones = dones_wp.numpy()

        # Auto-reset done worlds
        if np.any(world_dones):
            self._randomize_masked(world_dones)
            self._apply_init_positions(world_dones)
            for w in range(self.num_worlds):
                if world_dones[w]:
                    step_np[w] = 0
                    self.prev_dist_buf[w] = 0.0
                    self.prev_vel[w] = 0.0
                    self.eliminated_attackers[w] = False
            self.step_count_buf = wp.array(step_np, dtype=int, device=self.device)

        # Update wind periodically
        self.wind_step_counter += 1
        if self.wind_enabled and self.wind_step_counter % self.wind_change_freq == 0:
            self._randomize_wind()

        # Compute observations
        self._compute_obs()

        # Expand world dones to per-defender dones
        per_agent_dones = np.repeat(world_dones, self.N).astype(bool)
        per_agent_rewards = rewards  # already (num_worlds * N,)

        return self.obs_buf.copy(), per_agent_rewards, per_agent_dones, infos

    def _compute_attacker_velocities(self, vel_np):
        """Set attacker velocities — scripted pursuit or frozen RL model."""
        wp.synchronize()

        for a in range(self.M):
            # Scripted pursuit for each attacker
            temp_vel = wp.zeros(self.num_worlds, dtype=wp.vec3, device=self.device)
            wp.launch(
                attacker_pursuit_multi,
                dim=self.num_worlds,
                inputs=(
                    self.state_0.body_q, self.state_0.body_qd,
                    self.target_pos_wp, temp_vel,
                    self.bodies_per_world, self.N, a,
                    self.attacker_speed,
                ),
                device=self.device,
            )
            wp.synchronize()
            scripted = np.array([[v[0], v[1], v[2]] for v in temp_vel.numpy()])

            for w in range(self.num_worlds):
                idx = w * self._num_drones + self.N + a
                if not self.eliminated_attackers[w, a]:
                    vel_np[idx] = scripted[w]
                # eliminated attackers keep zero velocity

    def _compute_rewards_and_dones(self):
        """Compute per-defender rewards and check intercept/target-reached."""
        wp.synchronize()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        rewards = np.zeros(self.n_envs, dtype=np.float32)
        world_dones = np.zeros(self.num_worlds, dtype=int)
        infos = [{} for _ in range(self.n_envs)]

        target = self.target_pos_np.copy()
        # Use current (possibly randomized) target
        t = self.target_pos_wp
        target = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)

        for w in range(self.num_worlds):
            # Get all positions
            def_positions = []
            def_velocities = []
            for d in range(self.N):
                bidx = w * self.bodies_per_world + d
                def_positions.append(body_q[bidx][:3])
                def_velocities.append(body_qd[bidx][:3])

            att_positions = []
            for a in range(self.M):
                bidx = w * self.bodies_per_world + self.N + a
                att_positions.append(body_q[bidx][:3])

            any_intercept = False
            any_target_reached = False

            # Check intercept pairs
            for d in range(self.N):
                for a in range(self.M):
                    if self.eliminated_attackers[w, a]:
                        continue
                    dist = np.linalg.norm(def_positions[d] - att_positions[a])
                    if dist < self.cfg.intercept_dist:
                        self.eliminated_attackers[w, a] = True
                        any_intercept = True

            # Check target reached
            for a in range(self.M):
                if self.eliminated_attackers[w, a]:
                    continue
                dist_to_target = np.linalg.norm(att_positions[a] - target)
                if dist_to_target < self.cfg.target_reached_dist:
                    any_target_reached = True

            # Check if all attackers eliminated
            all_eliminated = np.all(self.eliminated_attackers[w])

            # Per-defender rewards
            for d in range(self.N):
                agent_idx = w * self.N + d
                reward = 0.0

                if any_intercept:
                    reward += 100.0  # team intercept bonus

                if any_target_reached:
                    reward -= 200.0  # team failure penalty

                if all_eliminated:
                    reward += 200.0  # mission complete bonus
                    world_dones[w] = 1

                if any_target_reached:
                    world_dones[w] = 1

                # Dense reward: approach nearest active attacker
                min_dist = float('inf')
                for a in range(self.M):
                    if self.eliminated_attackers[w, a]:
                        continue
                    dist = np.linalg.norm(def_positions[d] - att_positions[a])
                    if dist < min_dist:
                        min_dist = dist

                if min_dist < float('inf'):
                    prev = self.prev_dist_buf[w, d]
                    if prev > 0:
                        reward += (prev - min_dist) * 50.0
                    self.prev_dist_buf[w, d] = min_dist

                    # Proximity bonus
                    reward += 1.0 / (min_dist + 0.1) - 1.0 / (4.0 + 0.1)

                # Coordination: reward spread between defenders
                if self.N > 1:
                    for d2 in range(self.N):
                        if d2 != d:
                            dd = np.linalg.norm(def_positions[d] - def_positions[d2])
                            reward += self.cfg.coordination_bonus * min(dd, 2.0)

                # Time penalty
                reward -= 0.05

                # Obstacle penalties
                if self._num_obstacles > 0:
                    obs_pos = self.obstacle_pos_buf.numpy()
                    for o in range(self._num_obstacles):
                        opos = obs_pos[w][o]
                        odist = np.linalg.norm(def_positions[d] - opos)
                        if odist < self.cfg.obstacle_collision_radius:
                            reward -= self.cfg.obstacle_collision_penalty
                        elif odist < self.cfg.obstacle_danger_radius:
                            reward -= (self.cfg.obstacle_danger_radius - odist) * self.cfg.obstacle_proximity_penalty

                rewards[agent_idx] = reward

                infos[agent_idx] = {
                    "intercepted": int(any_intercept),
                    "target_reached": int(any_target_reached),
                    "all_eliminated": int(all_eliminated),
                    "active_attackers": int(self.M - np.sum(self.eliminated_attackers[w])),
                }

        return rewards, world_dones, infos

    def _compute_obs(self):
        """Build per-defender observations with distance-sorted attackers/teammates."""
        wp.synchronize()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        pos_scale = 0.25
        vel_scale = 0.2
        rel_scale = 1.0 / 6.0

        t = self.target_pos_wp
        target = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)

        for w in range(self.num_worlds):
            # Gather all drone data
            positions = []
            velocities = []
            for i in range(self._num_drones):
                bidx = w * self.bodies_per_world + i
                positions.append(body_q[bidx][:3].astype(np.float32))
                velocities.append(body_qd[bidx][:3].astype(np.float32))

            for d in range(self.N):
                agent_idx = w * self.N + d
                obs = np.zeros(OBS_DIM, dtype=np.float32)

                my_pos = positions[d]
                my_vel = velocities[d]

                # Self state [0:9]
                obs[0:3] = my_pos * pos_scale
                obs[3:6] = my_vel * vel_scale
                # Acceleration [6:9]
                acc = my_vel - self.prev_vel[w, d]
                obs[6:9] = acc * vel_scale
                self.prev_vel[w, d] = my_vel.copy()

                # Closing rate to nearest attacker [9]
                min_att_dist = float('inf')
                for a in range(self.M):
                    if self.eliminated_attackers[w, a]:
                        continue
                    att_pos = positions[self.N + a]
                    dist = np.linalg.norm(my_pos - att_pos)
                    if dist < min_att_dist:
                        min_att_dist = dist
                        att_vel = velocities[self.N + a]
                        rel = att_pos - my_pos
                        if dist > 0.01:
                            closing = np.dot(my_vel - att_vel, rel / dist)
                            obs[9] = closing * vel_scale

                # Target position [10:13]
                obs[10:13] = target * pos_scale

                # Per-attacker block [13:34] (sorted by distance)
                att_dists = []
                for a in range(self.M):
                    att_pos = positions[self.N + a]
                    dist = np.linalg.norm(my_pos - att_pos) if not self.eliminated_attackers[w, a] else 999.0
                    att_dists.append((dist, a))
                att_dists.sort()

                for k in range(min(self.M, MAX_ATTACKERS)):
                    dist, a = att_dists[k]
                    base = 13 + k * 7
                    if not self.eliminated_attackers[w, a]:
                        att_pos = positions[self.N + a]
                        att_vel = velocities[self.N + a]
                        obs[base:base+3] = att_pos * pos_scale
                        obs[base+3:base+6] = att_vel * vel_scale
                        obs[base+6] = dist * rel_scale

                # Per-teammate block [34:48] (sorted by distance)
                tm_dists = []
                for d2 in range(self.N):
                    if d2 == d:
                        continue
                    tm_pos = positions[d2]
                    dist = np.linalg.norm(my_pos - tm_pos)
                    tm_dists.append((dist, d2))
                tm_dists.sort()

                for k in range(min(self.N - 1, MAX_DEFENDERS - 1)):
                    dist, d2 = tm_dists[k]
                    base = 34 + k * 7
                    obs[base:base+3] = positions[d2] * pos_scale
                    obs[base+3:base+6] = velocities[d2] * vel_scale
                    obs[base+6] = dist * rel_scale

                # Obstacle block [48:57]
                if self._num_obstacles > 0:
                    obs_pos = self.obstacle_pos_buf.numpy()
                    for o in range(self._num_obstacles):
                        base = 48 + o * 3
                        orel = obs_pos[w][o] - my_pos
                        obs[base:base+3] = orel * rel_scale

                # Add observation noise
                if self.obs_noise_std > 0:
                    obs += np.random.normal(0, self.obs_noise_std, OBS_DIM).astype(np.float32)

                self.obs_buf[agent_idx] = obs

    def _randomize_all(self):
        """Randomize all world positions."""
        rng = np.random.default_rng()
        self._randomize_positions(rng, range(self.num_worlds))
        if self._num_obstacles > 0:
            self._randomize_obstacle_positions(rng, range(self.num_worlds))

    def _randomize_masked(self, mask):
        """Randomize positions for done worlds only."""
        rng = np.random.default_rng()
        indices = [i for i in range(self.num_worlds) if mask[i]]
        self._randomize_positions(rng, indices)
        if self._num_obstacles > 0:
            self._randomize_obstacle_positions(rng, indices)

    def _randomize_positions(self, rng, indices):
        """Randomize drone positions for intercept scenarios."""
        positions = self.init_pos_buf.numpy()
        target = np.array(self.cfg.target_pos, dtype=np.float32)

        # Randomize target altitude
        if self.cfg.target_alt_randomize:
            alt = rng.uniform(*self.cfg.target_alt_range_intercept)
            target[2] = alt
            self.target_pos_wp = wp.vec3(float(target[0]), float(target[1]), float(alt))
            self.target_pos_np = target.copy()

        for i in indices:
            # Spread defenders around target
            for d in range(self.N):
                angle = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(*self.cfg.defender_dist_range)
                positions[i, d] = [
                    target[0] + dist * np.cos(angle),
                    target[1] + dist * np.sin(angle),
                    rng.uniform(*self.cfg.defender_alt_range),
                ]

            # Spread attackers at distance
            for a in range(self.M):
                angle = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(*self.cfg.attacker_dist_range)
                positions[i, self.N + a] = [
                    target[0] + dist * np.cos(angle),
                    target[1] + dist * np.sin(angle),
                    rng.uniform(*self.cfg.attacker_alt_range),
                ]

        self.init_pos_buf = wp.array(
            [[wp.vec3(float(positions[i, j, 0]),
                      float(positions[i, j, 1]),
                      float(positions[i, j, 2]))
              for j in range(self.bodies_per_world)]
             for i in range(self.num_worlds)],
            dtype=wp.vec3, device=self.device
        )

    def _randomize_obstacle_positions(self, rng, indices):
        """Place obstacles at random positions within the arena."""
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
                    oz = self.cfg.obstacle_height

                    ok = True
                    for px, py, _ in placed:
                        if np.sqrt((ox - px)**2 + (oy - py)**2) < self.cfg.obstacle_min_spacing:
                            ok = False
                            break
                    if ok:
                        placed.append((ox, oy, oz))
                        break
                else:
                    placed.append((ox, oy, oz))

                obs_np[i][o] = [placed[-1][0], placed[-1][1], placed[-1][2]]
                obs_body_idx = self._num_drones + o
                positions[i, obs_body_idx] = [placed[-1][0], placed[-1][1], placed[-1][2]]

        self.obstacle_pos_buf = wp.array(
            [[wp.vec3(float(obs_np[i][o][0]), float(obs_np[i][o][1]), float(obs_np[i][o][2]))
              for o in range(MAX_OBSTACLES)]
             for i in range(self.num_worlds)],
            dtype=wp.vec3, device=self.device
        )
        self.init_pos_buf = wp.array(
            [[wp.vec3(float(positions[i, j, 0]),
                      float(positions[i, j, 1]),
                      float(positions[i, j, 2]))
              for j in range(self.bodies_per_world)]
             for i in range(self.num_worlds)],
            dtype=wp.vec3, device=self.device
        )

    def _apply_init_positions(self, mask=None):
        if mask is None:
            mask_wp = wp.ones(self.num_worlds, dtype=int, device=self.device)
        else:
            mask_wp = wp.array(mask, dtype=int, device=self.device)

        wp.launch(
            reset_bodies_swarm_kernel,
            dim=self.num_worlds,
            inputs=(
                self.state_0.body_q, self.state_0.body_qd,
                self.init_pos_buf, mask_wp,
                self.bodies_per_world, self.bodies_per_world,
            ),
            device=self.device,
        )
        wp.synchronize()

    def _randomize_wind(self):
        """Generate random wind direction and magnitude."""
        if self.wind_enabled:
            for w in range(self.num_worlds):
                angle = np.random.uniform(0, 2 * np.pi)
                mag = np.random.uniform(0, self.wind_max)
                self.wind_force[w] = [mag * np.cos(angle), mag * np.sin(angle), 0.0]

    def _apply_wind(self):
        """Apply wind forces to all drones."""
        body_f_np = self.state_0.body_f.numpy()
        for w in range(self.num_worlds):
            for i in range(self._num_drones):
                bidx = w * self.bodies_per_world + i
                body_f_np[bidx][0] += self.wind_force[w][0]
                body_f_np[bidx][1] += self.wind_force[w][1]
                body_f_np[bidx][2] += self.wind_force[w][2]
        self.state_0.body_f = wp.array(body_f_np, dtype=wp.spatial_vector, device=self.device)

    def close(self):
        pass

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.n_envs

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return []

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * self.n_envs

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def seed(self, seed=None):
        pass
