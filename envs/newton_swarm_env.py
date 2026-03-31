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
        # In evade mode, RL controls attackers; in intercept mode, RL controls defenders
        if self.cfg.mode == "evade":
            self.n_envs = num_worlds * self.M  # SB3 sees M attackers per world
            self.rl_agents_per_world = self.M
        else:
            self.n_envs = num_worlds * self.N  # SB3 sees N defenders per world
            self.rl_agents_per_world = self.N
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
        self.prev_vel = np.zeros((num_worlds, self.rl_agents_per_world, 3), dtype=np.float32)

        # Per-world state
        self.step_count_buf = wp.zeros(num_worlds, dtype=int, device=self.device)
        self.prev_dist_buf = np.zeros((num_worlds, self.rl_agents_per_world), dtype=np.float32)
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

        # Load frozen opponent models
        self.attacker_models = []
        self.defender_model = None
        if self.mode == "evade" and self.cfg.defender_model_path:
            from stable_baselines3 import SAC
            self.defender_model = SAC.load(self.cfg.defender_model_path, device="cpu")
        elif self.cfg.attacker_model_path:
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
        actions = self._actions  # (n_envs, 3)

        # Build velocity array for all drones: (num_worlds * (N+M), 3)
        vel_np = np.zeros((self.num_worlds * self._num_drones, 3), dtype=np.float32)

        if self.mode == "evade":
            # RL controls attackers, frozen model controls defenders
            act_per_world = actions.reshape(self.num_worlds, self.M, 3)
            for w in range(self.num_worlds):
                for a in range(self.M):
                    idx = w * self._num_drones + self.N + a
                    if not self.eliminated_attackers[w, a]:
                        vel_np[idx] = act_per_world[w, a] * self.attacker_speed_limit
            self._compute_defender_velocities(vel_np)
        else:
            # RL controls defenders, scripted/frozen attackers
            act_per_world = actions.reshape(self.num_worlds, self.N, 3)
            for w in range(self.num_worlds):
                for d in range(self.N):
                    idx = w * self._num_drones + d
                    vel_np[idx] = act_per_world[w, d] * self.defender_speed_limit
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

        # Expand world dones to per-agent dones
        per_agent_dones = np.repeat(world_dones, self.rl_agents_per_world).astype(bool)
        per_agent_rewards = rewards  # already (num_worlds * rl_agents_per_world,)

        return self.obs_buf.copy(), per_agent_rewards, per_agent_dones, infos

    def _compute_attacker_velocities(self, vel_np):
        """Set attacker velocities — scripted pursuit or frozen RL model."""
        wp.synchronize()

        if self.attacker_models:
            # Use frozen RL attacker model — build attacker obs and predict
            self._compute_obs_for_attackers()
            model = self.attacker_models[0]  # use first model for all
            actions, _ = model.predict(self._attacker_obs, deterministic=True)
            for w in range(self.num_worlds):
                for a in range(self.M):
                    idx = w * self._num_drones + self.N + a
                    agent_idx = w * self.M + a
                    if not self.eliminated_attackers[w, a]:
                        vel_np[idx] = actions[agent_idx] * self.attacker_speed_limit
            return

        strategy = self.cfg.attacker_strategy

        if strategy == "flanking" and self.M > 1:
            self._compute_flanking_attacker_velocities(vel_np)
        elif strategy == "mixed" and self.M > 1:
            self._compute_mixed_attacker_velocities(vel_np)
        else:
            # Direct pursuit (default)
            self._compute_direct_attacker_velocities(vel_np)

    def _compute_direct_attacker_velocities(self, vel_np):
        """All attackers beeline toward target."""
        for a in range(self.M):
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

    def _compute_flanking_attacker_velocities(self, vel_np):
        """Attackers approach target from spread-out angles to overwhelm defenders.

        Each attacker aims for a waypoint offset from the target, converging
        only when close. This forces defenders to split up.
        """
        body_q = self.state_0.body_q.numpy()
        target = self.target_pos_np

        for w in range(self.num_worlds):
            # Count active attackers for angle spacing
            active = [a for a in range(self.M) if not self.eliminated_attackers[w, a]]
            if not active:
                continue

            for i, a in enumerate(active):
                bidx = w * self.bodies_per_world + self.N + a
                att_pos = body_q[bidx][:3]
                dist_to_target = np.linalg.norm(att_pos - target)

                if dist_to_target < 1.0:
                    # Close enough — go direct
                    direction = target - att_pos
                    d = np.linalg.norm(direction)
                    if d > 0.01:
                        direction = direction / d
                else:
                    # Flanking: each attacker takes a different angular offset
                    angle_offset = (2 * np.pi * i / len(active))
                    flank_radius = min(1.5, dist_to_target * 0.4)
                    waypoint = target.copy()
                    waypoint[0] += flank_radius * np.cos(angle_offset)
                    waypoint[1] += flank_radius * np.sin(angle_offset)

                    direction = waypoint - att_pos
                    d = np.linalg.norm(direction)
                    if d > 0.01:
                        direction = direction / d

                idx = w * self._num_drones + self.N + a
                vel_np[idx] = direction * self.attacker_speed

    def _compute_mixed_attacker_velocities(self, vel_np):
        """Mixed strategy: first attacker flanks, rest go direct. Randomized per reset.

        In multi-attacker scenarios, a mix of direct and flanking creates
        harder-to-predict attack patterns for the defender.
        """
        body_q = self.state_0.body_q.numpy()
        target = self.target_pos_np

        for w in range(self.num_worlds):
            for a in range(self.M):
                if self.eliminated_attackers[w, a]:
                    continue

                bidx = w * self.bodies_per_world + self.N + a
                att_pos = body_q[bidx][:3]
                dist_to_target = np.linalg.norm(att_pos - target)

                # Even-indexed attackers flank, odd go direct
                if a % 2 == 0 and dist_to_target > 1.0:
                    angle_offset = (2 * np.pi * a / self.M) + np.pi / 4
                    flank_radius = min(1.5, dist_to_target * 0.4)
                    waypoint = target.copy()
                    waypoint[0] += flank_radius * np.cos(angle_offset)
                    waypoint[1] += flank_radius * np.sin(angle_offset)
                    direction = waypoint - att_pos
                else:
                    direction = target - att_pos

                d = np.linalg.norm(direction)
                if d > 0.01:
                    direction = direction / d

                idx = w * self._num_drones + self.N + a
                vel_np[idx] = direction * self.attacker_speed

    def _compute_defender_velocities(self, vel_np):
        """Set defender velocities from frozen model (evade mode)."""
        if self.defender_model is None:
            # No frozen defender — scripted pursuit toward nearest attacker
            wp.synchronize()
            body_q = self.state_0.body_q.numpy()
            for w in range(self.num_worlds):
                for d in range(self.N):
                    bidx = w * self.bodies_per_world + d
                    def_pos = body_q[bidx][:3]
                    # Find nearest active attacker
                    min_dist = float('inf')
                    nearest_att_pos = None
                    for a in range(self.M):
                        if self.eliminated_attackers[w, a]:
                            continue
                        abidx = w * self.bodies_per_world + self.N + a
                        att_pos = body_q[abidx][:3]
                        dist = np.linalg.norm(def_pos - att_pos)
                        if dist < min_dist:
                            min_dist = dist
                            nearest_att_pos = att_pos
                    if nearest_att_pos is not None and min_dist > 0.01:
                        direction = nearest_att_pos - def_pos
                        direction = direction / np.linalg.norm(direction)
                        idx = w * self._num_drones + d
                        vel_np[idx] = direction * self.defender_speed_limit
            return

        # Use frozen RL model — build defender observations and predict
        self._compute_obs_for_defenders()
        actions, _ = self.defender_model.predict(self._defender_obs, deterministic=True)
        for w in range(self.num_worlds):
            for d in range(self.N):
                idx = w * self._num_drones + d
                agent_idx = w * self.N + d
                vel_np[idx] = actions[agent_idx] * self.defender_speed_limit

    def _compute_obs_for_defenders(self):
        """Build observations for frozen defender model (intercept-mode obs format)."""
        wp.synchronize()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        pos_scale = 0.25
        vel_scale = 0.2
        rel_scale = 1.0 / 6.0

        t = self.target_pos_wp
        target = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)

        self._defender_obs = np.zeros((self.num_worlds * self.N, OBS_DIM), dtype=np.float32)

        for w in range(self.num_worlds):
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

                obs[0:3] = my_pos * pos_scale
                obs[3:6] = my_vel * vel_scale
                obs[10:13] = target * pos_scale

                # Attackers sorted by distance
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
                        obs[base:base+3] = positions[self.N + a] * pos_scale
                        obs[base+3:base+6] = velocities[self.N + a] * vel_scale
                        obs[base+6] = dist * rel_scale

                # Teammates
                tm_dists = []
                for d2 in range(self.N):
                    if d2 == d:
                        continue
                    dist = np.linalg.norm(my_pos - positions[d2])
                    tm_dists.append((dist, d2))
                tm_dists.sort()

                for k in range(min(self.N - 1, MAX_DEFENDERS - 1)):
                    dist, d2 = tm_dists[k]
                    base = 34 + k * 7
                    obs[base:base+3] = positions[d2] * pos_scale
                    obs[base+3:base+6] = velocities[d2] * vel_scale
                    obs[base+6] = dist * rel_scale

                if self._num_obstacles > 0:
                    obs_pos = self.obstacle_pos_buf.numpy()
                    for o in range(self._num_obstacles):
                        base = 48 + o * 3
                        obs[base:base+3] = (obs_pos[w][o] - my_pos) * rel_scale

                self._defender_obs[agent_idx] = obs

    def _compute_obs_for_attackers(self):
        """Build attacker-centric observations for frozen attacker model."""
        wp.synchronize()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        pos_scale = 0.25
        vel_scale = 0.2
        rel_scale = 1.0 / 6.0

        t = self.target_pos_wp
        target = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)

        self._attacker_obs = np.zeros((self.num_worlds * self.M, OBS_DIM), dtype=np.float32)

        for w in range(self.num_worlds):
            positions = []
            velocities = []
            for i in range(self._num_drones):
                bidx = w * self.bodies_per_world + i
                positions.append(body_q[bidx][:3].astype(np.float32))
                velocities.append(body_qd[bidx][:3].astype(np.float32))

            for a in range(self.M):
                agent_idx = w * self.M + a
                obs = np.zeros(OBS_DIM, dtype=np.float32)
                my_pos = positions[self.N + a]
                my_vel = velocities[self.N + a]

                obs[0:3] = my_pos * pos_scale
                obs[3:6] = my_vel * vel_scale
                obs[10:13] = target * pos_scale

                # Defenders as "opponents" sorted by distance
                def_dists = []
                for d in range(self.N):
                    dist = np.linalg.norm(my_pos - positions[d])
                    def_dists.append((dist, d))
                def_dists.sort()

                for k in range(min(self.N, MAX_ATTACKERS)):
                    dist, d = def_dists[k]
                    base = 13 + k * 7
                    obs[base:base+3] = positions[d] * pos_scale
                    obs[base+3:base+6] = velocities[d] * vel_scale
                    obs[base+6] = dist * rel_scale

                # Teammate attackers
                tm_dists = []
                for a2 in range(self.M):
                    if a2 == a or self.eliminated_attackers[w, a2]:
                        continue
                    dist = np.linalg.norm(my_pos - positions[self.N + a2])
                    tm_dists.append((dist, a2))
                tm_dists.sort()

                for k in range(min(len(tm_dists), MAX_DEFENDERS - 1)):
                    dist, a2 = tm_dists[k]
                    base = 34 + k * 7
                    obs[base:base+3] = positions[self.N + a2] * pos_scale
                    obs[base+3:base+6] = velocities[self.N + a2] * vel_scale
                    obs[base+6] = dist * rel_scale

                if self._num_obstacles > 0:
                    obs_pos = self.obstacle_pos_buf.numpy()
                    for o in range(self._num_obstacles):
                        base = 48 + o * 3
                        obs[base:base+3] = (obs_pos[w][o] - my_pos) * rel_scale

                self._attacker_obs[agent_idx] = obs

    def _compute_rewards_and_dones(self):
        """Compute per-agent rewards and check intercept/target-reached."""
        wp.synchronize()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        rewards = np.zeros(self.n_envs, dtype=np.float32)
        world_dones = np.zeros(self.num_worlds, dtype=int)
        infos = [{} for _ in range(self.n_envs)]

        t = self.target_pos_wp
        target = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)

        # Hoist obstacle positions out of inner loops (avoid repeated GPU→CPU transfers)
        obs_pos_np = self.obstacle_pos_buf.numpy() if self._num_obstacles > 0 else None

        for w in range(self.num_worlds):
            # Get all positions
            def_positions = []
            def_velocities = []
            for d in range(self.N):
                bidx = w * self.bodies_per_world + d
                def_positions.append(body_q[bidx][:3])
                def_velocities.append(body_qd[bidx][:3])

            att_positions = []
            att_velocities = []
            for a in range(self.M):
                bidx = w * self.bodies_per_world + self.N + a
                att_positions.append(body_q[bidx][:3])
                att_velocities.append(body_qd[bidx][:3])

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

            all_eliminated = np.all(self.eliminated_attackers[w])

            if self.mode == "evade":
                # --- Attacker rewards (evade mode) ---
                for a in range(self.M):
                    agent_idx = w * self.M + a
                    reward = 0.0

                    if any_target_reached:
                        reward += 200.0  # team success: reached target
                    if any_intercept:
                        reward -= 100.0  # team penalty: got intercepted

                    if all_eliminated:
                        reward -= 200.0  # mission failure
                        world_dones[w] = 1
                    if any_target_reached:
                        world_dones[w] = 1

                    # Dense reward: approach target
                    if not self.eliminated_attackers[w, a]:
                        dist_to_target = np.linalg.norm(att_positions[a] - target)
                        prev = self.prev_dist_buf[w, a]
                        if prev > 0:
                            reward += (prev - dist_to_target) * 50.0
                        self.prev_dist_buf[w, a] = dist_to_target

                        # Evasion bonus: distance from nearest defender
                        min_def_dist = float('inf')
                        for d in range(self.N):
                            dd = np.linalg.norm(att_positions[a] - def_positions[d])
                            if dd < min_def_dist:
                                min_def_dist = dd
                        if min_def_dist < 2.0:
                            reward += (min_def_dist - 0.5) * 5.0  # reward staying away

                    # Time penalty (slight urgency)
                    reward -= 0.05

                    # Obstacle penalties for attackers too
                    if obs_pos_np is not None and not self.eliminated_attackers[w, a]:
                        for o in range(self._num_obstacles):
                            opos = obs_pos_np[w][o]
                            odist = np.linalg.norm(att_positions[a] - opos)
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
            else:
                # --- Defender rewards (intercept mode) ---

                # Pre-compute target assignments for diversity bonus
                active_attackers = [a for a in range(self.M) if not self.eliminated_attackers[w, a]]
                nearest_attacker = []  # which attacker each defender is nearest to
                for d in range(self.N):
                    best_a = -1
                    best_dist = float('inf')
                    for a in active_attackers:
                        dist = np.linalg.norm(def_positions[d] - att_positions[a])
                        if dist < best_dist:
                            best_dist = dist
                            best_a = a
                    nearest_attacker.append(best_a)

                # Count how many defenders target each attacker
                assignment_counts = {}
                for a_id in nearest_attacker:
                    if a_id >= 0:
                        assignment_counts[a_id] = assignment_counts.get(a_id, 0) + 1
                # Diversity: bonus if each defender targets a different attacker
                unique_targets = len(assignment_counts)
                n_active = len(active_attackers)

                for d in range(self.N):
                    agent_idx = w * self.N + d
                    reward = 0.0

                    if any_intercept:
                        reward += 100.0
                    if any_target_reached:
                        reward -= 200.0
                    if all_eliminated:
                        reward += 200.0
                        world_dones[w] = 1
                    if any_target_reached:
                        world_dones[w] = 1

                    # Dense reward: approach nearest active attacker
                    min_dist = float('inf')
                    # Track most-threatening attacker (closest to target)
                    min_att_to_target = float('inf')
                    dist_to_priority = float('inf')
                    for a in active_attackers:
                        dist = np.linalg.norm(def_positions[d] - att_positions[a])
                        if dist < min_dist:
                            min_dist = dist
                        att_to_target = np.linalg.norm(att_positions[a] - target)
                        if att_to_target < min_att_to_target:
                            min_att_to_target = att_to_target
                            dist_to_priority = dist

                    if min_dist < float('inf'):
                        prev = self.prev_dist_buf[w, d]
                        if prev > 0:
                            reward += (prev - min_dist) * 50.0
                        self.prev_dist_buf[w, d] = min_dist
                        reward += 1.0 / (min_dist + 0.1) - 1.0 / (4.0 + 0.1)

                    # Priority targeting bonus: extra reward for approaching the
                    # attacker closest to target (most dangerous)
                    if dist_to_priority < float('inf') and dist_to_priority < 2.0:
                        reward += (2.0 - dist_to_priority) * 0.5

                    # Coordination: reward spread between defenders
                    if self.N > 1:
                        for d2 in range(self.N):
                            if d2 != d:
                                dd = np.linalg.norm(def_positions[d] - def_positions[d2])
                                reward += self.cfg.coordination_bonus * min(dd, 2.0)

                    # Target assignment diversity: penalize when this defender
                    # shares a target with another defender while attackers go unguarded
                    if self.N > 1 and n_active >= 2:
                        my_target = nearest_attacker[d]
                        if my_target >= 0 and assignment_counts.get(my_target, 0) > 1:
                            # Multiple defenders on same attacker — small penalty
                            reward -= 0.3

                    reward -= 0.05

                    # Obstacle penalties
                    if obs_pos_np is not None:
                        for o in range(self._num_obstacles):
                            opos = obs_pos_np[w][o]
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
        """Build per-agent observations with distance-sorted opponents/teammates."""
        wp.synchronize()
        body_q = self.state_0.body_q.numpy()
        body_qd = self.state_0.body_qd.numpy()

        pos_scale = 0.25
        vel_scale = 0.2
        rel_scale = 1.0 / 6.0

        t = self.target_pos_wp
        target = np.array([float(t[0]), float(t[1]), float(t[2])], dtype=np.float32)

        # Hoist obstacle positions out of inner loops
        obs_pos_np = self.obstacle_pos_buf.numpy() if self._num_obstacles > 0 else None

        for w in range(self.num_worlds):
            positions = []
            velocities = []
            for i in range(self._num_drones):
                bidx = w * self.bodies_per_world + i
                positions.append(body_q[bidx][:3].astype(np.float32))
                velocities.append(body_qd[bidx][:3].astype(np.float32))

            if self.mode == "evade":
                # Attacker-centric observations
                # Obs layout reuses same 57-dim format but from attacker perspective:
                # [0:3] self pos, [3:6] self vel, [6:9] accel, [9] closing rate to nearest defender
                # [10:13] target pos
                # [13:34] per-defender block (3 max, sorted by distance) — "opponents" are defenders
                # [34:48] per-teammate block (other attackers, 2 max)
                # [48:57] obstacles
                for a in range(self.M):
                    agent_idx = w * self.M + a
                    obs = np.zeros(OBS_DIM, dtype=np.float32)

                    my_pos = positions[self.N + a]
                    my_vel = velocities[self.N + a]

                    obs[0:3] = my_pos * pos_scale
                    obs[3:6] = my_vel * vel_scale
                    acc = my_vel - self.prev_vel[w, a]
                    obs[6:9] = acc * vel_scale
                    self.prev_vel[w, a] = my_vel.copy()

                    # Closing rate to nearest defender [9]
                    min_def_dist = float('inf')
                    for d in range(self.N):
                        def_pos = positions[d]
                        dist = np.linalg.norm(my_pos - def_pos)
                        if dist < min_def_dist:
                            min_def_dist = dist
                            def_vel = velocities[d]
                            rel = def_pos - my_pos
                            if dist > 0.01:
                                closing = np.dot(my_vel - def_vel, rel / dist)
                                obs[9] = closing * vel_scale

                    obs[10:13] = target * pos_scale

                    # Per-defender block [13:34] — "opponents" sorted by distance
                    def_dists = []
                    for d in range(self.N):
                        dist = np.linalg.norm(my_pos - positions[d])
                        def_dists.append((dist, d))
                    def_dists.sort()

                    for k in range(min(self.N, MAX_ATTACKERS)):
                        dist, d = def_dists[k]
                        base = 13 + k * 7
                        obs[base:base+3] = positions[d] * pos_scale
                        obs[base+3:base+6] = velocities[d] * vel_scale
                        obs[base+6] = dist * rel_scale

                    # Per-teammate block [34:48] — other attackers
                    tm_dists = []
                    for a2 in range(self.M):
                        if a2 == a:
                            continue
                        if self.eliminated_attackers[w, a2]:
                            continue
                        dist = np.linalg.norm(my_pos - positions[self.N + a2])
                        tm_dists.append((dist, a2))
                    tm_dists.sort()

                    for k in range(min(len(tm_dists), MAX_DEFENDERS - 1)):
                        dist, a2 = tm_dists[k]
                        base = 34 + k * 7
                        obs[base:base+3] = positions[self.N + a2] * pos_scale
                        obs[base+3:base+6] = velocities[self.N + a2] * vel_scale
                        obs[base+6] = dist * rel_scale

                    if obs_pos_np is not None:
                        for o in range(self._num_obstacles):
                            base = 48 + o * 3
                            obs[base:base+3] = (obs_pos_np[w][o] - my_pos) * rel_scale

                    if self.obs_noise_std > 0:
                        obs += np.random.normal(0, self.obs_noise_std, OBS_DIM).astype(np.float32)

                    self.obs_buf[agent_idx] = obs
            else:
                # Defender-centric observations (original)
                for d in range(self.N):
                    agent_idx = w * self.N + d
                    obs = np.zeros(OBS_DIM, dtype=np.float32)

                    my_pos = positions[d]
                    my_vel = velocities[d]

                    obs[0:3] = my_pos * pos_scale
                    obs[3:6] = my_vel * vel_scale
                    acc = my_vel - self.prev_vel[w, d]
                    obs[6:9] = acc * vel_scale
                    self.prev_vel[w, d] = my_vel.copy()

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

                    obs[10:13] = target * pos_scale

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
                            obs[base:base+3] = positions[self.N + a] * pos_scale
                            obs[base+3:base+6] = velocities[self.N + a] * vel_scale
                            obs[base+6] = dist * rel_scale

                    tm_dists = []
                    for d2 in range(self.N):
                        if d2 == d:
                            continue
                        dist = np.linalg.norm(my_pos - positions[d2])
                        tm_dists.append((dist, d2))
                    tm_dists.sort()

                    for k in range(min(self.N - 1, MAX_DEFENDERS - 1)):
                        dist, d2 = tm_dists[k]
                        base = 34 + k * 7
                        obs[base:base+3] = positions[d2] * pos_scale
                        obs[base+3:base+6] = velocities[d2] * vel_scale
                        obs[base+6] = dist * rel_scale

                    if obs_pos_np is not None:
                        for o in range(self._num_obstacles):
                            base = 48 + o * 3
                            obs[base:base+3] = (obs_pos_np[w][o] - my_pos) * rel_scale

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
