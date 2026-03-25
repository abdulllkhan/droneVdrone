"""
GPU-Accelerated Drone Interception Environment (Newton Physics)

Runs N parallel drone interception scenarios entirely on GPU using
NVIDIA's Newton physics engine (Warp backend). Each world contains
a defender drone (RL agent) and an attacker drone (scripted).

Compatible with Stable Baselines3 via VecEnv interface.
"""

import numpy as np
import warp as wp
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from train.config import EnvConfig


# ------------------------------------------------------------------ #
# Warp Kernels (run on GPU)
# ------------------------------------------------------------------ #

@wp.struct
class Propeller:
    body: int
    pos: wp.vec3
    dir: wp.vec3
    max_thrust: float
    max_torque: float
    turning_direction: float


@wp.kernel
def compute_prop_wrenches(
    props: wp.array(dtype=Propeller),
    controls: wp.array(dtype=float),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_f: wp.array(dtype=wp.spatial_vector),
):
    """Apply propeller thrust and torque to drone bodies."""
    tid = wp.tid()
    prop = props[tid]
    control = controls[tid]
    tf = body_q[prop.body]
    dir = wp.transform_vector(tf, prop.dir)
    force = dir * prop.max_thrust * control
    torque = dir * prop.max_torque * control * prop.turning_direction
    moment_arm = wp.transform_point(tf, prop.pos) - wp.transform_point(tf, body_com[prop.body])
    torque += wp.cross(moment_arm, force)
    torque *= 0.8  # angular damping
    wp.atomic_add(body_f, prop.body, wp.spatial_vector(force, torque))


@wp.kernel
def velocity_pid_to_throttle(
    body_q: wp.array(dtype=wp.transform),
    body_qd: wp.array(dtype=wp.spatial_vector),
    target_vel: wp.array(dtype=wp.vec3),
    throttle_out: wp.array(dtype=float),
    bodies_per_world: int,
    body_offset: int,
    prop_offset: int,
    arm_length: float,
    mass: float,
    gravity: float,
    kp: float,
    max_thrust_per_prop: float,
):
    """
    Convert desired velocity to 4 propeller throttles via a simple PID.

    For each drone, computes a desired net force from velocity error,
    then maps it to individual propeller throttles normalized by max thrust.
    """
    world_id = wp.tid()
    body_idx = world_id * bodies_per_world + body_offset

    # Current state
    tf = body_q[body_idx]
    vel = wp.vec3(
        body_qd[body_idx][0],
        body_qd[body_idx][1],
        body_qd[body_idx][2],
    )
    desired = target_vel[world_id]

    # PD velocity control -> desired force (Newtons)
    vel_error = desired - vel
    fx = vel_error[0] * kp * mass
    fy = vel_error[1] * kp * mass
    fz = vel_error[2] * kp * mass + mass * gravity  # gravity compensation

    # Get drone's up vector in world frame
    up_local = wp.vec3(0.0, 0.0, 1.0)
    up_world = wp.transform_vector(tf, up_local)
    # Project desired force onto drone's up axis for collective thrust
    thrust_total = wp.dot(wp.vec3(fx, fy, fz), up_world)

    # Simple mixer: distribute thrust + differential for roll/pitch
    right_local = wp.vec3(1.0, 0.0, 0.0)
    fwd_local = wp.vec3(0.0, 1.0, 0.0)
    right_world = wp.transform_vector(tf, right_local)
    fwd_world = wp.transform_vector(tf, fwd_local)

    # Desired torques for roll/pitch
    roll_torque = wp.dot(wp.vec3(fx, fy, fz), fwd_world) * arm_length
    pitch_torque = -wp.dot(wp.vec3(fx, fy, fz), right_world) * arm_length

    # Normalize by max thrust to get throttle [0, 1]
    base = thrust_total / (4.0 * max_thrust_per_prop)
    roll_diff = roll_torque / (4.0 * max_thrust_per_prop)
    pitch_diff = pitch_torque / (4.0 * max_thrust_per_prop)

    # Prop 0: +Y, Prop 1: -Y, Prop 2: +X, Prop 3: -X
    t0 = base + roll_diff
    t1 = base - roll_diff
    t2 = base + pitch_diff
    t3 = base - pitch_diff

    prop_base = world_id * 8 + prop_offset  # 8 props per world (2 drones x 4)
    throttle_out[prop_base + 0] = wp.clamp(t0, 0.0, 1.0)
    throttle_out[prop_base + 1] = wp.clamp(t1, 0.0, 1.0)
    throttle_out[prop_base + 2] = wp.clamp(t2, 0.0, 1.0)
    throttle_out[prop_base + 3] = wp.clamp(t3, 0.0, 1.0)


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
    """Compute attacker desired velocity: fly straight toward target."""
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
    """Compute 19-dim observation for each world."""
    world_id = wp.tid()
    def_idx = world_id * bodies_per_world + defender_offset
    att_idx = world_id * bodies_per_world + attacker_offset

    def_pos = wp.transform_get_translation(body_q[def_idx])
    att_pos = wp.transform_get_translation(body_q[att_idx])
    def_vel = wp.vec3(body_qd[def_idx][0], body_qd[def_idx][1], body_qd[def_idx][2])
    att_vel = wp.vec3(body_qd[att_idx][0], body_qd[att_idx][1], body_qd[att_idx][2])
    rel_pos = att_pos - def_pos
    dist = wp.length(rel_pos)

    obs_out[world_id, 0] = def_pos[0]
    obs_out[world_id, 1] = def_pos[1]
    obs_out[world_id, 2] = def_pos[2]
    obs_out[world_id, 3] = def_vel[0]
    obs_out[world_id, 4] = def_vel[1]
    obs_out[world_id, 5] = def_vel[2]
    obs_out[world_id, 6] = att_pos[0]
    obs_out[world_id, 7] = att_pos[1]
    obs_out[world_id, 8] = att_pos[2]
    obs_out[world_id, 9] = att_vel[0]
    obs_out[world_id, 10] = att_vel[1]
    obs_out[world_id, 11] = att_vel[2]
    obs_out[world_id, 12] = rel_pos[0]
    obs_out[world_id, 13] = rel_pos[1]
    obs_out[world_id, 14] = rel_pos[2]
    obs_out[world_id, 15] = dist
    obs_out[world_id, 16] = target_pos[0]
    obs_out[world_id, 17] = target_pos[1]
    obs_out[world_id, 18] = target_pos[2]


@wp.kernel
def compute_rewards_kernel(
    body_q: wp.array(dtype=wp.transform),
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
    """Compute reward and done flag for each world."""
    world_id = wp.tid()
    def_idx = world_id * bodies_per_world + defender_offset
    att_idx = world_id * bodies_per_world + attacker_offset

    def_pos = wp.transform_get_translation(body_q[def_idx])
    att_pos = wp.transform_get_translation(body_q[att_idx])
    dist = wp.length(att_pos - def_pos)
    att_to_target = wp.length(att_pos - target_pos)

    reward = 0.0
    done = 0

    # Interception bonus
    if dist < intercept_dist:
        reward = 100.0
        done = 1
        info_intercepted[world_id] = 1
    elif att_to_target < target_reached_dist:
        # Failure penalty
        reward = -100.0
        done = 1
        info_target_reached[world_id] = 1
    else:
        # Dense approach reward
        pd = prev_dist[world_id]
        if pd > 0.0:
            reward += (pd - dist) * 10.0

        # Proximity bonus
        if dist < 1.0:
            reward += (1.0 - dist) * 2.0

        # Altitude match penalty
        alt_diff = wp.abs(def_pos[2] - att_pos[2])
        reward -= alt_diff * 0.5

        # Time penalty
        reward -= 0.01

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
    dones: wp.array(dtype=int),
):
    """Check truncation conditions (bounds + timeout)."""
    world_id = wp.tid()

    if step_count[world_id] >= max_steps:
        dones[world_id] = 1

    for i in range(2):
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
):
    """Reset bodies in worlds that are done."""
    world_id = wp.tid()
    if reset_mask[world_id] == 0:
        return

    for i in range(2):
        body_idx = world_id * bodies_per_world + i
        body_q[body_idx] = wp.transform(init_positions[world_id, i], wp.quat_identity())
        body_qd[body_idx] = wp.spatial_vector(
            wp.vec3(0.0, 0.0, 0.0), wp.vec3(0.0, 0.0, 0.0)
        )


# ------------------------------------------------------------------ #
# Helper: build a single drone (2 crossbar boxes + 4 propellers)
# ------------------------------------------------------------------ #

def _build_drone(builder, arm_length, collision_group, label="drone"):
    """Add a drone body with crossbar shapes to the builder. Returns body index."""
    import newton as nt

    body = builder.add_body(label=label)
    bar_h = arm_length * 0.05
    bar_w = arm_length * 0.05
    density = 1750.0  # carbon fiber

    cfg = nt.ModelBuilder.ShapeConfig(
        density=density, collision_group=collision_group
    )
    builder.add_shape_box(body, hx=bar_w, hy=arm_length, hz=bar_h, cfg=cfg)
    builder.add_shape_box(body, hx=arm_length, hy=bar_w, hz=bar_h, cfg=cfg)

    return body


def _make_propellers(body, arm_length, fps):
    """Create 4 propellers for a drone body."""
    positions = [
        wp.vec3(0.0, arm_length, 0.0),
        wp.vec3(0.0, -arm_length, 0.0),
        wp.vec3(arm_length, 0.0, 0.0),
        wp.vec3(-arm_length, 0.0, 0.0),
    ]
    directions = [-1.0, 1.0, 1.0, -1.0]

    props = []
    for pos, td in zip(positions, directions):
        air_density = 1.225
        thrust_coeff = 0.109919
        power_coeff = 0.040164
        diameter = 0.2286
        max_rpm = 6396.667
        rps = max_rpm / fps

        prop = Propeller()
        prop.body = body
        prop.pos = pos
        prop.dir = wp.vec3(0.0, 0.0, 1.0)
        prop.max_thrust = thrust_coeff * air_density * rps**2 * diameter**4
        prop.max_torque = power_coeff * air_density * rps**2 * diameter**5 / wp.TAU
        prop.turning_direction = td
        props.append(prop)

    return props


# ------------------------------------------------------------------ #
# Main VecEnv class
# ------------------------------------------------------------------ #

class NewtonInterceptVecEnv(VecEnv):
    """
    GPU-accelerated vectorized drone interception environment.

    Runs `num_envs` parallel worlds on GPU using Newton physics.
    Compatible with Stable Baselines3 VecEnv interface.
    """

    OBS_DIM = 19
    ACT_DIM = 3

    def __init__(self, num_envs=256, config=None, device="cuda:0"):
        import newton as nt
        import gymnasium as gym

        self.cfg = config or EnvConfig()
        self.device = device
        self.n_envs = num_envs

        # Physics params
        self.fps = self.cfg.ctrl_freq
        self.dt = 1.0 / self.fps
        self.substeps = max(1, self.cfg.pyb_freq // self.cfg.ctrl_freq)
        self.sub_dt = self.dt / self.substeps
        self.arm_length = 0.2  # CrazyFlie arm length (meters)
        self.max_steps = int(self.cfg.episode_secs * self.cfg.ctrl_freq)
        self.target_pos_wp = wp.vec3(*self.cfg.target_pos)

        # Drone mass — will be read from Newton model after finalize()
        self.drone_mass = 0.027  # placeholder, updated below
        # Gravity magnitude
        self.gravity = 9.81

        # Speed limits
        self.defender_speed_limit = 0.15 * 30.0 * (1000.0 / 3600.0)  # ~1.25 m/s
        self.attacker_speed = self.cfg.attacker_speed_scale * 0.8  # m/s

        # PID gain for velocity controller (scales with mass)
        self.kp = 5.0

        # Build scene: N worlds, each with 2 drones
        builder = nt.ModelBuilder(gravity=-self.gravity)

        all_props = []
        for env_i in range(num_envs):
            builder.begin_world(label=f"world_{env_i}")

            # Defender (body 0 in this world)
            def_body = _build_drone(
                builder, self.arm_length,
                collision_group=env_i * 2, label=f"defender_{env_i}"
            )
            def_props = _make_propellers(def_body, self.arm_length, self.fps)

            # Attacker (body 1 in this world)
            att_body = _build_drone(
                builder, self.arm_length,
                collision_group=env_i * 2 + 1, label=f"attacker_{env_i}"
            )
            att_props = _make_propellers(att_body, self.arm_length, self.fps)

            all_props.extend(def_props)
            all_props.extend(att_props)

            builder.end_world()

        self.model = builder.finalize()
        self.bodies_per_world = 2
        self.props_per_world = 8  # 4 per drone x 2 drones

        # Read actual drone mass from Newton model
        body_masses = self.model.body_mass.numpy()
        self.drone_mass = float(body_masses[0])  # first body = defender

        # Max thrust per propeller (for throttle normalization)
        self.max_thrust_per_prop = float(all_props[0].max_thrust)

        # States (double buffered)
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        # Propellers
        self.props = wp.array(all_props, dtype=Propeller, device=device)
        self.prop_controls = wp.zeros(
            len(all_props), dtype=float, device=device
        )

        # Solver
        self.solver = nt.solvers.SolverSemiImplicit(self.model)

        # RL state arrays (GPU)
        self.obs_buf = wp.zeros((num_envs, self.OBS_DIM), dtype=float, device=device)
        self.rewards_buf = wp.zeros(num_envs, dtype=float, device=device)
        self.dones_buf = wp.zeros(num_envs, dtype=int, device=device)
        self.term_buf = wp.zeros(num_envs, dtype=int, device=device)
        self.prev_dist_buf = wp.zeros(num_envs, dtype=float, device=device)
        self.step_count_buf = wp.zeros(num_envs, dtype=int, device=device)
        self.info_intercepted = wp.zeros(num_envs, dtype=int, device=device)
        self.info_target_reached = wp.zeros(num_envs, dtype=int, device=device)

        # Velocity command buffers
        self.defender_vel = wp.zeros(num_envs, dtype=wp.vec3, device=device)
        self.attacker_vel = wp.zeros(num_envs, dtype=wp.vec3, device=device)

        # Initial positions buffer (for resets)
        self.init_pos_buf = wp.zeros(
            (num_envs, 2), dtype=wp.vec3, device=device
        )

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
        self._randomize_all_positions()
        self._apply_init_positions(mask=None)

    # ------------------------------------------------------------------ #
    # VecEnv interface
    # ------------------------------------------------------------------ #

    def reset(self):
        """Reset all environments."""
        self._randomize_all_positions()
        self._apply_init_positions(mask=None)
        self.prev_dist_buf.zero_()
        self.step_count_buf.zero_()
        self._compute_obs()
        return self.obs_buf.numpy().astype(np.float32)

    def step_async(self, actions):
        """Submit actions (non-blocking)."""
        self._actions = np.asarray(actions, dtype=np.float32)

    def step_wait(self):
        """Execute physics step and return results."""
        actions = self._actions  # (num_envs, 3)

        # Scale actions to velocity commands
        vel_np = actions * self.defender_speed_limit
        self.defender_vel = wp.array(
            [wp.vec3(v[0], v[1], v[2]) for v in vel_np],
            dtype=wp.vec3, device=self.device
        )

        # Compute attacker pursuit velocities
        wp.launch(
            attacker_pursuit,
            dim=self.n_envs,
            inputs=(
                self.state_0.body_q,
                self.state_0.body_qd,
                self.target_pos_wp,
                self.attacker_vel,
                self.bodies_per_world,
                1,  # attacker body offset
                self.attacker_speed,
            ),
            device=self.device,
        )

        # Convert velocities to propeller throttles
        # Defender (body offset 0, prop offset 0)
        wp.launch(
            velocity_pid_to_throttle,
            dim=self.n_envs,
            inputs=(
                self.state_0.body_q,
                self.state_0.body_qd,
                self.defender_vel,
                self.prop_controls,
                self.bodies_per_world,
                0,  # defender body offset
                0,  # defender prop offset (first 4 props)
                self.arm_length,
                self.drone_mass,
                self.gravity,
                self.kp,
                self.max_thrust_per_prop,
            ),
            device=self.device,
        )

        # Attacker (body offset 1, prop offset 4)
        wp.launch(
            velocity_pid_to_throttle,
            dim=self.n_envs,
            inputs=(
                self.state_0.body_q,
                self.state_0.body_qd,
                self.attacker_vel,
                self.prop_controls,
                self.bodies_per_world,
                1,  # attacker body offset
                4,  # attacker prop offset (second 4 props)
                self.arm_length,
                self.drone_mass,
                self.gravity,
                self.kp,
                self.max_thrust_per_prop,
            ),
            device=self.device,
        )

        # Physics substeps
        for _ in range(self.substeps):
            self.state_0.clear_forces()

            # Apply propeller forces
            wp.launch(
                compute_prop_wrenches,
                dim=len(self.props),
                inputs=(
                    self.props,
                    self.prop_controls,
                    self.state_0.body_q,
                    self.model.body_com,
                ),
                outputs=(self.state_0.body_f,),
                device=self.device,
            )

            self.solver.step(self.state_0, self.state_1, None, None, self.sub_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        # Increment step counters
        step_np = self.step_count_buf.numpy()
        step_np += 1
        self.step_count_buf = wp.array(step_np, dtype=int, device=self.device)

        # Compute rewards and termination
        self.info_intercepted.zero_()
        self.info_target_reached.zero_()
        self.term_buf.zero_()
        self.dones_buf.zero_()

        wp.launch(
            compute_rewards_kernel,
            dim=self.n_envs,
            inputs=(
                self.state_0.body_q,
                self.prev_dist_buf,
                self.target_pos_wp,
                self.cfg.intercept_dist,
                self.cfg.target_reached_dist,
                self.bodies_per_world,
                0, 1,
                self.rewards_buf,
                self.term_buf,
                self.info_intercepted,
                self.info_target_reached,
            ),
            device=self.device,
        )

        # Check bounds/timeout -> truncation
        wp.launch(
            check_bounds_kernel,
            dim=self.n_envs,
            inputs=(
                self.state_0.body_q,
                self.step_count_buf,
                self.max_steps,
                self.cfg.arena_bounds,
                self.cfg.altitude_min,
                self.cfg.altitude_max,
                self.bodies_per_world,
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
        intercepted = self.info_intercepted.numpy()
        target_reached = self.info_target_reached.numpy()

        # Build info dicts
        infos = []
        for i in range(self.n_envs):
            infos.append({
                "intercepted": bool(intercepted[i]),
                "target_reached": bool(target_reached[i]),
            })

        # Auto-reset done environments
        if np.any(dones):
            done_mask = dones.astype(np.int32)
            self._randomize_positions_masked(done_mask)
            reset_mask_wp = wp.array(done_mask, dtype=int, device=self.device)
            wp.launch(
                reset_bodies_kernel,
                dim=self.n_envs,
                inputs=(
                    self.state_0.body_q,
                    self.state_0.body_qd,
                    self.init_pos_buf,
                    reset_mask_wp,
                    self.bodies_per_world,
                ),
                device=self.device,
            )
            # Reset step counters and prev_dist for done envs
            step_np = self.step_count_buf.numpy()
            prev_np = self.prev_dist_buf.numpy()
            step_np[dones] = 0
            prev_np[dones] = 0.0
            self.step_count_buf = wp.array(step_np, dtype=int, device=self.device)
            self.prev_dist_buf = wp.array(prev_np, dtype=float, device=self.device)

        # Compute observations (after resets)
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
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _compute_obs(self):
        wp.launch(
            compute_obs_kernel,
            dim=self.n_envs,
            inputs=(
                self.state_0.body_q,
                self.state_0.body_qd,
                self.target_pos_wp,
                self.obs_buf,
                self.bodies_per_world,
                0, 1,
            ),
            device=self.device,
        )

    def _randomize_all_positions(self):
        """Generate random initial positions for all environments."""
        rng = np.random.default_rng()
        positions = np.zeros((self.n_envs, 2, 3), dtype=np.float32)
        target = np.array(self.cfg.target_pos)

        for i in range(self.n_envs):
            # Defender: near target
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(*self.cfg.defender_dist_range)
            positions[i, 0] = [
                target[0] + dist * np.cos(angle),
                target[1] + dist * np.sin(angle),
                rng.uniform(*self.cfg.defender_alt_range),
            ]
            # Attacker: far from target
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(*self.cfg.attacker_dist_range)
            positions[i, 1] = [
                target[0] + dist * np.cos(angle),
                target[1] + dist * np.sin(angle),
                rng.uniform(*self.cfg.attacker_alt_range),
            ]

        wp_pos = [
            [wp.vec3(float(positions[i, j, 0]),
                     float(positions[i, j, 1]),
                     float(positions[i, j, 2]))
             for j in range(2)]
            for i in range(self.n_envs)
        ]
        self.init_pos_buf = wp.array(wp_pos, dtype=wp.vec3, device=self.device)

    def _randomize_positions_masked(self, mask):
        """Randomize positions only for done environments."""
        rng = np.random.default_rng()
        target = np.array(self.cfg.target_pos)
        current = self.init_pos_buf.numpy()

        for i in range(self.n_envs):
            if mask[i]:
                angle = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(*self.cfg.defender_dist_range)
                current[i, 0] = [
                    target[0] + dist * np.cos(angle),
                    target[1] + dist * np.sin(angle),
                    rng.uniform(*self.cfg.defender_alt_range),
                ]
                angle = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(*self.cfg.attacker_dist_range)
                current[i, 1] = [
                    target[0] + dist * np.cos(angle),
                    target[1] + dist * np.sin(angle),
                    rng.uniform(*self.cfg.attacker_alt_range),
                ]

        wp_pos = [
            [wp.vec3(float(current[i, j, 0]),
                     float(current[i, j, 1]),
                     float(current[i, j, 2]))
             for j in range(2)]
            for i in range(self.n_envs)
        ]
        self.init_pos_buf = wp.array(wp_pos, dtype=wp.vec3, device=self.device)

    def _apply_init_positions(self, mask=None):
        """Apply initial positions to physics state."""
        if mask is None:
            mask_wp = wp.ones(self.n_envs, dtype=int, device=self.device)
        else:
            mask_wp = wp.array(mask, dtype=int, device=self.device)

        wp.launch(
            reset_bodies_kernel,
            dim=self.n_envs,
            inputs=(
                self.state_0.body_q,
                self.state_0.body_qd,
                self.init_pos_buf,
                mask_wp,
                self.bodies_per_world,
            ),
            device=self.device,
        )
