"""Hyperparameters and environment configuration."""

from dataclasses import dataclass, field


@dataclass
class EnvConfig:
    """Environment settings."""
    # Physics
    pyb_freq: int = 240           # PyBullet physics frequency (Hz)
    ctrl_freq: int = 48           # Control / RL step frequency (Hz)
    episode_secs: float = 10.0    # Max episode duration

    # Arena
    arena_bounds: float = 4.0     # XY bounds (meters) — half-width, total = 2x this
    altitude_max: float = 4.0     # Max altitude (meters)
    altitude_min: float = 0.3     # Min altitude (meters) — prevents ground crashes

    # Interception
    intercept_dist: float = 0.75  # Interception threshold (meters) — larger halo makes defense easier
    target_reached_dist: float = 0.3  # Attacker wins threshold (meters)

    # Attacker
    attacker_speed_scale: float = 1.0  # Scripted attacker speed (1.0 = full speed)
    defender_speed_limit: float = 1.25  # Defender max speed (m/s)
    attacker_speed_limit: float = 1.4  # RL attacker max speed (m/s) — slightly faster than defender (~1.25)
    attacker_strategy: str = "direct"  # Scripted attacker behavior: "direct", "flanking", "mixed"

    # Initialization ranges
    attacker_dist_range: tuple = (1.0, 2.0)   # Start distance from target (close = aggressive)
    attacker_alt_range: tuple = (1.5, 2.5)
    defender_dist_range: tuple = (0.3, 1.0)    # Start distance from target
    defender_alt_range: tuple = (1.5, 2.5)

    # Target position (z randomized during reset between target_alt_range)
    target_pos: tuple = (0.0, 0.0, 2.0)       # Default target pos (overridden if randomized)
    target_alt_randomize: bool = True          # Randomize target altitude each episode
    target_alt_range_intercept: tuple = (1.0, 3.0)  # Target altitude range for intercept/evade

    # Arena scale preset — "small" (8m lab) or "realistic" (300m field)
    arena_scale: str = "small"    # "small" or "realistic" — sets defaults if not overridden

    # Force model (direct body forces, no propeller simulation)
    # Drone mass ~0.250kg (250g quadrotor), hover force = mass * g = 2.45N
    # max_force must be >> hover force to allow maneuvering while compensating gravity
    drone_mass: float = 0.250      # Drone mass (kg) — 250g quadrotor
    force_kp: float = 10.0         # Proportional gain for velocity tracking
    force_kd: float = 4.0          # Derivative gain (damping, prevents oscillation)
    max_force: float = 8.0         # Max force magnitude (Newtons, ~3.3x drone weight)
    angular_damping: float = 5.0   # Angular velocity damping to prevent spinning

    # Training mode (for progressive training)
    mode: str = "intercept"        # "fly_to_target", "chase", "intercept", "adversarial"
    attacker_model_path: str = ""  # Path to frozen attacker model (intercept mode)
    attacker_model_paths: tuple = ()  # Multiple attacker models for mixture training
    defender_model_path: str = ""  # Path to frozen defender model (evade mode)

    # Obstacles
    num_obstacles: int = 3                       # Number of obstacles per world (0 = disabled)
    obstacle_radius: float = 0.15                # Obstacle cylinder radius (meters)
    obstacle_height: float = 2.0                 # Obstacle cylinder half-height (meters)
    obstacle_placement_range: float = 2.0        # Max distance from origin for obstacle placement
    obstacle_min_spacing: float = 0.8            # Min distance between obstacles
    obstacle_proximity_penalty: float = 10.0     # Reward penalty scale near obstacles
    obstacle_danger_radius: float = 0.5          # Proximity penalty kicks in at this distance
    obstacle_collision_penalty: float = 50.0     # Hard penalty for hitting obstacle (within radius)
    obstacle_collision_radius: float = 0.2       # Collision detection radius (slightly > physical 0.15)
    obstacle_dynamic: bool = False               # Enable moving obstacles
    obstacle_max_speed: float = 0.3              # Max obstacle drift speed (m/s)
    obstacle_direction_change_freq: int = 96     # Obstacle changes direction every N steps

    # Multi-drone swarm
    num_defenders: int = 1        # N defenders per world
    num_attackers: int = 1        # M attackers per world
    max_defenders: int = 3        # Max for obs padding (fixed obs dim) — DEPRECATED, use k_opponents/k_teammates
    max_attackers: int = 3        # Max for obs padding (fixed obs dim) — DEPRECATED, use k_opponents/k_teammates
    k_opponents: int = 3          # K-nearest opponents to observe (fixed obs dim, independent of swarm size)
    k_teammates: int = 2          # K-nearest teammates to observe (fixed obs dim, independent of swarm size)
    kamikaze: bool = True         # Defender is destroyed on interception (both drones eliminated)
    coordination_bonus: float = 0.5  # Reward for defender spread
    defender_leash_radius: float = 3.0   # Max distance from target before penalty kicks in
    defender_leash_penalty: float = 2.0  # Penalty scale for exceeding leash radius
    defender_zone_time_penalty: float = 0.1  # Per-step escalating penalty when outside leash radius
    angular_coverage_bonus: float = 0.3  # Reward for defenders spreading angles around target
    intercept_efficiency_bonus: float = 1.0  # Bonus when closest defender to attacker intercepts it

    # Domain randomization
    wind_enabled: bool = False        # Enable random wind forces
    wind_max_force: float = 0.3       # Max wind force (Newtons)
    wind_change_freq: int = 48        # Wind direction changes every N steps
    wind_gust_prob: float = 0.02      # Probability of sudden wind gust per step
    wind_gust_force: float = 1.0      # Max gust force (Newtons) — much stronger than steady wind
    obs_noise_std: float = 0.0        # Gaussian noise on observations (0 = disabled)
    action_delay_steps: int = 0       # Action delay in steps (0 = disabled)
    sensor_range: float = 0.0         # Max detection range (0 = unlimited). Beyond this, obs zeroed out.
    drone_mass_variance: float = 0.0  # Mass randomization ±fraction (0.1 = ±10% of drone_mass)
    attacker_speed_variance: float = 0.0  # Per-attacker speed randomization ±fraction

    # Bootstrap shaping (Stage 1 warm-up — set to 0 for pure sparse stages)
    approach_shaping_coeff: float = 0.0   # Per-meter closing toward nearest attacker (0=off; 0.1=stage1 bootstrap)

    # Phase-specific settings
    target_randomize_range: float = 2.0   # Random target radius (fly_to_target)
    target_alt_range: tuple = (0.5, 2.0)  # Random target altitude range
    arrival_dist: float = 0.5             # Arrival threshold (fly_to_target/chase) — matches intercept_dist
    arrival_bonus: float = 500.0          # Large reward for reaching target (must dominate dense rewards)
    moving_target_speed: float = 0.5      # Target speed (chase mode, m/s)

    def apply_realistic_scale(self):
        """Apply 300m x 300m field scale — call this to override defaults for realistic arena.

        Real drone speeds: 15-20 m/s for racing drones, 5-10 m/s for commercial.
        We use 15 m/s defender, 18 m/s attacker for fast-paced realistic combat.
        """
        self.arena_bounds = 150.0        # ±150m = 300m x 300m field
        self.altitude_max = 120.0        # 120m max altitude (FAA limit)
        self.altitude_min = 2.0          # 2m min altitude
        self.intercept_dist = 2.0        # 2m intercept radius (realistic drone collision)
        self.target_reached_dist = 1.0   # 1m target threshold
        self.defender_speed_limit = 15.0  # 15 m/s (~54 km/h)
        self.attacker_speed_limit = 18.0 # 18 m/s (~65 km/h)
        self.attacker_speed_scale = 1.0
        self.episode_secs = 30.0         # 30s episodes (covers 300m at 15 m/s)
        self.attacker_dist_range = (50.0, 120.0)   # Attackers spawn 50-120m from target
        self.attacker_alt_range = (10.0, 60.0)
        self.defender_dist_range = (5.0, 30.0)      # Defenders close to target
        self.defender_alt_range = (10.0, 40.0)
        self.target_pos = (0.0, 0.0, 20.0)
        self.target_alt_range_intercept = (10.0, 50.0)
        self.target_alt_range = (5.0, 30.0)
        self.target_randomize_range = 50.0
        # Force model scales with speed
        self.force_kp = 30.0             # Higher gain for faster drones
        self.force_kd = 10.0
        self.max_force = 80.0            # Much stronger forces for high-speed maneuvering
        # Larger leash for bigger arena
        self.defender_leash_radius = 200.0
        # Obstacles scale
        self.obstacle_radius = 2.0
        self.obstacle_height = 30.0
        self.obstacle_placement_range = 60.0
        self.obstacle_min_spacing = 15.0
        self.obstacle_danger_radius = 5.0
        self.obstacle_collision_radius = 3.0
        # Phase-specific scales
        self.arrival_dist = 2.0
        self.arrival_bonus = 500.0
        self.moving_target_speed = 5.0
        return self

    def apply_ukraine_scale(self):
        """Ukraine conflict scenario: Shahed-type attackers vs FPV interceptor defenders.

        Attacker (Shahed-136 analog): ~50 m/s, low-altitude terrain-hugging loitering munition.
        Defender (FPV interceptor analog): ~60 m/s, highly maneuverable kamikaze interceptor.

        Arena: 1km x 1km open field with hill terrain.
        Detection: defenders have radar-like detection range (300m); globally shared once detected.
        """
        # Arena
        self.arena_bounds = 500.0           # ±500m = 1km × 1km field
        self.altitude_max = 200.0           # Interceptors can climb high to intercept
        self.altitude_min = 5.0             # Min altitude (terrain clearance)
        # Intercept / target thresholds
        self.intercept_dist = 3.0           # 3m collision radius (larger drones)
        self.target_reached_dist = 5.0      # 5m target protection bubble
        # Drone speeds
        self.defender_speed_limit = 60.0    # FPV interceptor: ~216 km/h
        self.attacker_speed_limit = 50.0    # Shahed: ~180 km/h
        self.attacker_speed_scale = 1.0
        # Episode length
        self.episode_secs = 60.0            # 60s — enough to cover 1km at 50 m/s
        # Spawn ranges (relative to target at origin)
        self.attacker_dist_range = (200.0, 450.0)   # Shahed spawns far out
        self.attacker_alt_range = (30.0, 90.0)       # Low-altitude terrain-hugging
        self.defender_dist_range = (20.0, 120.0)     # Interceptors near target
        self.defender_alt_range = (30.0, 150.0)
        # Target
        self.target_pos = (0.0, 0.0, 30.0)
        self.target_alt_range_intercept = (20.0, 60.0)
        self.target_alt_range = (10.0, 50.0)
        self.target_randomize_range = 100.0
        # Force model — powerful for high-speed maneuvering
        self.drone_mass = 0.200
        self.force_kp = 50.0
        self.force_kd = 15.0
        self.max_force = 150.0
        # Hills terrain (wide, tall cylindrical hills)
        self.num_obstacles = 5
        self.obstacle_radius = 50.0          # Wide hill base (50m radius)
        self.obstacle_height = 60.0          # Tall hill (60m half-height → 120m above ground)
        self.obstacle_placement_range = 350.0
        self.obstacle_min_spacing = 120.0    # Hills well-spaced
        self.obstacle_danger_radius = 70.0   # Start penalizing at 70m from hill center
        self.obstacle_collision_radius = 55.0
        self.obstacle_proximity_penalty = 5.0
        self.obstacle_collision_penalty = 100.0
        # Detection range — shared radar (any defender detecting = all defenders know)
        self.sensor_range = 300.0
        # Wind (light — eastern European flatlands)
        self.wind_enabled = True
        self.wind_max_force = 0.5
        self.wind_gust_prob = 0.01
        self.wind_gust_force = 1.5
        # Leash
        self.defender_leash_radius = 800.0
        # Phase-specific
        self.arrival_dist = 3.0
        self.arrival_bonus = 500.0
        self.moving_target_speed = 10.0
        return self


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    # Algorithm
    algorithm: str = "SAC"
    total_timesteps: int = 1_000_000
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 100_000
    learning_starts: int = 10_000
    gamma: float = 0.99
    tau: float = 0.005
    ent_coef: str = "0.1"          # Fixed entropy to prevent collapse (was "auto" → collapsed to 0.003)

    # Policy network
    policy: str = "MlpPolicy"
    net_arch: list = field(default_factory=lambda: [512, 512, 256])

    # Logging & checkpoints
    log_dir: str = "logs"
    model_dir: str = "models"
    checkpoint_freq: int = 10_000
    eval_freq: int = 500          # Per step() call — with 256 sub-envs this evals every ~128K timesteps
    eval_episodes: int = 20
    tensorboard_log: str = "logs/tensorboard"

    # Environment
    n_envs: int = 16              # Parallel training environments
    normalize_obs: bool = False
    normalize_reward: bool = False

    # Seed
    seed: int = 42
