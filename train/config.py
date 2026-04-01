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
    arena_bounds: float = 4.0     # XY bounds (meters)
    altitude_max: float = 4.0     # Max altitude (meters)
    altitude_min: float = 0.3     # Min altitude (meters) — prevents ground crashes

    # Interception
    intercept_dist: float = 0.75  # Interception threshold (meters) — larger halo makes defense easier
    target_reached_dist: float = 0.3  # Attacker wins threshold (meters)

    # Attacker
    attacker_speed_scale: float = 1.0  # Scripted attacker speed (1.0 = full speed)
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

    # Phase-specific settings
    target_randomize_range: float = 2.0   # Random target radius (fly_to_target)
    target_alt_range: tuple = (0.5, 2.0)  # Random target altitude range
    arrival_dist: float = 0.5             # Arrival threshold (fly_to_target/chase) — matches intercept_dist
    arrival_bonus: float = 500.0          # Large reward for reaching target (must dominate dense rewards)
    moving_target_speed: float = 0.5      # Target speed (chase mode, m/s)


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
