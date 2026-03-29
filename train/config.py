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
    obstacle_proximity_penalty: float = 2.0      # Reward penalty scale near obstacles
    obstacle_danger_radius: float = 0.5          # Proximity penalty kicks in at this distance

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
    checkpoint_freq: int = 100_000
    eval_freq: int = 5_000
    eval_episodes: int = 20
    tensorboard_log: str = "logs/tensorboard"

    # Environment
    n_envs: int = 16              # Parallel training environments
    normalize_obs: bool = False
    normalize_reward: bool = False

    # Seed
    seed: int = 42
