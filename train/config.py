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
    altitude_max: float = 3.0     # Max altitude (meters)
    altitude_min: float = 0.05    # Min altitude (meters)

    # Interception
    intercept_dist: float = 0.50  # Interception threshold (meters)
    target_reached_dist: float = 0.3  # Attacker wins threshold (meters)

    # Attacker
    attacker_speed_scale: float = 0.6  # Scale PID output (1.0 = full speed)

    # Initialization ranges
    attacker_dist_range: tuple = (2.0, 3.0)   # Start distance from target
    attacker_alt_range: tuple = (0.5, 1.5)
    defender_dist_range: tuple = (0.3, 1.0)    # Start distance from target
    defender_alt_range: tuple = (0.7, 1.3)

    # Target position
    target_pos: tuple = (0.0, 0.0, 1.0)

    # Force model (direct body forces, no propeller simulation)
    # Drone mass ~0.027kg, hover force = 0.265N
    # max_force must be >> hover force to allow maneuvering while compensating gravity
    force_kp: float = 10.0         # Proportional gain for velocity tracking
    force_kd: float = 4.0          # Derivative gain (damping, prevents oscillation)
    max_force: float = 2.0         # Max force magnitude (Newtons, ~7.5x drone weight)
    angular_damping: float = 5.0   # Angular velocity damping to prevent spinning

    # Training mode (for progressive training)
    mode: str = "intercept"        # "fly_to_target", "chase", "intercept", "adversarial"
    attacker_model_path: str = ""  # Path to frozen attacker model (intercept mode)
    attacker_model_paths: tuple = ()  # Multiple attacker models for mixture training
    defender_model_path: str = ""  # Path to frozen defender model (evade mode)

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
    net_arch: list = field(default_factory=lambda: [256, 256])

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
