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
    normalize_obs: bool = True
    normalize_reward: bool = True

    # Seed
    seed: int = 42
