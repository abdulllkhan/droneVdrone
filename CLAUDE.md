# Drone vs Drone — RL Interception Defense System

## What This Project Is
An RL-trained **defense drone** that learns to intercept (crash into) an incoming **attack drone** to protect a target. The defender is trained using reinforcement learning in a 3D physics simulation, with the goal of eventually deploying on real hardware.

## Tech Stack
- **Simulator**: gym-pybullet-drones (3D quadrotor physics, multi-agent, Gymnasium API)
- **RL Framework**: Stable Baselines3 (SAC algorithm primary, PPO backup)
- **Language**: Python 3.10+
- **Hardware**: NVIDIA DGX Spark (128GB GPU) — compute is not a constraint
- **Future hardware target**: CrazyFlie 2.1 nano-drones for sim-to-real

## Architecture
- `envs/intercept_env.py` — Custom Gymnasium environment wrapping gym-pybullet-drones
  - Two drones: defender (RL agent) + attacker (scripted, then RL-trained)
  - State: positions, velocities, relative vectors (~19 dims)
  - Action: 4 motor RPMs (continuous)
  - Reward: dense (-distance) + collision bonus + failure penalty
- `train/train_defender.py` — SAC training pipeline with TensorBoard logging
- `train/config.py` — Hyperparameters and environment settings
- `eval/evaluate.py` — Model evaluation with PyBullet GUI visualization

## Development Phases
1. **Phase 1** (current): Custom env + train defender vs scripted attacker
2. **Phase 2**: Curriculum learning (increasingly evasive attacker)
3. **Phase 3**: Adversarial co-training (both drones learn)
4. **Phase 4**: Domain randomization, wind, obstacles
5. **Phase 5**: Sim-to-real transfer with CrazyFlie

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Train the defender
python -m train.train_defender

# Evaluate with visualization
python -m eval.evaluate --model models/defender_sac.zip --gui
```

## Key Design Decisions
- **Why PyBullet over Omniverse**: Faster iteration for algorithm development. Omniverse is planned for later scaling.
- **Why SAC over PPO**: Better sample efficiency for continuous 3D control. PPO is the backup.
- **Why dense reward**: Sparse rewards (only on collision) make learning too slow. Dense distance-based reward provides gradient signal throughout the episode.

## GPU Training Requirements
- **ALWAYS train on CUDA GPU**, never CPU. The DGX Spark GB10 (1536 CUDA cores, 128GB unified memory) was purchased specifically for training.
- Use high `num_worlds` (1024+), large `batch_size` (4096), and `gradient_steps=64` to maximize GPU utilization.
- Use `device="cuda"` in all SAC model creation and loading.
- Monitor GPU utilization with `nvidia-smi dmon -s u` during training — target 50%+ SM utilization.
