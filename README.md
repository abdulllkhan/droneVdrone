# droneVdrone

RL-trained defense drone that learns to intercept (crash into) an incoming attack drone to protect a target.

## How It Works

Two drones in a 3D physics simulation:
- **Defender** (RL agent) — learns to fly toward and intercept the attacker
- **Attacker** (scripted PID) — flies straight toward a target position

The defender is trained using **SAC (Soft Actor-Critic)** with velocity-based control — the RL agent outputs desired velocities and a PID controller handles low-level motor commands.

### Two Physics Backends

| Backend | Engine | Runs on | Parallel envs | Use case |
|---|---|---|---|---|
| **PyBullet** | gym-pybullet-drones | CPU | 4-16 | Prototyping, 3D visualization |
| **Newton** | NVIDIA Warp | GPU (CUDA) | 256-4096 | Fast training (~530x speedup) |

### Environment Details

| Property | Value |
|---|---|
| Observation | 19-dim: positions, velocities, relative vector, distance, target |
| Action | 3-dim: desired velocity (vx, vy, vz) in [-1, 1], PID converts to RPMs |
| Reward | Dense (approach + proximity bonus + altitude match) + interception (+100) / failure (-100) |
| Drone model | Crazyflie 2.x (27g nano-quadrotor) |
| Intercept distance | 0.5m (close enough to deploy countermeasure) |
| Control freq | 48 Hz (5 physics substeps at 240 Hz) |

## Setup

### Prerequisites

```bash
# Ubuntu/Debian — needed to compile PyBullet
sudo apt-get install python3-dev build-essential
```

### Install

```bash
git clone git@github.com:abdulllkhan/droneVdrone.git
cd droneVdrone

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

For GPU-accelerated training (Newton backend), also install:

```bash
pip install newton
```

For CUDA on DGX Spark (GB10 GPU), install the nightly PyTorch build:

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```

## Usage

### Train the defender (PyBullet — CPU)

```bash
source venv/bin/activate
python -m train.train_defender
```

Options:
- `--timesteps 2000000` — override total training steps (default: 1M)
- `--gui` — show PyBullet GUI during training (slow, for debugging)

### Train the defender (Newton — GPU)

```bash
source venv/bin/activate
python -m train.train_defender_newton --num-envs 256
```

Options:
- `--timesteps 50000000` — override total training steps (default: 1M)
- `--num-envs 256` — number of parallel GPU environments (default: from config)

### Monitor training

```bash
source venv/bin/activate
tensorboard --logdir logs/tensorboard
# Open http://localhost:6006
```

### Evaluate a trained model

PyBullet (with 3D visualization):
```bash
python -m eval.evaluate --model models/best_model.zip --episodes 20
python -m eval.evaluate --model models/best_model.zip --gui --episodes 5
```

Newton (fast, no GUI):
```bash
python -m eval.evaluate_newton --model models/best_model.zip --episodes 50
```

## Project Structure

```
droneVdrone/
├── envs/
│   ├── intercept_env.py          # PyBullet Gymnasium env (CPU)
│   └── newton_intercept_env.py   # Newton GPU VecEnv (CUDA)
├── train/
│   ├── config.py                 # Hyperparameters and env settings
│   ├── train_defender.py         # SAC training (PyBullet)
│   └── train_defender_newton.py  # SAC training (Newton GPU)
├── eval/
│   ├── evaluate.py               # PyBullet evaluation + 3D viewer
│   └── evaluate_newton.py        # Newton evaluation
├── tests/
│   └── test_newton_env.py        # Unit tests for Newton env
├── models/                       # Saved trained models (gitignored)
└── logs/                         # TensorBoard logs (gitignored)
```

## Configuration

All hyperparameters are in `train/config.py`:

- **EnvConfig** — arena bounds, interception thresholds, attacker speed, spawn ranges
- **TrainConfig** — SAC hyperparameters, learning rate, batch size, checkpoint frequency

## Roadmap

- [x] Phase 1: Custom env + train defender vs scripted attacker
- [x] Velocity-based control (PID handles flight stability)
- [x] GPU-accelerated physics (Newton/Warp backend — 530x speedup)
- [ ] Phase 2: Curriculum learning (increasingly evasive attacker)
- [ ] Phase 3: Adversarial co-training (both drones learn)
- [ ] Phase 4: Domain randomization, wind, obstacles
- [ ] Phase 5: Sim-to-real transfer with Crazyflie hardware

## Tech Stack

- **Simulators**: gym-pybullet-drones (CPU) + Newton/Warp (GPU)
- **RL**: Stable Baselines3 (SAC)
- **Hardware**: NVIDIA DGX Spark (GB10 GPU, 128GB)
