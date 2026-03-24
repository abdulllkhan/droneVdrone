# droneVdrone

RL-trained defense drone that learns to intercept (crash into) an incoming attack drone to protect a target.

## How It Works

Two drones in a 3D physics simulation:
- **Defender** (RL agent) — learns to fly toward and collide with the attacker
- **Attacker** (scripted PID) — flies straight toward a target position

The defender is trained using **SAC (Soft Actor-Critic)** in a realistic quadrotor simulation powered by [gym-pybullet-drones](https://github.com/utiasDSL/gym-pybullet-drones).

### Environment Details

| Property | Value |
|---|---|
| Observation | 22-dim: positions, velocities, orientation, relative vector, distance, target |
| Action | 4-dim: motor commands [-1, 1] mapped to RPMs |
| Reward | Dense (approach bonus) + interception (+100) / failure (-100) |
| Physics | PyBullet with ground effect, drag, and downwash |
| Drone model | Crazyflie 2.x (27g nano-quadrotor) |
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

## Usage

### Train the defender

```bash
source venv/bin/activate
python -m train.train_defender
```

Options:
- `--timesteps 2000000` — override total training steps (default: 1M)
- `--gui` — show PyBullet GUI during training (slow, for debugging)

### Monitor training

```bash
tensorboard --logdir logs/tensorboard
```

### Evaluate a trained model

```bash
python -m eval.evaluate --model models/best_model.zip --episodes 20
```

Options:
- `--gui` — visualize episodes in PyBullet 3D viewer
- `--episodes 50` — number of evaluation episodes (default: 20)

### Watch the model in 3D

```bash
python -m eval.evaluate --model models/best_model.zip --gui --episodes 5
```

Opens the PyBullet 3D viewer so you can watch the defender drone attempt to intercept the attacker in real-time.

### Play against the model (manual control)

```bash
python -m eval.play --model models/best_model.zip
```

YOU control the attacker drone with keyboard — try to reach the red target while the trained defender tries to crash into you.

| Key | Action |
|---|---|
| W / S | Forward / backward |
| A / D | Left / right |
| SPACE | Up |
| LEFT SHIFT | Down |
| R | Reset episode |
| ESC | Quit |

Click on the PyBullet window to give it keyboard focus. Run without `--model` to test controls against a random defender.

## Project Structure

```
droneVdrone/
├── envs/
│   └── intercept_env.py    # Custom Gymnasium env (defender vs attacker)
├── train/
│   ├── config.py           # Hyperparameters and env settings
│   └── train_defender.py   # SAC training pipeline
├── eval/
│   ├── evaluate.py         # Model evaluation + stats
│   └── play.py             # Manual play mode (you vs the AI)
├── models/                 # Saved trained models
└── logs/                   # TensorBoard logs
```

## Configuration

All hyperparameters are in `train/config.py`:

- **EnvConfig** — arena bounds, interception thresholds, attacker speed, spawn ranges
- **TrainConfig** — SAC hyperparameters, learning rate, batch size, checkpoint frequency

## Roadmap

- [x] Phase 1: Custom env + train defender vs scripted attacker
- [ ] Phase 2: Curriculum learning (increasingly evasive attacker)
- [ ] Phase 3: Adversarial co-training (both drones learn)
- [ ] Phase 4: Domain randomization, wind, obstacles
- [ ] Phase 5: Sim-to-real transfer with Crazyflie hardware

## Tech Stack

- **Simulator**: gym-pybullet-drones (PyBullet physics)
- **RL**: Stable Baselines3 (SAC)
- **Hardware**: Developed on NVIDIA DGX Spark (128GB GPU)
