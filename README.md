# droneVdrone

RL-trained defense drone that learns to intercept an incoming attack drone to protect a target. Trained through progressive curriculum learning and adversarial co-training on GPU-accelerated physics.

## How It Works

Two drones in a 3D physics simulation:
- **Defender** (RL agent) — learns to fly toward and intercept the attacker
- **Attacker** — starts as scripted PID, then becomes RL-trained and evasive through adversarial co-training

The defender is trained using **SAC (Soft Actor-Critic)** with a progressive curriculum:
1. **Phase 1**: Fly to random static targets (learn basic flight)
2. **Phase 2**: Chase a moving target at increasing speeds (learn pursuit)
3. **Phase 3**: Intercept an RL-trained attacker (learn interception)
4. **Adversarial**: Co-train attacker and defender against each other (arms race)

Weights transfer between phases via `SAC.load()` — each phase builds on the previous one.

### Physics

| Property | Value |
|---|---|
| Engine | Newton/NVIDIA Warp (GPU CUDA) |
| Parallel envs | 256 (default, up to 4096) |
| Force model | Direct world-frame forces (no propeller sim) |
| Observation | 19-dim: positions, velocities, relative vectors |
| Action | 3-dim: desired velocity (vx, vy, vz) in [-1, 1] |
| Drone model | Crazyflie 2.x (27g nano-quadrotor) |
| Intercept distance | 0.5m |
| Control freq | 48 Hz (5 substeps at 240 Hz) |

### Training Results

Full evaluation matrix (interception rate, 50 episodes each):

| Defender | Phase2 | Adv R1 | Adv R2 | Mix R1 | Mix R2 | Mix R3 | Scripted |
|---|---|---|---|---|---|---|---|
| Original (Phase 3) | 100% | 70% | 6% | 2% | 4% | 92% | 72% |
| Adv Round 1 | 72% | 86% | 36% | 0% | 2% | 100% | 68% |
| **Best Generalist** | **100%** | **98%** | **46%** | **44%** | **38%** | **100%** | **100%** |

**Best model**: `defender_best_generalist.zip` — mixture-trained against 7 RL attackers + scripted pursuit with reward v3 (closing speed bonus). Maintains 100% against simpler attackers, 38-46% against the hardest evasive opponents. **Zero attacker wins** across all matchups — even failed interceptions deny the target.

Key findings:
- **Mixture-of-opponents training** prevents catastrophic forgetting (Adv Round 2 dropped to 20% vs Phase2 without it)
- **Closing speed reward bonus** (v3) teaches aggressive pursuit behavior
- Progressive curriculum (fly → chase → intercept → adversarial) enables faster convergence via weight transfer

## Setup

```bash
git clone git@github.com:abdulllkhan/droneVdrone.git
cd droneVdrone

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install newton  # GPU physics

# For DGX Spark (GB10 GPU)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall
```

## Usage

### Progressive Training (full pipeline)

```bash
# All phases (fly → chase → intercept)
python -m train.train_progressive --num-envs 256

# Individual phases
python -m train.train_progressive --phase 1                                    # fly to target
python -m train.train_progressive --phase 2 --resume models/phase1_fly_final.zip  # chase
python -m train.train_progressive --phase 3 --attacker models/phase2_chase_speed0.5_final.zip  # intercept
```

### Adversarial Co-Training

```bash
# Train attacker and defender against each other in alternating rounds
python -m train.train_adversarial --defender models/defender_v2_99pct.zip --rounds 4
python -m train.train_adversarial --defender models/best_model.zip --attacker models/phase2v2_best_attacker.zip --rounds 6
```

### Evaluation

```bash
# Evaluate any mode
python -m eval.eval_progressive --model models/best_model.zip --mode fly_to_target
python -m eval.eval_progressive --model models/best_model.zip --mode chase --speed 0.5
python -m eval.eval_progressive --model models/best_model.zip --mode intercept --attacker models/phase2v2_best_attacker.zip

# With trajectory plots and animated GIF
python -m eval.eval_progressive --model models/defender_generalist.zip --mode intercept --attacker models/adv_attacker_round2.zip --gif --episodes 50
```

### Interactive Game (Physics Validation)

```bash
# Free flight — fly a drone with keyboard (WASD + Space/Shift)
python -m eval.game

# Watch RL models fight — best defender vs attacker
python -m eval.game --mode watch --defender models/defender_best_generalist.zip --attacker models/adv_attacker_round1.zip

# You defend — control the blue drone, RL attacker goes for target
python -m eval.game --mode defend --attacker models/mix_attacker_round2.zip

# You attack — control the red drone, RL defender tries to intercept
python -m eval.game --mode attack --defender models/defender_best_generalist.zip

# Larger arena
python -m eval.game --mode watch --arena 12 --defender models/defender_best_generalist.zip --attacker models/adv_attacker_round2.zip
```

Controls: WASD=move, Space=up, Shift=down, Arrows=rotate camera, +/-=zoom, R=reset, Q=quit

### Monitor Training

```bash
tensorboard --logdir logs/tensorboard
```

## Project Structure

```
droneVdrone/
├── envs/
│   ├── newton_env.py             # Unified GPU env (fly/chase/intercept/evade)
│   ├── newton_intercept_env.py   # Legacy Newton env (reference)
│   └── intercept_env.py          # PyBullet env (CPU, legacy)
├── train/
│   ├── config.py                 # Hyperparameters and env settings
│   ├── train_progressive.py      # Progressive training pipeline
│   └── train_adversarial.py      # Adversarial co-training
├── eval/
│   ├── eval_progressive.py       # Evaluation with plots and GIFs
│   └── eval_matrix.py            # Cross-evaluation of all defender×attacker pairs
├── tests/
│   └── test_newton_env.py        # Unit tests for Newton env
├── models/                       # Saved trained models (gitignored)
└── logs/                         # TensorBoard logs (gitignored)
```

## Configuration

All hyperparameters in `train/config.py`:

- **EnvConfig** — physics params, arena bounds, interception thresholds, spawn ranges, training mode
- **TrainConfig** — SAC hyperparameters, learning rate, network architecture, checkpoint frequency

Key settings:
- `ent_coef = 0.1` — fixed entropy to prevent collapse (was "auto" → collapsed to 0.003)
- Manual observation normalization in GPU kernels (pos/4, vel/5, rel/6) — replaced VecNormalize which caused train/eval mismatch
- Early stopping after 100 evals with no improvement

## Roadmap

- [x] Phase 1: Fly to random static targets (96% arrival rate)
- [x] Phase 2: Chase moving targets with speed curriculum (94% at 0.5 m/s)
- [x] Phase 3: Intercept RL-trained attacker (100% vs Phase 2 attacker)
- [x] Adversarial co-training (attacker/defender arms race)
- [x] GPU-accelerated physics (Newton/Warp — 256 parallel envs)
- [x] Velocity-based control (sim-to-real compatible)
- [x] Mixture-of-opponents training (prevents adversarial overfitting)
- [ ] Domain randomization (wind, noise, latency)
- [ ] Sim-to-real transfer with Crazyflie hardware

## Tech Stack

- **Physics**: Newton/NVIDIA Warp (GPU)
- **RL**: Stable Baselines3 (SAC)
- **Hardware**: NVIDIA DGX Spark (GB10 GPU, 128GB)
