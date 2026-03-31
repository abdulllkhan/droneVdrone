# droneVdrone

RL-trained defense drones that learn to intercept incoming attack drones to protect a target. Supports 1v1 through NvM swarm scenarios, trained through progressive curriculum learning and adversarial co-training on GPU-accelerated physics.

## How It Works

Drones in a 3D physics simulation:
- **Defenders** (RL agents) — learn to fly toward and intercept attackers
- **Attackers** — scripted PID → RL-trained → evasive through adversarial co-training

### Training Pipeline

1. **Phase 1**: Fly to random static targets (learn basic flight)
2. **Phase 2**: Chase a moving target at increasing speeds (learn pursuit)
3. **Phase 3**: Intercept an RL-trained attacker (learn interception)
4. **Adversarial**: Co-train attacker and defender (arms race)
5. **Mixture**: Train against pool of attacker generations (prevent forgetting)
6. **Swarm**: Scale to NvM with CTDE parameter sharing

Weights transfer between phases via `SAC.load()` — each phase builds on the previous one.

### Physics

| Property | Value |
|---|---|
| Engine | Newton/NVIDIA Warp (GPU CUDA) |
| Parallel envs | 256 (default, up to 4096) |
| Force model | Direct world-frame forces (no propeller sim) |
| Drone model | Crazyflie 2.x (250g, 8N max force) |
| Intercept distance | 0.75m |
| Control freq | 48 Hz (5 substeps at 240 Hz) |
| Arena | 8m × 8m × 4m |

### Observation Spaces

| Mode | Dims | Contents |
|---|---|---|
| 1v1 | 35 | Self pos/vel, opponent pos/vel, relative, closing rate, accel, obstacles |
| Swarm (NvM) | 57 | Self state, sorted attackers (×3), teammates (×2), obstacles (×3), target |

### Training Results

Full evaluation matrix (interception rate, 50 episodes each):

| Defender | Mix Att R4 | Adv Att R4 | Adv Att R2 | Phase2 | Scripted |
|---|---|---|---|---|---|
| **Best (Mix Def R4)** | **22%** | **20%** | **94%** | **96%** | **100%** |
| Adv Def R4 | 12% | 20% | 98% | 92% | 96% |
| Phase 3 Def | 14% | 4% | 86% | 90% | 98% |

**Best model**: `defender_best_v5.zip` — mixture-trained against pool of RL attackers (1.4 m/s, faster than defender's ~1.25 m/s) with 3 obstacles, variable altitude target (1.0–3.0m), and 250g drone mass. **Zero attacker target reaches** across all matchups.

Key findings:
- **Speed asymmetry** (attacker 1.4 m/s vs defender ~1.25 m/s) — interception requires prediction, not just pursuit
- **3 static obstacles** add environmental complexity (proximity penalties in reward)
- **Mixture-of-opponents** training prevents catastrophic forgetting across attacker generations
- **Variable target altitude** (1.0–3.0m) forces 3D spatial awareness

#### Swarm Results (2v2, 20 episodes each)

| Defender | vs Scripted | vs Flanking | vs Mixed | vs RL Att R2 |
|---|---|---|---|---|
| **Adv Def R2** | **100%** | **100%** | **100%** | **95%** |
| Original 2v2 | 100% | 100% | 100% | 25% |
| Adv Def R1 | - | - | - | 0% |

Adversarial co-training (2 rounds) improved RL attacker interception from 25% → 95%. Zero target reaches across all matchups. Flanking/mixed scripted strategies don't challenge trained defenders.

#### 3v3 Swarm Results (20 episodes each)

| Defender | vs Scripted | vs Flanking | vs Mixed |
|---|---|---|---|
| **3v3 v2** | **95%** | **95%** | **100%** |

3v3 is harder with more attackers to track. Mixed strategy easier to intercept than flanking in 3v3 (flanking forces defenders to split across more angles).

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

### Progressive Training (1v1)

```bash
# All phases (fly → chase → intercept)
python -m train.train_progressive --num-envs 256

# Individual phases
python -m train.train_progressive --phase 1                                    # fly to target
python -m train.train_progressive --phase 2 --resume models/phase1_fly_final.zip  # chase
python -m train.train_progressive --phase 3 --attacker models/phase2_chase_speed0.5_final.zip  # intercept
```

### Adversarial Co-Training (1v1)

```bash
# Train attacker and defender against each other in alternating rounds
python -m train.train_adversarial --defender models/defender_v2_99pct.zip --rounds 4
python -m train.train_adversarial --defender models/best_model.zip --attacker models/phase2v2_best_attacker.zip --rounds 6
```

### Swarm Adversarial Co-Training (NvM)

```bash
# 2v2 adversarial — train RL swarm attackers vs frozen defenders, then retrain defenders
python -m train.train_swarm_adversarial --defenders 2 --attackers 2 --defender models/swarm_2v2_final.zip --rounds 4

# With mixture-of-opponents (prevents overfitting to single attacker generation)
python -m train.train_swarm_adversarial --defenders 2 --attackers 2 --defender models/swarm_2v2_final.zip --mixture --rounds 4

# 3v2 asymmetric adversarial with wind + obstacles
python -m train.train_swarm_adversarial --defenders 3 --attackers 2 --defender models/swarm_3v2_final.zip --wind --obstacles 3 --rounds 3
```

### Swarm Training (NvM)

```bash
# 2v2 with obstacles and wind
python -m train.train_swarm --defenders 2 --attackers 2 --num-worlds 128 --obstacles 3 --wind

# Asymmetric configs
python -m train.train_swarm --defenders 2 --attackers 1 --num-worlds 128 --obstacles 3 --wind
python -m train.train_swarm --defenders 3 --attackers 2 --num-worlds 64 --obstacles 3 --wind

# 3v3 larger scenario
python -m train.train_swarm --defenders 3 --attackers 3 --num-worlds 64 --timesteps 3000000

# Curriculum training (progressive difficulty)
python -m train.train_swarm_curriculum --defenders 2 --attackers 2 --num-worlds 128

# Resume from checkpoint
python -m train.train_swarm --defenders 2 --attackers 2 --resume models/swarm_2v2_final.zip
```

### Evaluation

```bash
# 1v1 evaluation with plots and GIF
python -m eval.eval_progressive --model models/defender_best_v5.zip --mode intercept --attacker models/adv_attacker_round1.zip --gif --episodes 50

# Swarm evaluation with trajectory grid
python -m eval.eval_swarm --model models/swarm_2v2_final.zip --defenders 2 --attackers 2 --episodes 20 --gif
python -m eval.eval_swarm --model models/swarm_2v1_final.zip --defenders 2 --attackers 1 --episodes 20
python -m eval.eval_swarm --model models/swarm_3v2_final.zip --defenders 3 --attackers 2 --episodes 20

# Swarm cross-evaluation matrix (auto-discovers models)
python -m eval.eval_swarm_matrix --defenders 2 --attackers 2 --episodes 20
python -m eval.eval_swarm_matrix --defenders 2 --attackers 2 --defender-models models/swarm_2v2_final.zip models/swarm_adv_defender_2v2_round1.zip
```

### Interactive Game

```bash
# Free flight — fly a drone with keyboard (WASD + Space/Shift)
python -m eval.game

# Watch RL 1v1 — best defender vs attacker
python -m eval.game --mode watch --defender models/defender_best_v5.zip --attacker models/adv_attacker_round1.zip

# You defend — control the blue drone, RL attacker targets
python -m eval.game --mode defend --attacker models/mix_attacker_round2.zip

# You attack — control the red drone, RL defender intercepts
python -m eval.game --mode attack --defender models/defender_best_v5.zip

# Swarm mode — watch NvM defenders vs attackers
python -m eval.game --mode swarm --defenders 2 --attackers 2 --defender models/swarm_2v2_final.zip

# Swarm vs RL attackers
python -m eval.game --mode swarm --defenders 2 --attackers 2 --defender models/swarm_adv_defender_2v2_round2.zip --attacker-model models/swarm_adv_attacker_2v2_round2.zip

# Swarm with flanking strategy
python -m eval.game --mode swarm --defenders 2 --attackers 2 --defender models/swarm_2v2_final.zip --strategy flanking

# Larger arena
python -m eval.game --mode watch --arena 12 --defender models/defender_best_v5.zip --attacker models/adv_attacker_round2.zip
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
│   ├── newton_env.py             # 1v1 GPU env (fly/chase/intercept/evade + wind/noise)
│   ├── newton_swarm_env.py       # NvM swarm GPU env (CTDE parameter sharing)
│   ├── newton_intercept_env.py   # Legacy Newton env (reference)
│   └── intercept_env.py          # PyBullet env (CPU, legacy)
├── train/
│   ├── config.py                 # Hyperparameters and env settings
│   ├── train_progressive.py      # Progressive training pipeline (1v1)
│   ├── train_adversarial.py      # Adversarial co-training (1v1)
│   ├── train_swarm.py            # Swarm training (NvM)
│   ├── train_swarm_curriculum.py # Swarm curriculum (progressive difficulty)
│   └── train_swarm_adversarial.py # Swarm adversarial co-training (NvM arms race)
├── eval/
│   ├── eval_progressive.py       # 1v1 evaluation with plots and GIFs
│   ├── eval_swarm.py             # Swarm evaluation with trajectory grids
│   ├── eval_matrix.py            # Cross-evaluation of defender×attacker pairs (1v1)
│   ├── eval_swarm_matrix.py      # Swarm defender×attacker cross-evaluation (NvM)
│   └── game.py                   # Interactive game (free/defend/attack/watch/swarm)
├── tests/
│   ├── test_newton_env.py        # 1v1 env tests (46 tests)
│   └── test_swarm_env.py         # Swarm env tests (38 tests)
├── models/                       # Saved trained models (gitignored)
└── logs/                         # TensorBoard logs (gitignored)
```

## Configuration

All hyperparameters in `train/config.py`:

**EnvConfig** — physics, arena, interception thresholds, spawn ranges, swarm, domain randomization:
- `drone_mass = 0.250` — 250g quadrotor (Crazyflie-scale)
- `max_force = 8.0` — 3.3× drone weight for maneuvering
- `intercept_dist = 0.75` — interception halo radius
- `attacker_speed_limit = 1.4` — faster than defender (~1.25 m/s)
- `num_defenders / num_attackers` — swarm configuration (1–3)
- `wind_enabled / wind_max_force` — random wind perturbation
- `obs_noise_std` — Gaussian observation noise
- `num_obstacles` — static obstacle pillars (default 3)
- `target_alt_randomize` — variable target elevation (1.0–3.0m)
- `attacker_strategy` — scripted attacker behavior: `"direct"`, `"flanking"`, `"mixed"`

**TrainConfig** — SAC hyperparameters:
- `ent_coef = 0.1` — fixed entropy (auto collapsed to 0.003)
- `net_arch = [512, 512, 256]` — 3-layer MLP
- Manual observation normalization in GPU kernels (pos/4, vel/5, rel/6)
- Early stopping after 100 evals with no improvement

## Domain Randomization

Both 1v1 and swarm environments support:
- **Wind**: Random force vectors applied to all drones, direction changes periodically
- **Observation noise**: Gaussian noise on all observation dimensions
- **Variable target altitude**: Target height randomized 1.0–3.0m each episode
- **Obstacle layout**: Obstacles re-randomized each episode with minimum spacing

```bash
# 1v1 with domain randomization (via config)
# Swarm with all randomization
python -m train.train_swarm --defenders 2 --attackers 2 --wind --wind-force 0.3 --noise 0.02
```

## Roadmap

- [x] Phase 1: Fly to random static targets (96% arrival rate)
- [x] Phase 2: Chase moving targets with speed curriculum (94% at 0.5 m/s)
- [x] Phase 3: Intercept RL-trained attacker (100% vs Phase 2 attacker)
- [x] Adversarial co-training (attacker/defender arms race)
- [x] GPU-accelerated physics (Newton/Warp — 256 parallel envs)
- [x] Velocity-based control (sim-to-real compatible)
- [x] Mixture-of-opponents training (prevents adversarial overfitting)
- [x] Multi-drone swarm (NvM with CTDE parameter sharing)
- [x] Domain randomization (wind, observation noise)
- [x] Static obstacles with proximity penalties
- [x] Variable target altitude (1.0–3.0m)
- [x] 250g drone mass (realistic Crazyflie weight)
- [x] Dynamic/moving obstacles (configurable speed + bounce)
- [x] Significant obstacle collision penalties (-50 contact, -10x proximity)
- [x] Action delay simulation (configurable latency for sim-to-real)
- [x] Asymmetric swarm configs (2v1, 3v2, 1v3)
- [x] Swarm curriculum training (progressive difficulty stages)
- [x] Swarm evaluation tool with trajectory grids + GIFs
- [x] Swarm adversarial co-training (train NvM RL attacker swarms)
- [x] Coordinated attacker strategies (flanking, mixed)
- [x] Priority targeting reward shaping (focus on most-threatening attacker)
- [x] Swarm evaluation matrix (cross-evaluate defender×attacker pairs)
- [ ] Sim-to-real transfer with Crazyflie hardware

## Tech Stack

- **Physics**: Newton/NVIDIA Warp (GPU)
- **RL**: Stable Baselines3 (SAC)
- **Hardware**: NVIDIA DGX Spark (GB10 GPU, 128GB)
- **Game**: Pygame (isometric visualization)
