"""
Evaluate all defender × attacker combinations and print results matrix.

Usage:
    python -m eval.eval_matrix
"""
import os

import numpy as np
import warp as wp
from stable_baselines3 import SAC

from envs.newton_env import NewtonDroneEnv
from train.config import EnvConfig

wp.init()

NUM_EPISODES = 50
NUM_ENVS = 8
MAX_STEPS = 480

defenders = {
    "Original (Phase3)": "models/defender_v2_99pct.zip",
    "Adv Round1": "models/adv_defender_round1.zip",
    "Adv Round2": "models/adv_defender_round2.zip",
    "Mix Round1": "models/mix_defender_round1.zip",
    "Mix Round2": "models/mix_defender_round2.zip",
    "Mix Round3": "models/mix_defender_round3.zip",
}

attackers = {
    "Phase2": "models/phase2v2_best_attacker.zip",
    "Adv R1": "models/adv_attacker_round1.zip",
    "Adv R2": "models/adv_attacker_round2.zip",
    "Mix R1": "models/mix_attacker_round1.zip",
    "Mix R2": "models/mix_attacker_round2.zip",
    "Mix R3": "models/mix_attacker_round3.zip",
    "Scripted": None,
}


def eval_matchup(defender_path, attacker_path, num_episodes=NUM_EPISODES):
    """Run episodes and return (intercepted, target_reached, timeout) counts."""
    cfg_kwargs = {"mode": "intercept"}
    if attacker_path:
        cfg_kwargs["attacker_model_path"] = attacker_path
    cfg = EnvConfig(**cfg_kwargs)
    env = NewtonDroneEnv(num_envs=NUM_ENVS, config=cfg)
    model = SAC.load(defender_path)

    obs = env.reset()
    intercepted = 0
    target_reached = 0
    timeout = 0
    completed = 0

    while completed < num_episodes:
        batch = min(NUM_ENVS, num_episodes - completed)
        done_mask = np.zeros(batch, dtype=bool)
        outcomes = ["timeout"] * batch
        obs = env.reset()

        for step in range(MAX_STEPS):
            actions, _ = model.predict(obs[:batch], deterministic=True)
            if NUM_ENVS > batch:
                full_actions = np.zeros((NUM_ENVS, actions.shape[1]), dtype=np.float32)
                full_actions[:batch] = actions
                obs, rewards, dones, infos = env.step(full_actions)
            else:
                obs, rewards, dones, infos = env.step(actions)

            for i in range(batch):
                if dones[i] and not done_mask[i]:
                    done_mask[i] = True
                    if infos[i].get("intercepted"):
                        outcomes[i] = "intercepted"
                    elif infos[i].get("target_reached"):
                        outcomes[i] = "target_reached"
            if all(done_mask):
                break

        for i in range(batch):
            if outcomes[i] == "intercepted":
                intercepted += 1
            elif outcomes[i] == "target_reached":
                target_reached += 1
            else:
                timeout += 1
            completed += 1

    env.close()
    return intercepted, target_reached, timeout


# Run all matchups
print(f"\n{'='*100}")
print(f"  EVALUATION MATRIX ({NUM_EPISODES} episodes per matchup)")
print(f"{'='*100}\n")

# Header
att_names = list(attackers.keys())
header = f"{'Defender':<20}" + "".join(f"{'| ' + n + ' ':>14}" for n in att_names)
print(header)
print("-" * len(header))

results = {}
for def_name, def_path in defenders.items():
    if not os.path.exists(def_path):
        print(f"{def_name:<20} [MODEL NOT FOUND]")
        continue
    
    row = f"{def_name:<20}"
    for att_name, att_path in attackers.items():
        if att_path and not os.path.exists(att_path):
            row += f"{'N/A':>14}"
            continue
        
        i, t, o = eval_matchup(def_path, att_path)
        pct = i / NUM_EPISODES * 100
        row += f"| {i:2d}/{NUM_EPISODES} ({pct:3.0f}%) "
        results[(def_name, att_name)] = (i, t, o)
    
    print(row)

print(f"\n{'='*100}")
print("Format: intercepted/total (intercept%)")
print(f"{'='*100}\n")

# Summary
print("\nBest generalist (highest min interception across all attackers):")
for def_name in defenders:
    if not os.path.exists(defenders[def_name]):
        continue
    min_pct = 100
    for att_name in attackers:
        key = (def_name, att_name)
        if key in results:
            pct = results[key][0] / NUM_EPISODES * 100
            min_pct = min(min_pct, pct)
    print(f"  {def_name}: min {min_pct:.0f}%")
