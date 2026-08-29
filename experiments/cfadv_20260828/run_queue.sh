#!/bin/bash
# cf_lab adversarial verification: RL attackers vs the CrazyFlie-ready policy
# (small teams, minimal obs, no terrain — mirrors cf_lab's training world)
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/cfadv_20260828/queue.log
CF=models/aegis_cf_lab_stage3_realism_best/best_model.zip
ADV=models/aegis_adv_defender_cf_lab_best/best_model.zip
ATT=models/aegis_adv_attacker_cf_lab_best/best_model.zip
ENVSET='--env-set n_max=4 --env-set m_max=3'
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START A_cfadv_cotrain
timeout 14400 python -m train.train_aegis_adv --defender $CF --kamikaze-false \
  --round-tag cf_lab --num-worlds 512 \
  --env-set n_max=4 --env-set m_max=3 \
  --env-set "n_range=(1,4)" --env-set "m_range=(1,3)" \
  --env-set terrain=none --env-set num_obstacles=0 --env-set wind_enabled=False \
  --env-set sensor_range=350.0 --env-set obs_noise_std=0.02 \
  --env-set drone_mass_variance=0.1 --env-set attacker_speed_variance=0.1 \
  --env-set obs_opportunistic=False --env-set obs_zero_pip=True \
  --env-set r_bad_death=-60.0 --env-set r_safety_coeff=-0.10 \
  > experiments/cfadv_20260828/cotrain.log 2>&1
step "END(exit=$?)" A_cfadv_cotrain

step START B_gate_evals
if [ -f "$ADV" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $ADV \
  --cells 1x1,2x1,2x2,3x2,3x3,1x2,2x3 --episodes 200 --strategies rl \
  --set kamikaze=False --set obs_opportunistic=False --set obs_zero_pip=True \
  --set attacker_model_path=$ATT \
  --tag cfadv_vs_rl > experiments/cfadv_20260828/eval_vs_rl.log 2>&1
timeout 3600 python -m eval.eval_nvm_grid --model $ADV \
  --cells 1x1,2x1,2x2,3x1,3x2,3x3,1x2,2x3,1x3 --episodes 200 --strategies direct,jink \
  --set kamikaze=False --set obs_opportunistic=False --set obs_zero_pip=True \
  --tag cfadv_scripted > experiments/cfadv_20260828/eval_scripted.log 2>&1
fi
if [ -f "$ATT" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $CF \
  --cells 1x1,2x1,2x2,3x2,3x3,1x2,2x3 --episodes 200 --strategies rl \
  --set kamikaze=False --set obs_opportunistic=False --set obs_zero_pip=True \
  --set attacker_model_path=$ATT \
  --tag cf_vs_rl > experiments/cfadv_20260828/eval_cf_vs_rl.log 2>&1
fi
step "END(exit=$?)" B_gate_evals

step START C_dashboard
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" C_dashboard
step CFADV_DONE queue
