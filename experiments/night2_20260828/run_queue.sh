#!/bin/bash
# Night-2 part 1: CrazyFlie lab fine-tune (Strategy A unit-mapping) + gate evals
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/night2_20260828/queue.log
CF=models/aegis_cf_lab_stage3_realism_best/best_model.zip
LP=models/aegis_largeP_stage3_realism_best/best_model.zip
CELLS=1x1,2x1,2x2,3x1,3x2,3x3,1x2,2x3,1x3
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START L2_cf_lab_train
timeout 14400 python -m train.train_aegis --start-stage 3 --resume $LP --tag cf_lab \
  --timesteps-scale 0.42 \
  --set n_max=4 --set m_max=3 --set "n_range=(1,4)" --set "m_range=(1,3)" \
  --set terrain=none --set num_obstacles=0 --set wind_enabled=False \
  --set sensor_range=350.0 --set obs_noise_std=0.02 \
  --set drone_mass_variance=0.1 --set attacker_speed_variance=0.1 \
  --set "attacker_strategies=('direct','flanking','jink')" \
  --set kamikaze=False --set cull_prob=0.15 \
  --set obs_opportunistic=False --set obs_zero_pip=True \
  --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 \
  > experiments/night2_20260828/L2_train.log 2>&1
step "END(exit=$?)" L2_cf_lab_train

step START L3_gate
if [ -f "$CF" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $CF \
  --cells $CELLS --episodes 200 --strategies direct,jink \
  --set kamikaze=False --set obs_opportunistic=False --set obs_zero_pip=True \
  --tag cf_lab_gate > experiments/night2_20260828/L3_gate.log 2>&1
timeout 3600 python -m eval.eval_nvm_grid --model $LP \
  --cells $CELLS --episodes 200 --strategies direct,jink \
  --set kamikaze=False --set obs_opportunistic=False --set obs_zero_pip=True \
  --tag cf_lab_control > experiments/night2_20260828/L3_control.log 2>&1
fi
step "END(exit=$?)" L3_gate
step L_DONE queue
