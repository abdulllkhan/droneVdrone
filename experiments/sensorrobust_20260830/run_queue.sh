#!/bin/bash
# Is multi-target sensor fragility a TRAINING GAP or fundamental? Fine-tune vark
# with degraded sensing (sensor_range sampled 80-350) + variable K, then re-run
# the sensor stress at K=3: does it hold up at 150m/75m now?
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/sensorrobust_20260830/queue.log
WARM=models/aegis_vark_stage3_realism_best/best_model.zip
SR=models/aegis_sensrobust_stage3_realism_best/best_model.zip
E="python -m eval.eval_nvm_grid --model $SR --realism --set kamikaze=False --set num_targets=3 --set target_spread=180.0 --cells 8x8,15x15 --episodes 48 --strategies direct,saturation"
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START SR_train
timeout 14400 python -m train.train_aegis --start-stage 3 --resume $WARM --tag sensrobust \
  --num-worlds-scale 0.5 --timesteps-scale 0.8 \
  --set n_max=20 --set m_max=20 --set "n_range=(6,20)" --set "m_range=(3,20)" \
  --set "terrains=('hills','none')" --set num_obstacles=5 --set kamikaze=False \
  --set "attacker_strategies=('direct','flanking','saturation','waves')" \
  --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 \
  --set "target_count_range=(1,4)" --set target_spread=180.0 \
  --set "sensor_range_range=(80.0,350.0)" \
  > experiments/sensorrobust_20260830/train.log 2>&1
step "END(exit=$?)" SR_train

step START SR_eval
if [ -f "$SR" ]; then
timeout 3000 $E --tag sensrobust_base_k3 > experiments/sensorrobust_20260830/base.log 2>&1
timeout 3000 $E --set sensor_range=150.0 --tag sensrobust_s150_k3 > experiments/sensorrobust_20260830/s150.log 2>&1
timeout 3000 $E --set sensor_range=75.0 --tag sensrobust_s75_k3 > experiments/sensorrobust_20260830/s75.log 2>&1
# K=1 regression (must stay perfect)
timeout 3000 python -m eval.eval_nvm_grid --model $SR --realism --set kamikaze=False \
  --set num_targets=1 --cells 8x8,15x15 --episodes 48 --strategies direct,saturation \
  --tag sensrobust_k1 > experiments/sensorrobust_20260830/k1.log 2>&1
fi
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" SR_eval
step SENSROBUST_DONE queue
