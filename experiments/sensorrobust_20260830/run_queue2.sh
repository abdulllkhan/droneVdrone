#!/bin/bash
# Resume the sensor-robust fine-tune to convergence at 192 worlds (SR_train OOM'd
# at 24M under Gemma contention). Then re-eval K=3 sensor cells for a clean verdict.
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/sensorrobust_20260830/queue.log
SR=models/aegis_sensrobust_stage3_realism_best/best_model.zip
E="python -m eval.eval_nvm_grid --model $SR --realism --set kamikaze=False --set num_targets=3 --set target_spread=180.0 --cells 8x8,15x15 --episodes 48 --strategies direct,saturation"
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START SR_train2
timeout 14400 python -m train.train_aegis --start-stage 3 --resume $SR --tag sensrobust \
  --num-worlds-scale 0.375 --timesteps-scale 0.8 \
  --set n_max=20 --set m_max=20 --set "n_range=(6,20)" --set "m_range=(3,20)" \
  --set "terrains=('hills','none')" --set num_obstacles=5 --set kamikaze=False \
  --set "attacker_strategies=('direct','flanking','saturation','waves')" \
  --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 \
  --set "target_count_range=(1,4)" --set target_spread=180.0 \
  --set "sensor_range_range=(80.0,350.0)" \
  > experiments/sensorrobust_20260830/train2.log 2>&1
step "END(exit=$?)" SR_train2

step START SR_eval2
timeout 3000 $E --tag sensrobust2_base_k3 > experiments/sensorrobust_20260830/base2.log 2>&1
timeout 3000 $E --set sensor_range=150.0 --tag sensrobust2_s150_k3 > experiments/sensorrobust_20260830/s150_2.log 2>&1
timeout 3000 $E --set sensor_range=75.0 --tag sensrobust2_s75_k3 > experiments/sensorrobust_20260830/s75_2.log 2>&1
timeout 3000 python -m eval.eval_nvm_grid --model $SR --realism --set kamikaze=False \
  --set num_targets=1 --cells 8x8,15x15 --episodes 48 --strategies direct,saturation \
  --tag sensrobust2_k1 > experiments/sensorrobust_20260830/k1_2.log 2>&1
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" SR_eval2
step SENSROBUST2_DONE queue
