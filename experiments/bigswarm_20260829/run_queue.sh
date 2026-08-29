#!/bin/bash
# Exploit the lifted OOM cap: train a 24v24 generalist at 448 worlds (both beyond
# the old 20v20 / 256-384-world ceiling). Warm from the fresh full-worlds champion.
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/bigswarm_20260829/queue.log
WARM=models/aegis_oomval_stage3_realism_best/best_model.zip
BIG=models/aegis_big24_stage3_realism_best/best_model.zip
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START T_train_24v24
timeout 14400 python -m train.train_aegis --start-stage 3 --resume $WARM --tag big24 \
  --num-worlds-scale 0.875 --timesteps-scale 0.6 \
  --set n_max=24 --set m_max=24 --set "n_range=(6,24)" --set "m_range=(1,24)" \
  --set "terrains=('hills','urban','maze')" --set num_obstacles=5 --set kamikaze=False \
  --set "attacker_strategies=('direct','flanking','jink','saturation','waves','decoy')" \
  --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 \
  > experiments/bigswarm_20260829/train.log 2>&1
step "END(exit=$?)" T_train_24v24

step START E_evals
if [ -f "$BIG" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $BIG --realism --set kamikaze=False \
  --cells 15x15,20x20,24x24 --episodes 32 --strategies direct,saturation,decoy \
  --tag big24_terrain > experiments/bigswarm_20260829/eval_terrain.log 2>&1
timeout 3600 python -m eval.eval_nvm_grid --model $BIG --set kamikaze=False \
  --cells 30x30,40x40,50x40,20x40 --episodes 20 --strategies direct,flanking \
  --tag big24_scale > experiments/bigswarm_20260829/eval_scale.log 2>&1
fi
timeout 1200 python -m eval.render_aegis --model ${BIG:-$WARM} --nm 24x24 --strategy saturation \
  --terrain hills --stride 2 --fps 16 --seed 51 \
  --out models/eval_output/aegis_big24_24v24.gif > /dev/null 2>&1
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" E_evals
step BIGSWARM_DONE queue
