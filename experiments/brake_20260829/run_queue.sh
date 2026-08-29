#!/bin/bash
# Density-invariant attrition A/B: physical obstacle speed-brake. Fine-tune big24
# WITH the brake, then compare terrain deaths vs big24 at BOTH densities (20v20
# and 50v40) — the safety-shaping A/B failed exactly this cross-density test.
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/brake_20260829/queue.log
WARM=models/aegis_big24_stage3_realism_best/best_model.zip
BRK=models/aegis_brake_stage3_realism_best/best_model.zip
BRAKESET='--set obstacle_speed_limit=0.25 --set obstacle_brake_range=60.0'
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START B_train_brake
timeout 14400 python -m train.train_aegis --start-stage 3 --resume $WARM --tag brake \
  --num-worlds-scale 0.875 --timesteps-scale 0.6 \
  --set n_max=20 --set m_max=20 --set "n_range=(4,20)" --set "m_range=(1,20)" \
  --set "terrains=('hills','urban','maze')" --set num_obstacles=5 --set kamikaze=False \
  --set "attacker_strategies=('direct','flanking','jink','saturation','waves','decoy')" \
  --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 \
  --set obstacle_speed_limit=0.25 --set obstacle_brake_range=60.0 \
  > experiments/brake_20260829/train.log 2>&1
step "END(exit=$?)" B_train_brake

step START B_eval
if [ -f "$BRK" ]; then
# brake model eval WITH brake (matches its training dynamics)
timeout 3600 python -m eval.eval_nvm_grid --model $BRK --realism --set kamikaze=False \
  --set obstacle_speed_limit=0.25 --set obstacle_brake_range=60.0 \
  --cells 10x10,20x20 --episodes 32 --strategies saturation,decoy \
  --tag brake_dense > experiments/brake_20260829/eval_brake_dense.log 2>&1
timeout 3600 python -m eval.eval_nvm_grid --model $BRK --set kamikaze=False \
  --set obstacle_speed_limit=0.25 --set obstacle_brake_range=60.0 \
  --cells 40x40,50x40 --episodes 20 --strategies direct,flanking \
  --tag brake_scale > experiments/brake_20260829/eval_brake_scale.log 2>&1
fi
timeout 1200 python -m eval.render_aegis --model ${BRK:-$WARM} --nm 12x12 --strategy saturation \
  --terrain maze --stride 1 --fps 14 --seed 61 \
  --out models/eval_output/aegis_brake_maze_12v12.gif > /dev/null 2>&1
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" B_eval
step BRAKE_DONE queue
