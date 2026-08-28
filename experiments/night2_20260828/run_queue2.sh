#!/bin/bash
# Night-2 part 2: convoy fine-tune + gates + GIFs + dashboard
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/night2_20260828/queue.log
LP=models/aegis_largeP_stage3_realism_best/best_model.zip
CV=models/aegis_convoy_stage3_realism_best/best_model.zip
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START C7_convoy_train
timeout 14400 python -m train.train_aegis --start-stage 3 --resume $LP --tag convoy \
  --num-worlds-scale 0.75 --timesteps-scale 0.667 \
  --set n_max=20 --set m_max=20 --set "n_range=(4,20)" --set "m_range=(1,20)" \
  --set "terrains=('hills','urban','maze')" --set num_obstacles=5 --set kamikaze=False \
  --set "attacker_strategies=('direct','flanking','jink','saturation','waves','decoy')" \
  --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 \
  --set "target_speed_range=(0.0,10.0)" \
  > experiments/night2_20260828/C7_train.log 2>&1
step "END(exit=$?)" C7_convoy_train

step START C8_convoy_gates
if [ -f "$CV" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $CV --realism \
  --set kamikaze=False --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 \
  --set "target_speed_range=(10.0,10.0)" \
  --cells 5x5,10x10,15x10 --episodes 48 --strategies direct,saturation,decoy \
  --tag model_convoy_moving > experiments/night2_20260828/C8_moving.log 2>&1
timeout 3600 python -m eval.eval_nvm_grid --model $CV --realism \
  --set kamikaze=False --cells 5x5,10x10,20x20 --episodes 32 \
  --strategies direct,saturation,decoy --tag model_convoy_static \
  > experiments/night2_20260828/C8_static.log 2>&1
timeout 3600 python -m eval.eval_nvm_grid --model $LP --realism \
  --set kamikaze=False --set "target_speed_range=(10.0,10.0)" \
  --cells 5x5,10x10,15x10 --episodes 48 --strategies direct,saturation,decoy \
  --tag model_largeP_moving > experiments/night2_20260828/C8_control.log 2>&1
fi
step "END(exit=$?)" C8_convoy_gates

step START C9_gifs
BEST=$LP; [ -f "$CV" ] && BEST=$CV
timeout 900 python -m eval.render_aegis --model $BEST --nm 8x10 --strategy saturation \
  --terrain hills --target-speed 10 --stride 1 --fps 14 --seed 31 \
  --out models/eval_output/aegis_convoy_hills_8v10.gif > /dev/null 2>&1
timeout 900 python -m eval.render_aegis --model $BEST --nm 10x12 --strategy decoy \
  --terrain urban --target-speed 8 --stride 1 --fps 14 --seed 32 \
  --out models/eval_output/aegis_convoy_urban_10v12.gif > /dev/null 2>&1
step "END(exit=$?)" C9_gifs

step START C10_dashboard
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" C10_dashboard
step C_DONE queue
