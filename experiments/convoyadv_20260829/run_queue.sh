#!/bin/bash
# Convoy adversarial verification: can a LEARNED attacker exploit the moving-target
# escort, and does co-training harden it? Ukraine scale, 192 worlds (OOM-safe).
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/convoyadv_20260829/queue.log
CV=models/aegis_convoy_stage3_realism_best/best_model.zip
ADV=models/aegis_adv_defender_convoyadv_best/best_model.zip
ATT=models/aegis_adv_attacker_convoyadv_best/best_model.zip
MSET='--set kamikaze=False --set "target_speed_range=(10.0,10.0)"'
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START A_convoyadv_cotrain
timeout 14400 python -m train.train_aegis_adv --defender $CV --kamikaze-false \
  --round-tag convoyadv --num-worlds 192 \
  --env-set "target_speed_range=(0.0,10.0)" \
  --env-set r_bad_death=-60.0 --env-set r_safety_coeff=-0.10 \
  > experiments/convoyadv_20260829/cotrain.log 2>&1
step "END(exit=$?)" A_convoyadv_cotrain

step START B_evals
# frozen convoy defender vs the learned attacker (on MOVING targets)
timeout 3600 python -m eval.eval_nvm_grid --model $CV --realism --strategies rl \
  --set kamikaze=False --set "target_speed_range=(10.0,10.0)" \
  --set attacker_model_path=$ATT --cells 5x5,8x8,10x12 --episodes 48 \
  --tag convoy_vs_rl_moving > experiments/convoyadv_20260829/eval_cv_vs_rl.log 2>&1
if [ -f "$ADV" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $ADV --realism --strategies rl \
  --set kamikaze=False --set "target_speed_range=(10.0,10.0)" \
  --set attacker_model_path=$ATT --cells 5x5,8x8,10x12 --episodes 48 \
  --tag convoyadv_vs_rl_moving > experiments/convoyadv_20260829/eval_adv_vs_rl.log 2>&1
# regression: hardened defender on scripted moving + static
timeout 3600 python -m eval.eval_nvm_grid --model $ADV --realism \
  --set kamikaze=False --set "target_speed_range=(10.0,10.0)" \
  --cells 5x5,10x10,15x10 --episodes 32 --strategies direct,saturation,decoy \
  --tag convoyadv_scripted_moving > experiments/convoyadv_20260829/eval_scripted.log 2>&1
timeout 3600 python -m eval.eval_nvm_grid --model $ADV --realism \
  --set kamikaze=False --cells 5x5,10x10,20x20 --episodes 32 \
  --strategies direct,saturation --tag convoyadv_static \
  > experiments/convoyadv_20260829/eval_static.log 2>&1
fi
step "END(exit=$?)" B_evals

step START C_gif_dash
BEST=$CV; [ -f "$ADV" ] && BEST=$ADV
timeout 900 python -m eval.render_aegis --model $BEST --nm 8x10 --strategy saturation \
  --terrain hills --target-speed 10 --stride 1 --fps 14 --seed 41 \
  --out models/eval_output/aegis_convoyadv_hills_8v10.gif > /dev/null 2>&1
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" C_gif_dash
step CONVOYADV_DONE queue
