#!/bin/bash
# First adversarial test of MULTI-TARGET defense: do learned attackers find a
# seam in split defense that scripted raids don't? Adversarial round vs frozen
# aegis_vark at K=3. Box is free (Gemma done) -> full 512 worlds.
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/varkadv_20260831/queue.log
VK=models/aegis_vark_stage3_realism_best/best_model.zip
ADV=models/aegis_adv_defender_varkadv_best/best_model.zip
ATT=models/aegis_adv_attacker_varkadv_best/best_model.zip
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START A_varkadv_cotrain
timeout 14400 python -m train.train_aegis_adv --defender $VK --kamikaze-false \
  --round-tag varkadv --num-worlds 256 \
  --env-set "target_count_range=(3,3)" --env-set target_spread=180.0 \
  --env-set r_bad_death=-60.0 --env-set r_safety_coeff=-0.10 \
  > experiments/varkadv_20260831/cotrain.log 2>&1
step "END(exit=$?)" A_varkadv_cotrain

step START B_evals
# frozen vark vs the learned multi-target attacker, K=3
timeout 3600 python -m eval.eval_nvm_grid --model $VK --realism --strategies rl \
  --set kamikaze=False --set num_targets=3 --set target_spread=180.0 \
  --set attacker_model_path=$ATT --cells 8x8,15x15 --episodes 60 \
  --tag vark_vs_rl_k3 > experiments/varkadv_20260831/eval_vark_vs_rl.log 2>&1
if [ -f "$ADV" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $ADV --realism --strategies rl \
  --set kamikaze=False --set num_targets=3 --set target_spread=180.0 \
  --set attacker_model_path=$ATT --cells 8x8,15x15 --episodes 60 \
  --tag varkadv_vs_rl_k3 > experiments/varkadv_20260831/eval_adv_vs_rl.log 2>&1
timeout 3600 python -m eval.eval_nvm_grid --model $ADV --realism \
  --set kamikaze=False --set num_targets=3 --set target_spread=180.0 \
  --cells 8x8,15x15 --episodes 60 --strategies direct,saturation \
  --tag varkadv_scripted_k3 > experiments/varkadv_20260831/eval_scripted.log 2>&1
fi
step "END(exit=$?)" B_evals

step START C_dash
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" C_dash
step VARKADV_DONE queue
