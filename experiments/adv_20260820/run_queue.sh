#!/bin/bash
# Phase-3 adversarial co-training queue — 2026-08-20
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/adv_20260820/queue.log
PATCH=models/aegis_patch_stage3_realism_best/best_model.zip
ATT=models/aegis_adv_attacker_best/best_model.zip
ADVDEF=models/aegis_adv_defender_best/best_model.zip
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START adv_cotrain
timeout 14400 python -m train.train_aegis_adv --defender $PATCH \
  > experiments/adv_20260820/cotrain.log 2>&1
step "END(exit=$?)" adv_cotrain

step START eval_patch_vs_rl
if [ -f "$ATT" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $PATCH --realism \
  --strategies rl --set attacker_model_path=$ATT \
  --cells 3x5,5x3,5x5,8x5,8x8 --episodes 48 --tag model_patch_vs_rl \
  > experiments/adv_20260820/eval_patch_vs_rl.log 2>&1
fi
step "END(exit=$?)" eval_patch_vs_rl

step START eval_advdef_vs_rl
if [ -f "$ADVDEF" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $ADVDEF --realism \
  --strategies rl --set attacker_model_path=$ATT \
  --cells 3x5,5x3,5x5,8x5,8x8 --episodes 48 --tag model_advdef_vs_rl \
  > experiments/adv_20260820/eval_advdef_vs_rl.log 2>&1
fi
step "END(exit=$?)" eval_advdef_vs_rl

step START eval_advdef_scripted
if [ -f "$ADVDEF" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $ADVDEF --realism \
  --strategies direct,flanking,jink --episodes 32 --tag model_advdef_scripted \
  > experiments/adv_20260820/eval_advdef_scripted.log 2>&1
fi
step "END(exit=$?)" eval_advdef_scripted

step ALL_DONE queue
