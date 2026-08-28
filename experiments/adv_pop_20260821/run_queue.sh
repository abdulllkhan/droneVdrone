#!/bin/bash
# Adversarial population loop r2/r3 + reuse-defender track — 2026-08-21
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/adv_pop_20260821/queue.log
A1=models/aegis_adv_attacker_best/best_model.zip
D1=models/aegis_adv_defender_best/best_model.zip
A2=models/aegis_adv_attacker_r2_best/best_model.zip
D2=models/aegis_adv_defender_r2_best/best_model.zip
A3=models/aegis_adv_attacker_r3_best/best_model.zip
D3=models/aegis_adv_defender_r3_best/best_model.zip
REUSE=models/aegis_reuse_stage3_realism_best/best_model.zip
AR=models/aegis_adv_attacker_reuse_best/best_model.zip
DR=models/aegis_adv_defender_reuse_best/best_model.zip
CELLS=3x5,5x3,5x5,8x5,8x8
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START round2
timeout 14400 python -m train.train_aegis_adv --defender $D1 --pool $A1 --round-tag r2 \
  > experiments/adv_pop_20260821/r2.log 2>&1
step "END(exit=$?)" round2

step START round3
if [ -f "$D2" ]; then
timeout 14400 python -m train.train_aegis_adv --defender $D2 --pool $A1,$A2 --round-tag r3 \
  > experiments/adv_pop_20260821/r3.log 2>&1
fi
step "END(exit=$?)" round3

step START reuse_round
timeout 14400 python -m train.train_aegis_adv --defender $REUSE --kamikaze-false --round-tag reuse \
  > experiments/adv_pop_20260821/reuse.log 2>&1
step "END(exit=$?)" reuse_round

step START eval_b3_vs_gens
if [ -f "$D3" ]; then
for K in 1 2 3; do
  AK=$(eval echo \$A$K)
  timeout 3600 python -m eval.eval_nvm_grid --model $D3 --realism --strategies rl \
    --set attacker_model_path=$AK --cells $CELLS --episodes 48 --tag model_b3_vs_a$K \
    > experiments/adv_pop_20260821/eval_b3_vs_a$K.log 2>&1
done
fi
step "END(exit=$?)" eval_b3_vs_gens

step START eval_b3_scripted
if [ -f "$D3" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $D3 --realism \
  --strategies direct,flanking,jink --episodes 32 --tag model_b3_scripted \
  > experiments/adv_pop_20260821/eval_b3_scripted.log 2>&1
fi
step "END(exit=$?)" eval_b3_scripted

step START eval_reuse_adv
if [ -f "$DR" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $DR --realism --strategies rl \
  --set kamikaze=False --set attacker_model_path=$AR --cells $CELLS --episodes 48 \
  --tag model_reuseadv_vs_ar > experiments/adv_pop_20260821/eval_reuse_vs_ar.log 2>&1
timeout 3600 python -m eval.eval_nvm_grid --model $DR --set kamikaze=False \
  --cells 3x5,5x8,10x20 --episodes 48 --strategies direct,flanking --tag model_reuseadv_nlm \
  > experiments/adv_pop_20260821/eval_reuse_nlm.log 2>&1
fi
step "END(exit=$?)" eval_reuse_adv

step ALL_DONE queue
