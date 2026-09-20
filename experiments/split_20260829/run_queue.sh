#!/bin/bash
# Multi-target split-defense: fine-tune big24 to protect 3 separate points at once.
# 256 worlds (Gemma probe may share the box).
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/split_20260829/queue.log
WARM=models/aegis_big24_stage3_realism_best/best_model.zip
SPLIT=models/aegis_split_stage3_realism_best/best_model.zip
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START S_train_split
timeout 14400 python -m train.train_aegis --start-stage 3 --resume $WARM --tag split \
  --num-worlds-scale 0.5 --timesteps-scale 0.7 \
  --set n_max=20 --set m_max=20 --set "n_range=(6,20)" --set "m_range=(3,20)" \
  --set "terrains=('hills','none')" --set num_obstacles=5 --set kamikaze=False \
  --set "attacker_strategies=('direct','flanking','saturation','waves')" \
  --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 \
  --set num_targets=3 --set target_spread=180.0 \
  > experiments/split_20260829/train.log 2>&1
step "END(exit=$?)" S_train_split

step START S_eval
if [ -f "$SPLIT" ]; then
# split-defense model at K=3 vs the big24 baseline (also at K=3, zero-shot)
for MODEL in $SPLIT $WARM; do
  T=$([ "$MODEL" = "$SPLIT" ] && echo split || echo big24)
  timeout 3600 python -m eval.eval_nvm_grid --model $MODEL --realism \
    --set kamikaze=False --set num_targets=3 --set target_spread=180.0 \
    --cells 10x10,15x15,20x20 --episodes 40 --strategies direct,saturation \
    --tag ${T}_split3 > experiments/split_20260829/eval_${T}.log 2>&1
done
# regression: split model still perfect at single-target?
timeout 3600 python -m eval.eval_nvm_grid --model $SPLIT --realism --set kamikaze=False \
  --cells 10x10,20x20 --episodes 32 --strategies direct,saturation \
  --tag split_single > experiments/split_20260829/eval_single.log 2>&1
fi
timeout 1200 python -m eval.render_aegis --model ${SPLIT:-$WARM} --nm 15x15 --strategy saturation \
  --terrain hills --num-targets 3 --stride 1 --fps 14 --seed 71 \
  --out models/eval_output/aegis_split3_15v15.gif > /dev/null 2>&1
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" S_eval
step SPLIT_DONE queue
