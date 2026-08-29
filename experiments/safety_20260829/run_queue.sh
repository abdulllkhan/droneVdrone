#!/bin/bash
# Attrition A/B: does stronger safety shaping cut terrain deaths (16% @ 50v40)
# without win regression? Warm from big24, retrain w/ r_safety_coeff -0.15 +
# obstacle_danger_radius 110 (baseline was -0.10 / 90).
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/safety_20260829/queue.log
WARM=models/aegis_big24_stage3_realism_best/best_model.zip
SAFE=models/aegis_safe_stage3_realism_best/best_model.zip
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START S_train_safe
timeout 14400 python -m train.train_aegis --start-stage 3 --resume $WARM --tag safe \
  --num-worlds-scale 0.875 --timesteps-scale 0.6 \
  --set n_max=20 --set m_max=20 --set "n_range=(4,20)" --set "m_range=(1,20)" \
  --set "terrains=('hills','urban','maze')" --set num_obstacles=5 --set kamikaze=False \
  --set "attacker_strategies=('direct','flanking','jink','saturation','waves','decoy')" \
  --set r_bad_death=-60.0 --set r_safety_coeff=-0.15 --set obstacle_danger_radius=110.0 \
  > experiments/safety_20260829/train.log 2>&1
step "END(exit=$?)" S_train_safe

step START S_eval
# safe arm (must eval WITH its own danger radius so terrain geometry matches training)
for M in $SAFE $WARM; do
  T=$([ "$M" = "$SAFE" ] && echo safe || echo big24)
  RAD=$([ "$M" = "$SAFE" ] && echo 110.0 || echo 90.0)
  SC=$([ "$M" = "$SAFE" ] && echo -0.15 || echo -0.10)
  timeout 3600 python -m eval.eval_nvm_grid --model $M --realism --set kamikaze=False \
    --set r_safety_coeff=$SC --set obstacle_danger_radius=$RAD \
    --cells 10x10,20x20 --episodes 32 --strategies saturation,decoy \
    --tag ${T}_dense > experiments/safety_20260829/eval_${T}_dense.log 2>&1
  timeout 3600 python -m eval.eval_nvm_grid --model $M --set kamikaze=False \
    --set r_safety_coeff=$SC --set obstacle_danger_radius=$RAD \
    --cells 40x40,50x40 --episodes 20 --strategies direct,flanking \
    --tag ${T}_scale > experiments/safety_20260829/eval_${T}_scale.log 2>&1
done
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" S_eval
step SAFETY_DONE queue
