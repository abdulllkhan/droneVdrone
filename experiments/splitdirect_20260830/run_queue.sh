#!/bin/bash
# Close the split-vs-direct gap: fine-tune aegis_split with DIRECT simultaneous
# raids weighted heavily at K=3, so defenders learn to pre-position across the 3
# targets instead of reacting. Target: lift the 74-81% direct cells toward >=90%.
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/splitdirect_20260830/queue.log
WARM=models/aegis_split_stage3_realism_best/best_model.zip
SD=models/aegis_splitdirect_stage3_realism_best/best_model.zip
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START SD_train
timeout 14400 python -m train.train_aegis --start-stage 3 --resume $WARM --tag splitdirect \
  --num-worlds-scale 0.5 --timesteps-scale 0.7 \
  --set n_max=20 --set m_max=20 --set "n_range=(6,20)" --set "m_range=(3,20)" \
  --set "terrains=('hills','none')" --set num_obstacles=5 --set kamikaze=False \
  --set "attacker_strategies=('direct','direct','direct','flanking','saturation','waves')" \
  --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 \
  --set num_targets=3 --set target_spread=180.0 \
  > experiments/splitdirect_20260830/train.log 2>&1
step "END(exit=$?)" SD_train

step START SD_eval
if [ -f "$SD" ]; then
# the KEY comparison: direct K=3 cells, new vs old split model
for MODEL in $SD $WARM; do
  T=$([ "$MODEL" = "$SD" ] && echo splitdirect || echo split)
  timeout 3600 python -m eval.eval_nvm_grid --model $MODEL --realism --set kamikaze=False \
    --set num_targets=3 --set target_spread=180.0 \
    --cells 10x10,15x15,20x20 --episodes 100 --strategies direct,saturation \
    --tag ${T}_k3direct > experiments/splitdirect_20260830/eval_${T}.log 2>&1
done
# regression: new model still perfect at single-target?
timeout 3600 python -m eval.eval_nvm_grid --model $SD --realism --set kamikaze=False \
  --cells 10x10,20x20 --episodes 60 --strategies direct,saturation \
  --tag splitdirect_single > experiments/splitdirect_20260830/eval_single.log 2>&1
fi
timeout 1200 python -m eval.render_aegis --model ${SD:-$WARM} --nm 15x15 --strategy direct \
  --terrain hills --num-targets 3 --stride 1 --fps 14 --seed 81 \
  --out models/eval_output/aegis_splitdirect_15v15.gif > /dev/null 2>&1
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" SD_eval
step SPLITDIRECT_DONE queue
