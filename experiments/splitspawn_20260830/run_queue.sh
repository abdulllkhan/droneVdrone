#!/bin/bash
# Close the STRUCTURAL split-vs-direct residual: distributed defender spawns
# (pre-position across the 3 targets). Fine-tune aegis_splitdirect with the new
# spawn geometry, direct-weighted. Target: direct K=3 cells 83-85% -> >=90%.
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/splitspawn_20260830/queue.log
WARM=models/aegis_splitdirect_stage3_realism_best/best_model.zip
SS=models/aegis_splitspawn_stage3_realism_best/best_model.zip
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START SS_train
timeout 14400 python -m train.train_aegis --start-stage 3 --resume $WARM --tag splitspawn \
  --num-worlds-scale 0.5 --timesteps-scale 0.7 \
  --set n_max=20 --set m_max=20 --set "n_range=(6,20)" --set "m_range=(3,20)" \
  --set "terrains=('hills','none')" --set num_obstacles=5 --set kamikaze=False \
  --set "attacker_strategies=('direct','direct','direct','flanking','saturation','waves')" \
  --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 \
  --set num_targets=3 --set target_spread=180.0 \
  > experiments/splitspawn_20260830/train.log 2>&1
step "END(exit=$?)" SS_train

step START SS_eval
if [ -f "$SS" ]; then
for MODEL in $SS $WARM; do
  T=$([ "$MODEL" = "$SS" ] && echo splitspawn || echo splitdirect)
  timeout 3600 python -m eval.eval_nvm_grid --model $MODEL --realism --set kamikaze=False \
    --set num_targets=3 --set target_spread=180.0 \
    --cells 10x10,15x15,20x20 --episodes 100 --strategies direct,saturation \
    --tag ${T}_k3b > experiments/splitspawn_20260830/eval_${T}.log 2>&1
done
timeout 3600 python -m eval.eval_nvm_grid --model $SS --realism --set kamikaze=False \
  --cells 10x10,20x20 --episodes 60 --strategies direct,saturation \
  --tag splitspawn_single > experiments/splitspawn_20260830/eval_single.log 2>&1
fi
timeout 1200 python -m eval.render_aegis --model ${SS:-$WARM} --nm 15x15 --strategy direct \
  --terrain hills --num-targets 3 --stride 1 --fps 14 --seed 91 \
  --out models/eval_output/aegis_splitspawn_15v15.gif > /dev/null 2>&1
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" SS_eval
step SPLITSPAWN_DONE queue
