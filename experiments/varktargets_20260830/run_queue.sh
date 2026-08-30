#!/bin/bash
# Variable-target-count generalist: ONE policy that defends 1-4 protected points.
# Fine-tune aegis_splitspawn with per-world K~U[1,4]. Eval at each fixed K.
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/varktargets_20260830/queue.log
WARM=models/aegis_splitspawn_stage3_realism_best/best_model.zip
VK=models/aegis_vark_stage3_realism_best/best_model.zip
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START VK_train
timeout 14400 python -m train.train_aegis --start-stage 3 --resume $WARM --tag vark \
  --num-worlds-scale 0.5 --timesteps-scale 0.8 \
  --set n_max=20 --set m_max=20 --set "n_range=(6,20)" --set "m_range=(3,20)" \
  --set "terrains=('hills','none')" --set num_obstacles=5 --set kamikaze=False \
  --set "attacker_strategies=('direct','direct','flanking','saturation','waves')" \
  --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 \
  --set "target_count_range=(1,4)" --set target_spread=180.0 \
  > experiments/varktargets_20260830/train.log 2>&1
step "END(exit=$?)" VK_train

step START VK_eval
if [ -f "$VK" ]; then
# the vark generalist evaluated at each fixed K (1,2,3,4)
for K in 1 2 3 4; do
  timeout 3600 python -m eval.eval_nvm_grid --model $VK --realism --set kamikaze=False \
    --set num_targets=$K --set target_spread=180.0 \
    --cells 10x10,15x15,20x20 --episodes 60 --strategies direct,saturation \
    --tag vark_k$K > experiments/varktargets_20260830/eval_k$K.log 2>&1
done
fi
timeout 1200 python -m eval.render_aegis --model ${VK:-$WARM} --nm 16x16 --strategy direct \
  --terrain hills --num-targets 4 --stride 1 --fps 14 --seed 44 \
  --out models/eval_output/aegis_vark_4targets_16x16.gif > /dev/null 2>&1
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" VK_eval
step VARK_DONE queue
