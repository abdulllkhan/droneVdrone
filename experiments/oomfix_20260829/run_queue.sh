#!/bin/bash
# Validate the OOM fix over a REAL training run at 512 worlds n_max=20 (the config
# that exit-137'd 4x). Produces a bonus full-worlds terrain champion.
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/oomfix_20260829/queue.log
LP=models/aegis_largeP_stage3_realism_best/best_model.zip
V=models/aegis_oomval_stage3_realism_best/best_model.zip
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START V_train_512
timeout 10800 python -m train.train_aegis --start-stage 3 --resume $LP --tag oomval \
  --timesteps-scale 0.5 \
  --set n_max=20 --set m_max=20 --set "n_range=(4,20)" --set "m_range=(1,20)" \
  --set "terrains=('hills','urban','maze')" --set num_obstacles=5 --set kamikaze=False \
  --set "attacker_strategies=('direct','flanking','jink','saturation','waves','decoy')" \
  --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 \
  > experiments/oomfix_20260829/train.log 2>&1
step "END(exit=$?)" V_train_512

step START V_eval
if [ -f "$V" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $V --realism --set kamikaze=False \
  --set terrain=maze --cells 10x10,20x20 --episodes 32 --strategies direct,saturation \
  --tag oomval_maze > experiments/oomfix_20260829/eval.log 2>&1
timeout 3600 python -m eval.eval_nvm_grid --model $V --set kamikaze=False \
  --cells 40x30,50x40 --episodes 24 --strategies direct,flanking --tag oomval_scale \
  > experiments/oomfix_20260829/eval_scale.log 2>&1
fi
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" V_eval
step OOMVAL_DONE queue
