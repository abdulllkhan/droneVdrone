#!/bin/bash
# Close the verification loop: does multi-target champion aegis_vark hold up under
# realistic-deployment stressors (only ever tested on single-target models)?
# Sensor degradation, speed parity, kill injection — at K=1 and K=3.
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/varkstress_20260830/queue.log
VK=models/aegis_vark_stage3_realism_best/best_model.zip
E="python -m eval.eval_nvm_grid --model $VK --realism --set kamikaze=False --set target_spread=180.0 --cells 8x8,15x15 --episodes 48 --strategies direct,saturation"
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START stress_sweep
for K in 1 3; do
  timeout 3000 $E --set num_targets=$K --tag vark_base_k$K \
    > experiments/varkstress_20260830/vark_base_k$K.log 2>&1
  timeout 3000 $E --set num_targets=$K --set sensor_range=150.0 --tag vark_sensor150_k$K \
    > experiments/varkstress_20260830/vark_sensor150_k$K.log 2>&1
  timeout 3000 $E --set num_targets=$K --set sensor_range=75.0 --tag vark_sensor75_k$K \
    > experiments/varkstress_20260830/vark_sensor75_k$K.log 2>&1
  timeout 3000 $E --set num_targets=$K --set attacker_speed_scale=1.2 --tag vark_speed12_k$K \
    > experiments/varkstress_20260830/vark_speed12_k$K.log 2>&1
  timeout 3000 $E --set num_targets=$K --culled --set cull_count=2 --tag vark_kill2_k$K \
    > experiments/varkstress_20260830/vark_kill2_k$K.log 2>&1
done
step "END(exit=$?)" stress_sweep

step START dash
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" dash
step VARKSTRESS_DONE queue
