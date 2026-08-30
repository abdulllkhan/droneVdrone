#!/bin/bash
# Definitive final scorecard: rigorous n=100 eval of the champions across the full
# capability surface, compiled into an authoritative CAPABILITIES report.
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/scorecard_20260829/queue.log
SPLIT=models/aegis_split_stage3_realism_best/best_model.zip     # superset champion
CFADV=models/aegis_adv_defender_cf_lab_best/best_model.zip      # CrazyFlie deploy model
KAM='--set kamikaze=False'
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START SC_single
timeout 5400 python -m eval.eval_nvm_grid --model $SPLIT --realism $KAM \
  --cells 1x1,3x3,5x5,7x5,10x10,15x15,20x20 --episodes 100 --strategies direct,saturation \
  --tag sc_single > experiments/scorecard_20260829/single.log 2>&1
step "END(exit=$?)" SC_single

step START SC_scale
timeout 5400 python -m eval.eval_nvm_grid --model $SPLIT $KAM \
  --cells 30x30,40x40,50x40,20x40,10x20 --episodes 60 --strategies direct,flanking \
  --tag sc_scale > experiments/scorecard_20260829/scale.log 2>&1
step "END(exit=$?)" SC_scale

step START SC_split
timeout 5400 python -m eval.eval_nvm_grid --model $SPLIT --realism $KAM \
  --set num_targets=3 --set target_spread=180.0 \
  --cells 10x10,15x15,20x20 --episodes 100 --strategies direct,saturation \
  --tag sc_split3 > experiments/scorecard_20260829/split.log 2>&1
step "END(exit=$?)" SC_split

step START SC_cflab
timeout 5400 python -m eval.eval_nvm_grid --model $CFADV --realism $KAM \
  --set obs_opportunistic=False --set obs_zero_pip=True \
  --cells 1x1,2x2,3x3,2x1,3x2,1x2 --episodes 100 --strategies direct,jink \
  --tag sc_cflab > experiments/scorecard_20260829/cflab.log 2>&1
step "END(exit=$?)" SC_cflab

step START SC_dash
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" SC_dash
step SCORECARD_DONE queue
