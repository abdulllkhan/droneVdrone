#!/bin/bash
# Overnight AEGIS experiment queue — 2026-08-19 (see docs/REWARDS.md for rationale)
# Sequential; each step logs START/END to queue.log. Monitor cron reads that log.
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/overnight_20260819/queue.log
BEST=models/aegis_stage3_realism_best/best_model.zip
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START q1_scale_model
timeout 3600 python -m eval.eval_nvm_grid --model $BEST --cells 20x15,30x20,10x20,40x30 \
  --episodes 24 --strategies direct,flanking --tag model_scale \
  > experiments/overnight_20260819/q1_model.log 2>&1
step "END(exit=$?)" q1_scale_model

step START q1_scale_baseline
timeout 3600 python -m eval.eval_nvm_grid --baseline --cells 20x15,30x20,10x20,40x30 \
  --episodes 24 --strategies direct,flanking --tag baseline_scale \
  > experiments/overnight_20260819/q1_baseline.log 2>&1
step "END(exit=$?)" q1_scale_baseline

step START q2_patch_train
timeout 7200 python -m train.train_aegis --tag patch \
  --set r_bad_death=-30.0 --set r_breach_unowned=-5.0 \
  --set r_safety_coeff=-0.05 --set obstacle_danger_radius=90.0 \
  > experiments/overnight_20260819/q2_train.log 2>&1
step "END(exit=$?)" q2_patch_train

step START q2_patch_eval
timeout 3600 python -m eval.eval_nvm_grid --model models/aegis_patch_stage3_realism_best/best_model.zip \
  --realism --strategies direct,flanking,jink --episodes 32 --tag model_patch_realism \
  > experiments/overnight_20260819/q2_eval.log 2>&1
step "END(exit=$?)" q2_patch_eval

step START q3_reuse_train
timeout 7200 python -m train.train_aegis --tag reuse \
  --set kamikaze=False --set r_bad_death=-30.0 --set r_breach_unowned=-5.0 \
  --set r_safety_coeff=-0.05 --set obstacle_danger_radius=90.0 \
  > experiments/overnight_20260819/q3_train.log 2>&1
step "END(exit=$?)" q3_reuse_train

step START q3_reuse_eval
timeout 3600 python -m eval.eval_nvm_grid --model models/aegis_reuse_stage3_realism_best/best_model.zip \
  --set kamikaze=False --realism --strategies direct,flanking,jink --episodes 32 \
  --tag model_reuse_realism > experiments/overnight_20260819/q3_eval_realism.log 2>&1
step "END(exit=$?)" q3_reuse_eval

step START q3_reuse_nlm
timeout 3600 python -m eval.eval_nvm_grid --model models/aegis_reuse_stage3_realism_best/best_model.zip \
  --set kamikaze=False --cells 3x5,3x8,5x8,7x10,10x20 --episodes 48 \
  --strategies direct,flanking --tag model_reuse_nlm \
  > experiments/overnight_20260819/q3_eval_nlm.log 2>&1
step "END(exit=$?)" q3_reuse_nlm

step START q4_kill2
timeout 3600 python -m eval.eval_nvm_grid --model $BEST --realism --culled \
  --set cull_count=2 --set "cull_step_range=(120,121)" --cells 5x3,5x5,8x5 \
  --episodes 48 --strategies direct,flanking --tag model_kill2 \
  > experiments/overnight_20260819/q4_kill2.log 2>&1
step "END(exit=$?)" q4_kill2

step START q4_sensor150
timeout 3600 python -m eval.eval_nvm_grid --model $BEST --set sensor_range=150.0 \
  --cells 5x3,5x5,8x8 --episodes 48 --strategies direct,flanking --tag model_sensor150 \
  > experiments/overnight_20260819/q4_sensor150.log 2>&1
step "END(exit=$?)" q4_sensor150

step START q4_sensor75
timeout 3600 python -m eval.eval_nvm_grid --model $BEST --set sensor_range=75.0 \
  --cells 5x3,5x5,8x8 --episodes 48 --strategies direct,flanking --tag model_sensor75 \
  > experiments/overnight_20260819/q4_sensor75.log 2>&1
step "END(exit=$?)" q4_sensor75

step START q4_speed12
timeout 3600 python -m eval.eval_nvm_grid --model $BEST --set attacker_speed_scale=1.2 \
  --cells 3x3,5x3,5x5 --episodes 48 --strategies direct,flanking --tag model_speed12 \
  > experiments/overnight_20260819/q4_speed12.log 2>&1
step "END(exit=$?)" q4_speed12

step START q4_stale24
timeout 3600 python -m eval.eval_nvm_grid --model $BEST --set realloc_every=24 \
  --cells 5x5,8x5 --episodes 48 --strategies direct,flanking --culled --tag model_stale24 \
  > experiments/overnight_20260819/q4_stale24.log 2>&1
step "END(exit=$?)" q4_stale24

step START q5_scratch3_train
timeout 7200 python -m train.train_aegis --tag scratch3 --start-stage 3 --skip-gates \
  > experiments/overnight_20260819/q5_train.log 2>&1
step "END(exit=$?)" q5_scratch3_train

step START q5_scratch3_eval
timeout 3600 python -m eval.eval_nvm_grid --model models/aegis_scratch3_stage3_realism_best/best_model.zip \
  --realism --strategies direct,flanking,jink --episodes 32 --tag model_scratch3_realism \
  > experiments/overnight_20260819/q5_eval.log 2>&1
step "END(exit=$?)" q5_scratch3_eval

step ALL_DONE queue
