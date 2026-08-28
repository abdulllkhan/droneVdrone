#!/bin/bash
# Overnight 2026-08-27: large-swarm terrain-generalist + improved attackers.
# Fully self-driving: training -> adversarial round -> evals -> GIF pack.
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/overnight_20260827/queue.log
WARM=models/aegis_large_stage3_realism_best/best_model.zip
LARGE=models/aegis_large_stage3_realism_best/best_model.zip
ALARGE=models/aegis_adv_attacker_large_best/best_model.zip
DLARGE=models/aegis_adv_defender_large_best/best_model.zip
TSET='--set terrains=("hills","urban","maze") --set num_obstacles=5 --set kamikaze=False'
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START q1_train_large
timeout 21600 python -m train.train_aegis --start-stage 3 --resume $WARM --tag large --num-worlds-scale 0.75 \
  --timesteps-scale 1.5 \
  --set n_max=20 --set m_max=20 --set "n_range=(4,20)" --set "m_range=(1,20)" \
  --set "terrains=('hills','urban','maze')" --set num_obstacles=5 --set kamikaze=False \
  --set "attacker_strategies=('direct','flanking','jink','saturation','waves','decoy')" \
  > experiments/overnight_20260827/q1_train.log 2>&1
step "END(exit=$?)" q1_train_large

step START q2_adv_large
if [ -f "$LARGE" ]; then
timeout 21600 python -m train.train_aegis_adv --defender $LARGE --kamikaze-false \
  --round-tag large --num-worlds 320 \
  --env-set n_max=12 --env-set m_max=12 --env-set "n_range=(4,12)" --env-set "m_range=(2,12)" \
  --env-set "terrains=('hills','urban','maze')" \
  > experiments/overnight_20260827/q2_adv.log 2>&1
fi
step "END(exit=$?)" q2_adv_large

step START q3_eval_terrain
if [ -f "$LARGE" ]; then
for T in hills urban maze; do
  timeout 3600 python -m eval.eval_nvm_grid --model $LARGE --realism \
    --set kamikaze=False --set terrain=$T \
    --cells 5x5,10x10,15x10,20x20 --episodes 32 \
    --strategies direct,saturation,decoy --tag model_large_$T \
    > experiments/overnight_20260827/q3_$T.log 2>&1
done
timeout 3600 python -m eval.eval_nvm_grid --model $LARGE --set kamikaze=False \
  --cells 40x30,50x40,10x20,20x30 --episodes 24 --strategies direct,flanking \
  --tag model_large_scale > experiments/overnight_20260827/q3_scale.log 2>&1
fi
step "END(exit=$?)" q3_eval_terrain

step START q4_eval_advlarge
if [ -f "$DLARGE" ]; then
timeout 3600 python -m eval.eval_nvm_grid --model $DLARGE --realism --strategies rl \
  --set kamikaze=False --set attacker_model_path=$ALARGE \
  --cells 5x5,8x8,10x12 --episodes 48 --tag model_advlarge_vs_rl \
  > experiments/overnight_20260827/q4_vs_rl.log 2>&1
timeout 3600 python -m eval.eval_nvm_grid --model $DLARGE --realism --set kamikaze=False \
  --set terrain=maze --cells 5x5,10x10,20x20 --episodes 32 \
  --strategies direct,saturation,decoy --tag model_advlarge_maze \
  > experiments/overnight_20260827/q4_maze.log 2>&1
fi
step "END(exit=$?)" q4_eval_advlarge

step START q5_gifs
BEST=$LARGE; [ -f "$DLARGE" ] && BEST=$DLARGE
if [ -f "$BEST" ]; then
timeout 900 python -m eval.render_aegis --model $BEST --nm 8x8 --strategy saturation \
  --terrain maze --stride 1 --fps 14 --seed 11 > /dev/null 2>&1
timeout 900 python -m eval.render_aegis --model $BEST --nm 10x15 --strategy decoy \
  --terrain urban --stride 1 --fps 14 --seed 12 > /dev/null 2>&1
timeout 900 python -m eval.render_aegis --model $BEST --nm 20x20 --strategy waves \
  --terrain hills --stride 2 --fps 16 --seed 13 > /dev/null 2>&1
timeout 1200 python -m eval.render_aegis --model $BEST --nm 50x40 --strategy flanking \
  --terrain none --plain --stride 2 --fps 16 --seed 14 > /dev/null 2>&1
fi
step "END(exit=$?)" q5_gifs

step START q6_dashboard
timeout 900 python -m eval.dashboard > experiments/overnight_20260827/q6_dash.log 2>&1
step "END(exit=$?)" q6_dashboard

step ALL_DONE queue
