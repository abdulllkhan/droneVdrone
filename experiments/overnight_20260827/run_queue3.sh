#!/bin/bash
# Round 2 (2026-08-27): attrition shaping A/B + routed-maze evals + sensor knockouts
cd /home/abdulllkhan/Documents/droneVdrone
Q=experiments/overnight_20260827/queue.log
LARGE=models/aegis_large_stage3_realism_best/best_model.zip
LARGEP=models/aegis_largeP_stage3_realism_best/best_model.zip
step() { echo "=== $(date '+%F %T') $1 $2" >> "$Q"; }

step START q7_train_largeP
timeout 21600 python -m train.train_aegis --start-stage 3 --resume $LARGE --tag largeP \
  --num-worlds-scale 0.75 \
  --set n_max=20 --set m_max=20 --set "n_range=(4,20)" --set "m_range=(1,20)" \
  --set "terrains=('hills','urban','maze')" --set num_obstacles=5 --set kamikaze=False \
  --set "attacker_strategies=('direct','flanking','jink','saturation','waves','decoy')" \
  --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 \
  > experiments/overnight_20260827/q7_train.log 2>&1
step "END(exit=$?)" q7_train_largeP

step START q8_eval_largeP
if [ -f "$LARGEP" ]; then
for T in maze urban; do
  timeout 3600 python -m eval.eval_nvm_grid --model $LARGEP --realism \
    --set kamikaze=False --set r_bad_death=-60.0 --set r_safety_coeff=-0.10 --set terrain=$T \
    --cells 5x5,10x10,20x20 --episodes 32 --strategies direct,saturation,decoy \
    --tag model_largeP_$T > experiments/overnight_20260827/q8_$T.log 2>&1
done
fi
step "END(exit=$?)" q8_eval_largeP

step START q9_knockouts
for K in "" pip opportunistic teammates assignmeta; do
  TAGK=${K:-none}
  timeout 3600 python -m eval.eval_nvm_grid --model $LARGE --realism \
    --set kamikaze=False --cells 5x5,8x8,10x10 --episodes 48 \
    --strategies direct,flanking ${K:+--knockout $K} --tag model_ko_$TAGK \
    > experiments/overnight_20260827/q9_$TAGK.log 2>&1
done
step "END(exit=$?)" q9_knockouts

step START q10_gif
BEST=$LARGE; [ -f "$LARGEP" ] && BEST=$LARGEP
timeout 900 python -m eval.render_aegis --model $BEST --nm 10x12 --strategy saturation \
  --terrain maze --stride 1 --fps 14 --seed 21 \
  --out models/eval_output/aegis_maze_routed_10v12.gif > /dev/null 2>&1
step "END(exit=$?)" q10_gif

step START q11_dashboard
timeout 900 python -m eval.dashboard > /dev/null 2>&1
step "END(exit=$?)" q11_dashboard

step ROUND2_DONE queue
