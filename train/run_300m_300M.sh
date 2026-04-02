#!/bin/bash
# 300M step training on 300m arena (60M/stage x 5 stages)
# Warm-starts from the 200M run's best model
set -e

echo "=============================================="
echo "  300M STEP TRAINING (300m arena)"
echo "  60M steps × 5 stages = 300M total"
echo "=============================================="

# Find best model from the 200M run to warm-start
RESUME=""
for f in models/swarm_15v10_stage5_realworld_300m_final.zip \
         models/swarm_15v10_stage4_hard_300m_final.zip \
         models/best_model.zip; do
    if [ -f "$f" ]; then
        RESUME="--resume $f"
        echo "  Warm-starting from: $f"
        break
    fi
done

python3 -u -m train.train_300m_fixed \
    --defenders 15 --attackers 10 \
    --num-worlds 1024 \
    --timesteps-per-stage 60000000 \
    --obstacles 0 \
    --k-opponents 5 --k-teammates 4 \
    --batch-size 4096 \
    --buffer-size 200000 \
    $RESUME

echo ""
echo "=============================================="
echo "  300M TRAINING COMPLETE! $(date)"
echo "=============================================="
