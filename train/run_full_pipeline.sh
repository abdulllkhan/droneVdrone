#!/bin/bash
# Full training pipeline: small arena curriculum → realistic 300m arena
# Runs for 8-12+ hours on DGX Spark (128GB, 1536 CUDA cores)
set -e

echo "=============================================="
echo "  FULL TRAINING PIPELINE"
echo "  Phase A: 15v10 small arena (100M steps)"
echo "  Phase B: 15v10 realistic 300m arena (200M steps)"
echo "  Total: ~300M steps"
echo "=============================================="

# Phase A: Wait for current small-arena curriculum to finish
SMALL_PID=$(pgrep -f "train_swarm_curriculum.*--defenders 15.*--num-worlds 1024" || echo "")
if [ -n "$SMALL_PID" ]; then
    echo "Phase A already running (PID $SMALL_PID), waiting..."
    while kill -0 "$SMALL_PID" 2>/dev/null; do
        STEP=$(tail -1 logs/train_15v10_maxgpu_v4.log 2>/dev/null | grep -oP 'total_timesteps \| \K[0-9]+' || echo "?")
        echo "  $(date '+%H:%M') — Steps: $STEP"
        sleep 120
    done
    echo "Phase A complete!"
fi

# Phase B: Realistic 300m arena training
echo ""
echo "=============================================="
echo "  PHASE B: 15v10 REALISTIC 300m ARENA"
echo "  40M steps × 5 stages = 200M total"
echo "=============================================="

# Find best model from Phase A to warm-start
RESUME=""
for f in models/swarm_15v10_stage5_realworld_final.zip \
         models/swarm_15v10_stage4_hard_final.zip \
         models/best_model.zip; do
    if [ -f "$f" ]; then
        RESUME="--resume $f"
        echo "  Warm-starting from: $f"
        break
    fi
done

python3 -m train.train_swarm_curriculum \
    --defenders 15 --attackers 10 \
    --num-worlds 1024 \
    --timesteps-per-stage 40000000 \
    --obstacles 0 \
    --k-opponents 5 --k-teammates 4 \
    --kamikaze \
    --batch-size 4096 \
    --buffer-size 200000 \
    --realistic \
    $RESUME

echo ""
echo "=============================================="
echo "  PIPELINE COMPLETE! $(date)"
echo "=============================================="
