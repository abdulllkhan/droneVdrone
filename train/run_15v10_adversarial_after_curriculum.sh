#!/bin/bash
# Waits for curriculum training to finish, then runs adversarial co-training.
# This is meant to run unattended for 12+ hours.

set -e

CURRICULUM_PID=$(pgrep -f "train.train_swarm_curriculum.*--defenders 15" || echo "")

if [ -n "$CURRICULUM_PID" ]; then
    echo "Waiting for curriculum training (PID $CURRICULUM_PID) to finish..."
    while kill -0 "$CURRICULUM_PID" 2>/dev/null; do
        sleep 60
        echo "  Still running... $(date)"
    done
    echo "Curriculum training finished!"
fi

# Find the best model from curriculum training
DEFENDER_MODEL=""
for f in models/swarm_15v10_stage5_realworld_final.zip \
         models/swarm_15v10_stage4_hard_final.zip \
         models/swarm_15v10_stage3_dynamic_final.zip \
         models/best_model.zip; do
    if [ -f "$f" ]; then
        DEFENDER_MODEL="$f"
        break
    fi
done

if [ -z "$DEFENDER_MODEL" ]; then
    echo "ERROR: No defender model found from curriculum training!"
    exit 1
fi

echo "Using defender model: $DEFENDER_MODEL"
echo "Starting adversarial co-training: 15v10, 4 rounds, 10M steps each..."

python3 -m train.train_swarm_adversarial \
    --defenders 15 --attackers 10 \
    --defender "$DEFENDER_MODEL" \
    --rounds 4 \
    --num-worlds 128 \
    --timesteps 10000000 \
    --obstacles 3 \
    --wind --wind-force 0.3 \
    --noise 0.02 \
    --mixture \
    --sensor-range 5.0 \
    --gust-prob 0.02 --gust-force 1.0 \
    --speed-var 0.15 --mass-var 0.1 \
    --k-opponents 5 --k-teammates 4 \
    --kamikaze

echo "Adversarial co-training complete! $(date)"
