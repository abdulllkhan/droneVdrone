#!/bin/bash
# Ukraine scenario: 7 FPV interceptors vs 5 Shahed-type attackers
# Fresh training from scratch — outcome-only rewards, detection mechanic, hill terrain
# 5-stage curriculum: 40M steps/stage = 200M total
set -e

cd "$(dirname "$0")/.."

echo "=============================================="
echo "  UKRAINE 7v5 — Shahed vs FPV Interceptors"
echo "  40M steps × 5 stages = 200M total"
echo "  NEW env: detection flag, hills, outcome reward"
echo "  $(date)"
echo "=============================================="

python3 -u -m train.train_ukraine \
    2>&1 | tee logs/train_ukraine_7v5.log

echo ""
echo "=============================================="
echo "  UKRAINE TRAINING COMPLETE! $(date)"
echo "=============================================="
