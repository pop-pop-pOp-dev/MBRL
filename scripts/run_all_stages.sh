#!/bin/bash
set -e

mkdir -p logs

echo "Starting Stage 1: PPO Only (Real data only)"
.venv/bin/python scripts/run_ablation.py --config configs/experiments/fuhua_low_risk_real_only.yaml --log-file logs/stage1_real_only.log

echo "Starting Stage 2: Conservative MBRL"
.venv/bin/python scripts/run_ablation.py --config configs/experiments/fuhua_low_risk_conservative.yaml --log-file logs/stage2_conservative.log

echo "Starting Stage 3: Planning Enabled MBRL"
.venv/bin/python scripts/run_ablation.py --config configs/experiments/fuhua_low_risk_planning.yaml --log-file logs/stage3_planning.log

echo "All stages completed successfully!"
