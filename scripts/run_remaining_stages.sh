#!/bin/bash
set -e

STAGE1_PID=$1

if [ -n "$STAGE1_PID" ]; then
    echo "Waiting for Stage 1 (PID: $STAGE1_PID) to finish..."
    while kill -0 $STAGE1_PID 2>/dev/null; do
        sleep 5
    done
    echo "Stage 1 finished."
fi

echo "Starting Stage 2: Conservative MBRL"
.venv/bin/python scripts/run_ablation.py --config configs/experiments/fuhua_low_risk_conservative.yaml --log-file logs/stage2_conservative.log

echo "Starting Stage 3: Planning Enabled MBRL"
.venv/bin/python scripts/run_ablation.py --config configs/experiments/fuhua_low_risk_planning.yaml --log-file logs/stage3_planning.log

echo "All stages completed successfully!"
