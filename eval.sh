#!/bin/bash
# Evaluate trained IPPO agents for HireRL

if [ -z "$1" ]; then
    echo "Usage: ./eval.sh <run_name>"
    echo "Example: ./eval.sh hirerl_ippo_20250113_143022"
    echo ""
    echo "Available runs:"
    ls -1 runs/ | grep -v "\.DS_Store"
    exit 1
fi

RUN_NAME=$1

python evaluate_policy.py --run_name "$RUN_NAME" --n_episodes 10 --save_plots
