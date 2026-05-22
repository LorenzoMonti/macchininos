#!/bin/bash
# Submit Full and Light training jobs.
# Usage: bash launch_train_all.sh
# Each job writes its own log: logs/train_full_<jobid>.out / train_light_<jobid>.out

set -euo pipefail

ACCOUNT="IscrC_TranRR"
PARTITION="boost_usr_prod"
WALLTIME="04:00:00"

submit_job() {
    local VARIANT=$1
    sbatch \
        -A "$ACCOUNT" \
        -p "$PARTITION" \
        --time="$WALLTIME" \
        -N 1 \
        --ntasks-per-node=1 \
        --gres=gpu:1 \
        --cpus-per-task=16 \
        --mem=123000 \
        --job-name="train_${VARIANT}" \
        --output="logs/train_${VARIANT}_%j.out" \
        --error="logs/train_${VARIANT}_%j.err" \
        --wrap="
            mkdir -p logs saved_models
            unset PYTHONPATH
            source macch/bin/activate
            export PYTHONPATH=\$PYTHONPATH:.
            export TF_FORCE_GPU_ALLOW_GROWTH=true
            export TF_XLA_FLAGS='--tf_xla_enable_xla_devices=false'
            export TF_CPP_MIN_LOG_LEVEL=2
            export HF_HUB_OFFLINE=1
            echo '=========================================='
            echo 'VARIANT: ${VARIANT}'
            echo 'START:' \$(date)
            echo '=========================================='
            srun python3 -u main.py --variant ${VARIANT}
            echo '=========================================='
            echo 'DONE:' \$(date)
            echo '=========================================='
            deactivate
        "
}

mkdir -p logs

JOB_FULL=$(submit_job full)
JOB_LIGHT=$(submit_job light)

echo "$JOB_FULL"
echo "$JOB_LIGHT"
echo "Logs: logs/train_full_<id>.out  /  logs/train_light_<id>.out"
