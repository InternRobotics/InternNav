#!/bin/bash

# Activate conda automatically
source /root/miniconda3/etc/profile.d/conda.sh

mode="$1"
shift   # remove first argument so only extra args left (--config ...)

case "$mode" in
    habitat)
        echo "[run.sh] Starting HABITAT evaluation..."

        conda activate habitat

        python scripts/eval/eval.py \
            --config scripts/eval/configs/habitat_cfg.py \
            > logs/internvla_n1_log.txt 2>&1

        ;;

    internutopia)
        echo "[run.sh] Starting INTERNUTOPIA evaluation..."

        conda activate internutopia

        # -------- parse remaining arguments (e.g., --config xxx) --------
        while [[ $# -gt 0 ]]; do
            case $1 in
                --config)
                    CONFIG="$2"
                    shift 2
                    ;;
                *)
                    echo "Unknown parameter: $1"
                    exit 1
                    ;;
            esac
        done
        # ----------------------------------------------------------------

        if [ "$RANK" -eq 0 ]; then
            echo "[run.sh] Starting Ray head..."
            RAY_max_direct_call_object_size=104857600 \
                ray start --head --port=6379

            sleep 20s

            echo "[run.sh] Exec start_eval.sh..."
            bash scripts/eval/bash/start_eval.sh

            sleep inf
        else
            echo "[run.sh] Starting Ray worker..."
            RAY_max_direct_call_object_size=104857600 \
                ray start --address=${MASTER_ADDR}:6379

            sleep inf
        fi
        ;;

    *)
        echo "Usage: $0 {habitat|internutopia} [--config xxx]"
        exit 1
        ;;
esac
