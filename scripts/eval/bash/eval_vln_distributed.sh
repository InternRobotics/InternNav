#!/bin/bash
# ----------------------------------------------------------------------------
# USAGE INSTRUCTIONS:
#
# This script runs a distributed evaluation for Habitat or Internutopia.
# You can choose from the following modes:
#
# 1. **habitat**: Runs evaluation using the Habitat environment.
# 2. **internutopia**: Runs evaluation using the Internutopia environment.
# 3. **internutopia_vec_env**: Runs distributed evaluation with Ray in vectorized environments.
#
# The script automatically activates the appropriate Conda environment based on the mode.
# You can specify a custom configuration file using the `--config` argument.
#
# Example usage:
# ./scripts/eval/bash/eval_vln_distributed.sh habitat --config <path_to_config>
# ./<run.sh> internutopia --config <path_to_config>
# ./<run.sh> internutopia_vec_env --config <path_to_config>
#
# Make sure required Conda environments (habitat, internutopia) and Ray (for internutopia_vec_env) are set up.
# ----------------------------------------------------------------------------

# Activate conda (update to your local path)
source /root/miniconda3/etc/profile.d/conda.sh

mode="$1"
shift   # remove first argument so only extra args left (--config ...)

CONFIG=scripts/eval/configs/h1_internvla_n1_async_cfg.py
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

case "$mode" in
    # start to evaluate habitat in dlc
    habitat)
        echo "[run.sh] Starting HABITAT evaluation..."

        conda activate habitat

        python scripts/eval/eval.py \
            --config $CONFIG

        ;;
    internutopia)
        echo "[run.sh] Starting INTERNUTOPIA evaluation..."

        conda activate internutopia

        python scripts/eval/eval.py \
            --config $CONFIG

        ;;
    internutopia_vec_env)
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
        echo "Usage: $0 {habitat|internutopia|internutopia_vec_env} [--config xxx]"
        exit 1
        ;;
esac
