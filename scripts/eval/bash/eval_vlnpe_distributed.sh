#!/bin/bash
# vlnpe distributed scripts for dlc and slurm
# use this to run distributed eval with 1 gpus on single node
# set use_agent_server=False in config file to use local agent on single node

source /root/miniconda3/etc/profile.d/conda.sh
conda activate internutopia

CONFIG=scripts/eval/configs/h1_internvla_n1_cfg.py
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

# use srun for slurm eval
python scripts/eval/eval.py \
    --config $CONFIG \
