MID_RUN_NAME="InternVLA-N1"
CONFIG="scripts/eval/configs/habitat_dual_system_cfg.py"

srun -p <YOUR_PARTITION_NAME> \
    --gres=gpu:8 \
    --ntasks=8 \
    --time=0-20:00:00 \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python scripts/eval/eval.py \
        --config $CONFIG \
    > logs/${MID_RUN_NAME}_log.txt 2>&1
