export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5

MID_RUN_NAME="InternVLA-N1"
CONFIG="scripts/eval/configs/habitat_cfg.py"

srun -p efm_t \
    --gres=gpu:8 \
    --ntasks=8 \
    -x HOST-10-140-66-68,HOST-10-140-66-182,HOST-10-140-66-181 \
    --time=0-20:00:00 \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python scripts/eval/eval.py \
        --config $CONFIG \
    > logs/${MID_RUN_NAME}_log.txt 2>&1
