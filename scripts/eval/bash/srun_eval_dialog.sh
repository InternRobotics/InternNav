# use to run distributed eval with 8 gpus on single node
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export NCCL_SOCKET_IFNAME=bond0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5

srun -p mozi_t \
    --gres=gpu:1 \
    --ntasks=1 \
    --time=0-20:00:00 \
    --ntasks-per-node=1 \
    --cpus-per-task=8 \
    --exclude=HOST-10-140-66-53,HOST-10-140-66-69 \
    --kill-on-bad-exit=1 \
    python scripts/eval/eval.py \
      --config scripts/eval/configs/habitat_dialog_cfg.py \
