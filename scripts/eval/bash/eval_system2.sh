export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet

MID_RUN_NAME=InternVLA-N1-System2

srun -p gpu_partition \
    --gres=gpu:8 \
    --ntasks=8 \
    --time=0-20:00:00 \
    --ntasks-per-node=8 \
    --cpus-per-task=16 \
    --kill-on-bad-exit=1 \
    python scripts/eval/eval_habitat.py \
    --model_path checkpoints/${MID_RUN_NAME} \
    --mode system2 \
    --output_path results/$MID_RUN_NAME/val_unseen \
    --habitat_config_path scripts/eval/configs/vln_r2r.yaml
