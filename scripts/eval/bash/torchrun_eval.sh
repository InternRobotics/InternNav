# use to run distributed eval with 4 gpus on single node

# MID_RUN_NAME="InternVLA-N1"
# torchrun \
#   --nproc_per_node=8 \
#   --master_port=2333 \
#   scripts/eval/eval.py \
#     --config scripts/eval/configs/habitat_cfg.py \
#   > logs/${MID_RUN_NAME}_log.txt 2>&1

# CUDA_VISIBLE_DEVICES=6,7
MID_RUN_NAME="InternVLA-N1"
torchrun \
  --nproc_per_node=8 \
  --master_port=29501 \
  scripts/eval/eval_habitat.py \
    --model_path checkpoints/InternVLA-N1 \
    --continuous_traj \
    --output_path logs/habitat/test_new_checkpoint2 \
  > logs/${MID_RUN_NAME}_old_log1.txt 2>&1
