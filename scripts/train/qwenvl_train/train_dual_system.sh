#!/bin/bash
#SBATCH -J qwenvl
#SBATCH -p gpu_partition
#SBATCH -N 8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=1
#SBATCH -o ./slurm-%j.out
#SBATCH -e ./slurm-%j.err

# Distributed training configuration
MASTER_ADDR=`scontrol show hostname $SLURM_JOB_NODELIST | head -n1`
MASTER_PORT=$((RANDOM % 101 + 20001))

# DeepSpeed configuration
deepspeed=scripts/train/qwenvl_train/zero2.json

# Model configuration
llm=Qwen/Qwen2.5-VL-7B-Instruct

# Training hyperparameters
lr=1e-4
batch_size=2
grad_accum_steps=1
max_pixels=313600
min_pixels=3136

# Dataset configuration (replace with public dataset names)
vln_datasets=r2r_125cm_0_30%30,r2r_60cm_15_15%30,rxr_125cm_0_30%30,rxr_60cm_15_15%30,scalevln_125cm_0_30%30,scalevln_60cm_30_30%30

# Output configuration
run_name=InternVLA-N1-DualVLN
output_dir=checkpoints/${run_name}
# system 1 options: nextdit_async, navdp_async, nextdit
system1=nextdit_async

system2_ckpt=checkpoints/InternVLA-N1-System2

srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    internnav/trainer/internvla_n1_trainer.py \
    --deepspeed ${deepspeed} \
    --model_name_or_path "${system2_ckpt}" \
    --vln_dataset_use ${vln_datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --bf16 \
    \
    --num_history 8 \
    --data_augmentation True \
    --resize_h 384 \
    --resize_w 384 \
    --sample_step 4 \
    --num_future_steps 4 \
    --predict_step_num 32 \
    --pixel_goal_only True \
    --system1 ${system1} \
    \
    --output_dir ${output_dir} \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels ${max_pixels} \
    --min_pixels ${min_pixels} \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.003 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine_with_min_lr" \
    --lr_scheduler_kwargs '{"min_lr": 1e-05}' \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb
