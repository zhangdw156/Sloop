#!/bin/bash

# =========================================================
# Layer 3: Execution Engine (DeepSpeed 生成 + 启动命令)
# =========================================================

# 1. 准备输出目录
OUTPUT_DIR="$CHECKPOINT_ROOT/$FULL_JOB_NAME"
mkdir -p "$OUTPUT_DIR"
export SWANLAB_LOG_DIR="$OUTPUT_DIR/swanlab_logs"
mkdir -p "$SWANLAB_LOG_DIR"

# 2. 自动生成 DeepSpeed Config
DS_CONFIG_PATH="$OUTPUT_DIR/ds_config.json"
cat <<EOF > "$DS_CONFIG_PATH"
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 200000000,
    "reduce_scatter": true,
    "reduce_bucket_size": 200000000,
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "bf16": { "enabled": "auto" },
  "fp16": { "enabled": "auto" }
}
EOF
echo "📝 DeepSpeed Config generated at: $DS_CONFIG_PATH"

# 3. 设置分布式环境 (DDP)
# 如果 GPU_COUNT 没定义，重新获取一下作为保底
if [ -z "$GPU_COUNT" ]; then
    GPU_COUNT=$(nvidia-smi -L | wc -l 2>/dev/null || echo 1)
fi
export NPROC_PER_NODE=$GPU_COUNT
export MASTER_PORT=$(($RANDOM + 20000))

echo "🚀 Launching Swift Task..."

# 4. 执行训练命令
# 注意：--dataset 使用不带引号的 $DATA_FILE 以支持多数据集
swift sft \
    --model "$BASE_MODEL" \
    --train_type "$TRAIN_TYPE" \
    --dataset $DATA_FILE \
    --torch_dtype "$DTYPE" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LR" \
    --lora_rank "$LORA_RANK" \
    --lora_alpha "$LORA_ALPHA" \
    --target_modules "$TARGET_MODULES" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_LIMIT" \
    --logging_steps "$LOGGING_STEPS" \
    --max_length "$MAX_LENGTH" \
    --output_dir "$OUTPUT_DIR" \
    --warmup_ratio "$WARMUP_RATIO" \
    --dataloader_num_workers "$NUM_WORKERS" \
    --model_author "$MODEL_AUTHOR" \
    --model_name "$FULL_JOB_NAME" \
    --report_to "$REPORT_TO" \
    --swanlab_project "$PROJECT_NAME" \
    --swanlab_exp_name "$FULL_JOB_NAME" \
    --gradient_checkpointing "$GRAD_CHECKPOINTING" \
    --packing true \
    --attn_impl "$ATTN_IMPL" \
    --deepspeed "$DS_CONFIG_PATH" \
    --loss_scale ${LOSS_SCALE}

echo "✅ Experiment Finished: $FULL_JOB_NAME"