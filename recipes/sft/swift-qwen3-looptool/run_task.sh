#!/bin/bash

# =========================================================
# 1. 自动命名逻辑
# =========================================================
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "$CURRENT_DIR")"
GROUP_NAME="$(basename "$CURRENT_DIR")"

if [ -z "$RECIPE_NAME" ]; then
    echo "❌ Error: RECIPE_NAME is not set. Please run from v1.sh."
    exit 1
fi

export JOB_TIMESTAMP="$(date +%Y%m%d_%H%M)"
export FULL_JOB_NAME="${GROUP_NAME}-${RECIPE_NAME}-${JOB_TIMESTAMP}"

# =========================================================
# 2. 加载环境
# =========================================================
source "$PARENT_DIR/global_config.sh"

if [ -n "$USE_LOCAL_SWIFT" ]; then
    echo "🔌 Activating Local Venv: $SWIFT_ENV_PATH"
    source "$SWIFT_ENV_PATH/bin/activate"
else
    echo "⚡️ Using System Swift (Default)"
fi

OUTPUT_DIR="$CHECKPOINT_ROOT/$FULL_JOB_NAME"
mkdir -p "$OUTPUT_DIR"

export SWANLAB_LOG_DIR="$OUTPUT_DIR/swanlab_logs"
mkdir -p "$SWANLAB_LOG_DIR"

echo "======================================================="
echo "🚀 Launching Sloop Experiment: $FULL_JOB_NAME"
echo "======================================================="

# =========================================================
# 3. 定义全量默认参数
# =========================================================

# --- A. 模型与数据 ---
: "${BASE_MODEL:=/dfs/data/models/Qwen3-8B}"
: "${DATA_FILE:=/dfs/data/datasets/LoopTool-23k/LoopTool_grpo_training_data.json}"
: "${MAX_LENGTH:=40960}"

# --- B. 训练基础超参 ---
: "${TRAIN_TYPE:=lora}"
: "${EPOCHS:=2}"
: "${LR:=1e-5}"

: "${BATCH_SIZE:=1}"

# --- 自动计算 Accum ---
TARGET_GLOBAL_BATCH=64
GPU_COUNT=$(nvidia-smi -L | wc -l 2>/dev/null || echo 1)
if [ "$GPU_COUNT" -eq 0 ]; then GPU_COUNT=1; fi

CALC_ACCUM=$((TARGET_GLOBAL_BATCH / (BATCH_SIZE * GPU_COUNT)))
if [ "$CALC_ACCUM" -lt 1 ]; then CALC_ACCUM=1; fi

: "${GRAD_ACCUM:=$CALC_ACCUM}"

echo "🧮 Auto-Scaling Config:"
echo "   GPUs: $GPU_COUNT | Local BS: $BATCH_SIZE | Accum: $GRAD_ACCUM"
echo "   => Global Batch Size: $((BATCH_SIZE * GPU_COUNT * GRAD_ACCUM))"

: "${WARMUP_RATIO:=0.05}"
: "${DTYPE:=bfloat16}"
: "${ATTN_IMPL:=flash_attention_2}"

# --- C. LoRA 专属配置 ---
: "${LORA_RANK:=16}"
: "${LORA_ALPHA:=32}"
: "${TARGET_MODULES:=all-linear}"

# --- D. 验证与保存 ---
: "${EVAL_STEPS:=10}"  
: "${SAVE_STEPS:=10}"
: "${SAVE_LIMIT:=2}"
: "${LOGGING_STEPS:=5}"

# --- E. 系统与日志 ---
: "${NUM_WORKERS:=8}"
: "${GRAD_CHECKPOINTING:=true}" 
: "${REPORT_TO:=swanlab}"

# =========================================================
# [DeepSpeed] 自动生成 DeepSpeed Zero2 Offload 配置文件
# =========================================================
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
  "bf16": {
    "enabled": "auto"
  },
  "fp16": {
    "enabled": "auto"
  }
}
EOF

echo "📝 DeepSpeed Config generated at: $DS_CONFIG_PATH"

# =========================================================
# 4. 执行 Swift (启动分布式 DDP 模式)
# =========================================================

# [🔥 关键] 设置进程数，触发 torchrun DDP 模式
export NPROC_PER_NODE=$GPU_COUNT
export MASTER_PORT=$(($RANDOM + 20000))

swift sft \
    --model "$BASE_MODEL" \
    --train_type "$TRAIN_TYPE" \
    --dataset "$DATA_FILE" \
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
    --deepspeed "$DS_CONFIG_PATH" 
    # [删除了 --device_map ""，DDP模式下不需要且会报错]

echo "✅ Experiment Finished: $FULL_JOB_NAME"