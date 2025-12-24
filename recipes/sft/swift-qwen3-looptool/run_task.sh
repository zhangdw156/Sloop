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

export FULL_JOB_NAME="${GROUP_NAME}-${RECIPE_NAME}"

# =========================================================
# 2. 加载环境
# =========================================================
source "$PARENT_DIR/global_config.sh"
source "$SWIFT_ENV_PATH/bin/activate"
OUTPUT_DIR="$CHECKPOINT_ROOT/$FULL_JOB_NAME"

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
: "${EPOCHS:=4}"
: "${LR:=1e-5}"
: "${BATCH_SIZE:=2}"
: "${EVAL_BATCH_SIZE:=2}"
: "${GRAD_ACCUM:=8}"
: "${WARMUP_RATIO:=0.05}"
: "${DTYPE:=bfloat16}"

# --- C. LoRA 专属配置 ---
: "${LORA_RANK:=16}"
: "${LORA_ALPHA:=32}"
: "${TARGET_MODULES:=all-linear}"

# --- D. 验证与保存 ---
# 建议调大一点，每 50 步测一次有点太频繁了，会拖慢训练
: "${EVAL_STEPS:=200}"  
: "${SAVE_STEPS:=200}"
: "${SAVE_LIMIT:=2}"
: "${LOGGING_STEPS:=5}"

# --- E. 系统与日志 ---
: "${NUM_WORKERS:=4}"
: "${GRAD_CHECKPOINTING:=true}"
: "${REPORT_TO:=swanlab}"

# =========================================================
# 4. 执行 Swift
# =========================================================

mkdir -p "$OUTPUT_DIR"

swift sft \
    --model "$BASE_MODEL" \
    --train_type "$TRAIN_TYPE" \
    --dataset "$DATA_FILE" \
    --torch_dtype "$DTYPE" \
    --num_train_epochs "$EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
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
    --gradient_checkpointing "$GRAD_CHECKPOINTING"

echo "✅ Experiment Finished: $FULL_JOB_NAME"