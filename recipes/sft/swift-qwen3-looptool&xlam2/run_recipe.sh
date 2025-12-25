#!/bin/bash

# =========================================================
# Layer 2: Recipe Configuration (定义模型、数据、默认超参)
# =========================================================

# 1. 路径推导与环境加载
RECIPE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 假设 run_task.sh 在 recipes/ 根目录下 (向上两级)
# 如果目录层级不同，请调整这里的 ../.. 
ROOT_RECIPES_DIR="$(cd "$RECIPE_DIR/../.." && pwd)"

# 自动生成 Job Name
GROUP_NAME="$(basename "$RECIPE_DIR")"
export JOB_TIMESTAMP="$(date +%Y%m%d_%H%M)"
export FULL_JOB_NAME="${GROUP_NAME}-${RECIPE_NAME}-${JOB_TIMESTAMP}"

# 加载全局配置 (global_config.sh)
source "$ROOT_RECIPES_DIR/global_config.sh"

# 激活 Python 环境
if [ -n "$USE_LOCAL_SWIFT" ]; then
    echo "🔌 Activating Local Venv: $SWIFT_ENV_PATH"
    source "$SWIFT_ENV_PATH/bin/activate"
else
    echo "⚡️ Using System Swift (Default)"
fi

# =========================================================
# 2. 定义默认参数 (Default Hyperparameters)
# =========================================================

# --- A. 模型与数据 ---
: "${BASE_MODEL:=/dfs/data/models/Qwen3-8B}"
# 再次提醒：不加引号，空格分隔
: "${DATA_FILE:=/dfs/data/datasets/APIGen-MT-5k/apigen-mt_5k.json /dfs/data/datasets/LoopTool-23k/LoopTool_grpo_training_data.json}"
: "${MAX_LENGTH:=40960}"

# --- B. 训练配置 ---
: "${TRAIN_TYPE:=lora}"
: "${EPOCHS:=2}"
: "${LR:=1e-5}"
: "${BATCH_SIZE:=1}"
: "${LORA_RANK:=16}"
: "${LORA_ALPHA:=32}"
: "${TARGET_MODULES:=all-linear}"

# --- C. 自动计算 Accum (为了保持代码整洁，计算逻辑也可以放这里) ---
TARGET_GLOBAL_BATCH=64
GPU_COUNT=$(nvidia-smi -L | wc -l 2>/dev/null || echo 1)
[ "$GPU_COUNT" -eq 0 ] && GPU_COUNT=1
CALC_ACCUM=$((TARGET_GLOBAL_BATCH / (BATCH_SIZE * GPU_COUNT)))
[ "$CALC_ACCUM" -lt 1 ] && CALC_ACCUM=1
: "${GRAD_ACCUM:=$CALC_ACCUM}"

# --- D. 其他固定参数 ---
: "${WARMUP_RATIO:=0.05}"
: "${DTYPE:=bfloat16}"
: "${ATTN_IMPL:=flash_attention_2}"
: "${EVAL_STEPS:=200}" : "${SAVE_STEPS:=200}" : "${SAVE_LIMIT:=2}"
: "${NUM_WORKERS:=8}" : "${GRAD_CHECKPOINTING:=true}" : "${REPORT_TO:=swanlab}"

echo "======================================================="
echo "🥣 Recipe Configured: $FULL_JOB_NAME"
echo "   GPUs: $GPU_COUNT | BS: $BATCH_SIZE | Accum: $GRAD_ACCUM"
echo "======================================================="

# 3. 召唤核心引擎 (Call Layer 3)
source "$ROOT_RECIPES_DIR/swift_run_task.sh"