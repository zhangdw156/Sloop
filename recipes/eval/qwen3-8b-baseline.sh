#!/bin/bash

# ==========================================
# Layer 3: Recipe
# ==========================================

export EVAL_CORE_DIR="/dfs/data/work/Sloop/evals/core"
export VENV_PATH="/dfs/data/uv-venv/modelscope"

# --- 模型配置 ---
export EVAL_MODEL_NAME="qwen3-8b-baseline"
# export EVAL_API_URL="http://10.254.13.246:8443/service-large-1045-1766758890786/llm/v1"
# export EVAL_API_KEY="KJLBQJc7jnwl86jl67r687187mAQcsBvrr8gpArw0rmR74tqlzZHtS9sc76kqdvGJ8pkwS0wc64dSWPv0mVx87mVr7F74k4N6m7THFWqrzzmB5KnjD997p8CS7t064X7"
export EVAL_API_URL="http://127.0.0.1:8000/v1"

# --- 评测范围配置 ---
export EVAL_SUBSET_LIST="multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context"
# export EVAL_SUBSET_LIST=""  # 留空则跑全量

# --- 参数配置 ---
export EVAL_BATCH_SIZE="32"
export EVAL_LIMIT=""
export EVAL_MAX_TOKENS="4096"

# --- 输出路径 ---
# TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# export EVAL_OUTPUT_DIR="/dfs/data/work/Sloop/eval_results/${EVAL_MODEL_NAME}_${TIMESTAMP}"
export EVAL_OUTPUT_DIR="/dfs/data/work/Sloop/eval_results/${EVAL_MODEL_NAME}"

# --- 启动 ---
if [ -f "$EVAL_CORE_DIR/run_inner.sh" ]; then
    bash "$EVAL_CORE_DIR/run_inner.sh"
else
    echo "❌ Error: Cannot find driver script."
    exit 1
fi
