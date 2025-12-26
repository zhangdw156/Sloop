#!/bin/bash

# ==========================================
# Layer 3: Recipe
# ==========================================

export EVAL_CORE_DIR="/dfs/data/work/Sloop/evals/core"
export VENV_PATH="/dfs/data/uv-venv/modelscope"

# --- 模型配置 ---
export EVAL_MODEL_NAME="qwen3-8b-baseline" 
export EVAL_API_URL="http://10.254.8.156:8443/service-large-1045-1766723863116/llm/v1"
export EVAL_API_KEY="f6g5Mfq4bxKSw28bDDaBS0gPFQqAac6864RcTQJh5zwShaqbvJsLW88TaSr6pArCcQsfGJ6VRscr8NcQHXV5rr8bLBPasrpNAGrKhQfJcD7F086K7w7THgPS5F0wrX6A"

# --- 评测范围配置 
export EVAL_SUBSET_LIST="multi_turn_base,multi_turn_miss_func,multi_turn_miss_param,multi_turn_long_context"
# export EVAL_SUBSET_LIST=""  # 留空则跑全量

# --- 参数配置 ---
export EVAL_BATCH_SIZE="10"
export EVAL_LIMIT="" 
export EVAL_MAX_TOKENS="32000" # 可以控制生成长度了

# --- 输出路径 ---
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export EVAL_OUTPUT_DIR="/dfs/data/work/Sloop/eval_results/${EVAL_MODEL_NAME}_${TIMESTAMP}"

# --- 启动 ---
if [ -f "$EVAL_CORE_DIR/run_inner.sh" ]; then
    bash "$EVAL_CORE_DIR/run_inner.sh"
else
    echo "❌ Error: Cannot find driver script."
    exit 1
fi