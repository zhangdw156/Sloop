#!/bin/bash
# -*- coding: utf-8 -*-

# ==========================================
# Layer 3: Recipe
# ==========================================

export EVAL_CORE_DIR="/dfs/data/work/Sloop/evals/core"
export VENV_PATH="/dfs/data/uv-venv/modelscope"

# --- 模型配置 ---
export EVAL_MODEL_NAME="Qwen3-235B-A22B-Instruct-2507"
export EVAL_API_URL="http://103.212.13.119:8000/v1/"
export EVAL_API_KEY="qwertiasagv"

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
