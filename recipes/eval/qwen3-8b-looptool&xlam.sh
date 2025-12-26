#!/bin/bash
# -*- coding: utf-8 -*-

# ==========================================
# Layer 3: Recipe (复制并修改这里)
# ==========================================

# --- 1. 核心路径配置 (指向 Layer 1/2 所在的目录) ---
# 确保这里指向存放 run_inner.sh 的绝对路径
export EVAL_CORE_DIR="/dfs/data/work/Sloop/evals/core"
export VENV_PATH="/dfs/data/uv-venv/modelscope"

# --- 2. 评测目标配置 (根据当次评测修改) ---
# 这里的名字要和 vLLM 启动时的 --served-model-name 一致
export EVAL_MODEL_NAME="qwen3-8b-looptool_xlam2" 
export EVAL_API_URL="http://10.254.228.90:8443/service-large-1045-1766731737925/llm/v1"
export EVAL_API_KEY="7jqk6wQ2z8kpsq4zdG7FrjslwSJ08dQ5pb7wtrA27F0AD7kkbrVH8ZQPcnbcvxzT27tqCH88a7mTWSfB8NqX6KfcP7K7KSrZaG9Ht8z7b8XR6Z0tSQ0ArFG09D768wca"

# --- 3. 评测参数 ---
export EVAL_BATCH_SIZE="10"
# 设置为 "10" 进行快速测试，设置为 "" (空) 进行全量评测
export EVAL_LIMIT="" 

# --- 4. 输出路径 ---
# 自动生成一个带时间戳的文件夹，避免覆盖
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export EVAL_OUTPUT_DIR="/dfs/data/work/Sloop/eval_results/${EVAL_MODEL_NAME}_${TIMESTAMP}"

# ==========================================
# 启动驱动层 (Handover to Layer 2)
# ==========================================
if [ -f "$EVAL_CORE_DIR/run_inner.sh" ]; then
    bash "$EVAL_CORE_DIR/run_inner.sh"
else
    echo "❌ Error: Cannot find driver script at $EVAL_CORE_DIR/run_inner.sh"
    exit 1
fi
