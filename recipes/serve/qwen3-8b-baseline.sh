#!/bin/bash

# ==========================================
# Layer 3: vLLM Recipe (LoRA Version)
# ==========================================

# --- 1. 核心路径 ---
export EVAL_CORE_DIR="/dfs/data/work/Sloop/serve/core"

# --- 2. 基础模型配置 ---
# 这里是基座模型 (Base Model)
export SERVE_MODEL_NAME="qwen3-8b-baseline"
export SERVE_MODEL_PATH="/dfs/data/models/Qwen3-8B" 

# --- 3. LoRA配置 (按照你的要求) ---
export SERVE_ENABLE_LORA="false"
# export SERVE_MAX_LORA_RANK="128"
# export SERVE_LORA_MODULES="bfclv3=/dfs/data/work/Sloop/checkpoints/swift-qwen3-xlam2-v2-20251226_1203/v0-20251226-120344/checkpoint-22"

# --- 4. 硬件与性能配置 ---
export SERVE_PORT="8000"
export SERVE_TP_SIZE="1"
export SERVE_GPU_UTIL="0.90"     # 你的要求
export SERVE_SWAP_SPACE="16"     # 你的要求 (16GiB CPU Swap)
export SERVE_MAX_LEN="65536"     # 你的要求

# --- 5. 工具配置 ---
export SERVE_TOOL_PARSER="hermes" # 你的要求

# --- 6. 启动模式 ---
export SERVE_MODE="foreground" 

# ==========================================
# 启动驱动
# ==========================================
bash "$EVAL_CORE_DIR/run_vllm_driver.sh"
