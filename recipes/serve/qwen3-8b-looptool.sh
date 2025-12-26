#!/bin/bash
# -*- coding: utf-8 -*-

# ==========================================
# Layer 3: vLLM Recipe (LoRA Version)
# ==========================================

# --- 核心路径 ---
export EVAL_CORE_DIR="/dfs/data/work/Sloop/serve/core"

# --- 模型配置 ---
# 使用 :- 语法，允许外部覆盖，同时保留默认值
export SERVE_MODEL_NAME=${SERVE_MODEL_NAME:-"qwen3-8b-looptool"}
export SERVE_MODEL_PATH=${SERVE_MODEL_PATH:-"/dfs/data/models/Qwen3-8B"}

# --- LoRA配置 ---
export SERVE_ENABLE_LORA=${SERVE_ENABLE_LORA:-"true"}
export SERVE_MAX_LORA_RANK=${SERVE_MAX_LORA_RANK:-"128"}
export SERVE_LORA_MODULES=${SERVE_LORA_MODULES:-"bfclv3=/dfs/data/work/Sloop/checkpoints/swift-qwen3-looptool-.v1.1766635255394-20251225_1201/v0-20251225-120127/checkpoint-86"}

# --- 硬件与性能 ---
export SERVE_PORT=${SERVE_PORT:-"8000"}
export SERVE_TP_SIZE=${SERVE_TP_SIZE:-"1"}
export SERVE_GPU_UTIL=${SERVE_GPU_UTIL:-"0.90"}
export SERVE_SWAP_SPACE=${SERVE_SWAP_SPACE:-"16"}
export SERVE_MAX_LEN=${SERVE_MAX_LEN:-"65536"}
export SERVE_TOOL_PARSER=${SERVE_TOOL_PARSER:-"hermes"}

# --- 启动模式 ---
# foreground or daemon
export SERVE_MODE=${SERVE_MODE:-"daemon"}

# ==========================================
# 启动驱动
# ==========================================
bash "$EVAL_CORE_DIR/run_vllm_driver.sh"
