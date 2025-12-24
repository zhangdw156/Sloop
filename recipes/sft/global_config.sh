#!/bin/bash

# =========================================================
# Sloop Project Global Configuration
# =========================================================

# --- 身份标识 ---
export PROJECT_NAME="Sloop"
export MODEL_AUTHOR="zhangdw"

# --- 基础设施路径 ---
# 虚拟环境路径
export SWIFT_ENV_PATH="/dfs/data/uv-venv/modelscope"
# Checkpoint 根目录
export CHECKPOINT_ROOT="/dfs/data/work/Sloop/checkpoints"

# --- 全局默认行为 ---
# export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
