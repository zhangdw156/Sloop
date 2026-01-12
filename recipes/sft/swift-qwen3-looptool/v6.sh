#!/bin/bash
# -*- coding: utf-8 -*-

# 1. 定义本次实验的特定版本名
export RECIPE_NAME="v6"

# 2. (可选)在这里覆盖默认参数，例如：
export DATA_FILE="/dfs/data/datasets/LoopTool-23k/LoopTool_grpo_training_data_emptythink.json"
export LOSS_SCALE=ignore_empty_think
export TRAIN_TYPE="full"
export ZERO_STAGE=3
export LR=1e-6
export EPOCHS=1

# 3. 移交控制权给当前目录下的“配方脚本”
# 获取当前脚本所在目录，确保 source 路径正确
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CURRENT_DIR/run_recipe.sh"
