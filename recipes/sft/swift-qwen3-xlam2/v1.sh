#!/bin/bash
# -*- coding: utf-8 -*-

# 1. 定义本次实验的特定版本名
export RECIPE_NAME="v1"

# 2. (可选)在这里覆盖默认参数，例如：
# export LR=2e-5  # v2 可以改成 2e-5

# 3. 移交控制权给当前目录下的“配方脚本”
# 获取当前脚本所在目录，确保 source 路径正确
CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$CURRENT_DIR/run_recipe.sh"
