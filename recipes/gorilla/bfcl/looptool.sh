#!/bin/bash

# ================= 0. 路径定位 (关键修改) =================
# 获取当前脚本所在的绝对路径，确保无论在哪里运行都能找到 pipeline_exec.sh
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# ================= 配置区域 (Configuration) =================

# 1. 基础环境路径
export SETUP_SCRIPT="/dfs/data/sbin/setup.sh"
export VENV_ACTIVATE="/dfs/data/uv-venv/gorilla/bin/activate"

# 2. 项目工作目录 (BFCL 代码所在的根目录)
# 脚本会 cd 到这个目录执行，确保 result/score 生成在正确位置
export PROJECT_ROOT="/dfs/data/work/gorilla/berkeley-function-call-leaderboard"

# 3. 模型配置
export MODEL_NAME="Qwen/Qwen3-8B-FC"
export LOCAL_MODEL_PATH="/dfs/data/models/Qwen3-8B"

# 4. 评测配置
export TEST_CATEGORY="multi_turn"          # generate 阶段用的
export EVAL_CATEGORY="multi_turn_base"     # evaluate 阶段用的
export THREADS=32
export GPU_MEM_UTIL=0.9

# 5. 结果归档配置
# 最终结果会被移动到: $PROJECT_ROOT/$OUTPUT_DIR_NAME
export OUTPUT_DIR_NAME="lootool" 

# 6. 核心脚本路径 (Layer 1 脚本的位置)
# 假设核心脚本在当前目录下，你也可以写绝对路径
export CORE_SCRIPT="${SCRIPT_DIR}/run_task.sh"

# ================= 启动执行 =================
echo "📋 Configuration loaded for: $MODEL_NAME"
echo "📂 Output will be saved to: $PROJECT_ROOT/$OUTPUT_DIR_NAME"

chmod +x "$CORE_SCRIPT"
"$CORE_SCRIPT"