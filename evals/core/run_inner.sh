#!/bin/bash

# ==========================================
# Layer 2: Driver (通用驱动，不含业务参数)
# ==========================================

SECONDS=0
# 1. 获取绝对路径
CORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$CORE_DIR/eval_bfcl.py"

# 2. 检查必要变量
if [ -z "$VENV_PATH" ] || [ -z "$EVAL_OUTPUT_DIR" ]; then
    echo "❌ Error: VENV_PATH or EVAL_OUTPUT_DIR not set."
    exit 1
fi

# 3. 激活环境
source "$VENV_PATH/bin/activate"
# (可选) 只有当文件存在时才加载 setup.sh
# 这是我在modelfactory平台的初始化脚本
if [ -f "/dfs/data/sbin/setup.sh" ]; then
    source /dfs/data/sbin/setup.sh
fi

# 4. 准备日志
mkdir -p "$EVAL_OUTPUT_DIR"
LOG_FILE="$EVAL_OUTPUT_DIR/eval.log"

echo "========================================"
echo "🚀 Starting Evaluation Wrapper"
echo "📂 Output Dir: $EVAL_OUTPUT_DIR"
echo "📝 Logging to: $LOG_FILE"
echo "🐍 Script: $PYTHON_SCRIPT"
echo "========================================"

# 5. 执行 Python (保留你的 uv run 需求)
# 方法 A: 加上 --no-project (推荐，更优雅，不需要 cd /)
uv run --no-project "$PYTHON_SCRIPT" 2>&1 | tee -a "$LOG_FILE"

# 方法 B: 你原来的 cd / (保留你的习惯)
# 注意：因为 PYTHON_SCRIPT 已经是绝对路径，所以这里 cd / 也是安全的
# (
#     cd /
#     uv run "$PYTHON_SCRIPT"
# ) 2>&1 | tee -a "$LOG_FILE"

# 6. 结果处理
EXIT_CODE=${PIPESTATUS[0]}
DURATION=$SECONDS
TIME_STR=$(printf "%02d:%02d:%02d" $((DURATION/3600)) $(((DURATION%3600)/60)) $((DURATION%60)))

echo "----------------------------------------"
echo "⏱️  Total Time: $TIME_STR ($DURATION seconds)"

if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Evaluation Failed (Exit Code: $EXIT_CODE)"
    exit $EXIT_CODE
fi

echo "✅ All Done."