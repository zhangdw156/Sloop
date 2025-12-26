#!/bin/bash

# ==========================================
# Layer 2: Driver (不要修改此文件)
# 负责：激活环境 -> 准备目录 -> 记录日志 -> 启动 Python
# ==========================================

# [新增] 1. 计时器归零
SECONDS=0

# 2. 确定当前脚本所在的目录 (Core 目录)
CORE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="$CORE_DIR/eval_bfcl.py"

# 3. 检查必要变量
if [ -z "$VENV_PATH" ] || [ -z "$EVAL_OUTPUT_DIR" ]; then
    echo "❌ Error: VENV_PATH or EVAL_OUTPUT_DIR not set."
    exit 1
fi

# 4. 激活环境
source "$VENV_PATH/bin/activate"
source /dfs/data/sbin/setup.sh

# 5. 准备输出目录
mkdir -p "$EVAL_OUTPUT_DIR"
LOG_FILE="$EVAL_OUTPUT_DIR/eval.log"

echo "========================================"
echo "🚀 Starting Evaluation Wrapper"
echo "📂 Output Dir: $EVAL_OUTPUT_DIR"
echo "📝 Logging to: $LOG_FILE"
echo "🐍 Python Script: $PYTHON_SCRIPT"
echo "========================================"

# 6. 执行 Python

cd /
uv run "$PYTHON_SCRIPT" 2>&1 | tee -a "$LOG_FILE"

# 7. 捕获退出状态 (这是 Python 脚本的退出码)
EXIT_CODE=${PIPESTATUS[0]}

# [新增] 8. 计算耗时并格式化
DURATION=$SECONDS
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECS=$((DURATION % 60))
TIME_STR=$(printf "%02d:%02d:%02d" $HOURS $MINUTES $SECS)

echo "----------------------------------------"
echo "⏱️  Total Time: $TIME_STR ($DURATION seconds)"

# 9. 根据退出码决定最终状态
if [ $EXIT_CODE -ne 0 ]; then
    echo "❌ Evaluation Failed (Exit Code: $EXIT_CODE)"
    exit $EXIT_CODE
fi

echo "✅ All Done."