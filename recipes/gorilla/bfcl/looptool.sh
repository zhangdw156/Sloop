#!/bin/bash

# ================= 0. 路径定位 =================
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# ================= 配置区域 =================

# 1. 基础环境
export SETUP_SCRIPT="/dfs/data/sbin/setup.sh"
export VENV_ACTIVATE="/dfs/data/uv-venv/gorilla/bin/activate"

# 2. 项目与输出
export PROJECT_ROOT="/dfs/data/work/gorilla/berkeley-function-call-leaderboard"
export OUTPUT_DIR_NAME="lootool" 
# 拼接最终的产物路径，传递给 Layer 2 使用
export ARTIFACT_DIR="$PROJECT_ROOT/$OUTPUT_DIR_NAME"

# 3. 模型配置
export MODEL_NAME="Qwen/Qwen3-8B-FC"
export LOCAL_MODEL_PATH="/dfs/data/models/Qwen3-8B"

# 4. LoRA 配置
export ENABLE_LORA="true"
export MAX_LORA_RANK=128
export LORA_MODULES="bfclv3=/dfs/data/work/Sloop/checkpoints/swift-qwen3-looptool-.v1.1766635255394-20251225_1201/v0-20251225-120127/checkpoint-86"

# 5. 评测参数
export TEST_CATEGORY="multi_turn"
export THREADS=32
export GPU_MEM_UTIL=0.9

# 6. 核心脚本
export CORE_SCRIPT="${SCRIPT_DIR}/run_task.sh"

# ================= 启动 =================
echo "📋 Configuration loaded for: $MODEL_NAME"
if [ "$ENABLE_LORA" == "true" ]; then
    echo "🧩 LoRA Enabled: $LORA_MODULES"
fi
echo "📂 Direct Output Path: $ARTIFACT_DIR/result"
echo "📂 Direct Score Path:  $ARTIFACT_DIR/score"

if [ ! -f "$CORE_SCRIPT" ]; then
    echo "❌ Error: Core script not found at $CORE_SCRIPT"
    exit 1
fi
chmod +x "$CORE_SCRIPT"

"$CORE_SCRIPT"