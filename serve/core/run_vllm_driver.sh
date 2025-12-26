#!/bin/bash
# ================================================================
# Layer 2: vLLM Driver Script (Enhanced for LoRA)
# 负责接收环境变量，组装命令，启动 vLLM Server
# ================================================================

# =======================================================
# Environment Setup
# =======================================================

# 1. 确定虚拟环境路径 (优先用 Layer 3 传进来的，没有就用默认的)
TARGET_VENV=${VENV_PATH:-"/dfs/data/uv-venv/modelscope"}

# 2. 检查当前是否已经在 venv 里了 (防止重复激活)
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "$TARGET_VENV/bin/activate" ]; then
        echo "🔌 Activating Venv: $TARGET_VENV"
        source "$TARGET_VENV/bin/activate"
    else
        echo "⚠️  Warning: Venv not found at $TARGET_VENV. Using system python."
    fi
else
    echo "✅ Already in venv: $VIRTUAL_ENV"
fi

# 3. 检查 uv (可选，如果你想确保 uv 命令可用)
if ! command -v uv &> /dev/null; then
    echo "⚠️  Warning: 'uv' command not found."
fi
# =======================================================

# 1. 检查核心环境变量
if [ -z "$SERVE_MODEL_PATH" ]; then
    echo "❌ Error: SERVE_MODEL_PATH is not set."
    exit 1
fi

# 2. 设置默认值
PORT=${SERVE_PORT:-8000}
HOST=${SERVE_HOST:-"0.0.0.0"}
TP_SIZE=${SERVE_TP_SIZE:-1}
MAX_LEN=${SERVE_MAX_LEN:-32768}
GPU_UTIL=${SERVE_GPU_UTIL:-0.90}
DTYPE=${SERVE_DTYPE:-"bfloat16"}
TOOL_PARSER=${SERVE_TOOL_PARSER:-"hermes"}
SWAP_SPACE=${SERVE_SWAP_SPACE:-4}  # [新增] 默认 4GB Swap

# 3. 准备日志路径
LOG_DIR="/dfs/data/work/Sloop/serve/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/${SERVE_MODEL_NAME}_${PORT}.log"

# 4. 激活环境
if [ -z "$VIRTUAL_ENV" ]; then
    export VENV_PATH="/dfs/data/uv-venv/modelscope"
    if [ -f "$VENV_PATH/bin/activate" ]; then
        source "$VENV_PATH/bin/activate"
    fi
fi

# 5. [新增] 构建 LoRA 参数逻辑
LORA_ARGS=""
if [ "$SERVE_ENABLE_LORA" == "true" ]; then
    echo "🧩 LoRA Enabled."
    LORA_ARGS="--enable-lora --max-lora-rank ${SERVE_MAX_LORA_RANK:-64}"
    
    if [ -n "$SERVE_LORA_MODULES" ]; then
        # 注意：这里不需要加引号，以便让 shell 正确拆分多个 module
        LORA_ARGS="$LORA_ARGS --lora-modules $SERVE_LORA_MODULES"
    fi
fi

# 6. 组装启动命令
CMD="uv run --no-project -m vllm.entrypoints.openai.api_server \
    --model $SERVE_MODEL_PATH \
    --served-model-name $SERVE_MODEL_NAME \
    --trust-remote-code \
    --host $HOST \
    --port $PORT \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len $MAX_LEN \
    --gpu-memory-utilization $GPU_UTIL \
    --swap-space $SWAP_SPACE \
    --dtype $DTYPE \
    --enable-auto-tool-choice \
    --tool-call-parser $TOOL_PARSER \
    $LORA_ARGS"  # 追加 LoRA 参数

echo "========================================================"
echo "🚀 Starting vLLM Service..."
echo "🤖 Base Model: $SERVE_MODEL_NAME"
echo "🧩 LoRA:       ${SERVE_ENABLE_LORA:-false}"
echo "📂 Path:       $SERVE_MODEL_PATH"
echo "🔌 Port:       $PORT"
echo "📝 Log:        $LOG_FILE"
echo "========================================================"

# 7. 启动模式
if [ "$SERVE_MODE" == "daemon" ]; then
    nohup $CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "✅ vLLM started in background. PID: $PID"
    echo $PID > "$LOG_DIR/${SERVE_MODEL_NAME}_${PORT}.pid"
else
    echo "⚠️  Running in FOREGROUND mode..."
    $CMD
fi
