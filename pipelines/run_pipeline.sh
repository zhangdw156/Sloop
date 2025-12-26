#!/bin/bash

# ================================================================
# Layer 4: Automated Evaluation Pipeline (Robust Version)
# ================================================================

SERVE_RECIPE=$1
EVAL_RECIPE=$2

# --- 0. 环境与参数检查 ---
# (可选) 加载你的平台初始化脚本
if [ -f "/dfs/data/sbin/setup.sh" ]; then
    source /dfs/data/sbin/setup.sh
fi

if [ -z "$SERVE_RECIPE" ] || [ -z "$EVAL_RECIPE" ]; then
    echo "❌ Usage: bash pipelines/run_pipeline.sh <serve_recipe> <eval_recipe>"
    exit 1
fi

# 启用 Errexit: 遇到任何命令报错立即退出
set -e 

# 记录开始时间
START_TIME=$(date +%s)

# --- 1. 定义清理函数 ---
cleanup() {
    # 捕获原始的退出码
    EXIT_CODE=$?
    
    echo ""
    echo "========================================================"
    echo "🧹 Pipeline Teardown..."
    
    if [ -n "$SERVE_MODEL_NAME" ] && [ -n "$SERVE_PORT" ]; then
        STOP_SCRIPT="/dfs/data/work/Sloop/serve/core/stop_service.sh"
        if [ -f "$STOP_SCRIPT" ]; then
            # 临时关闭 set -e，防止停止脚本报错导致 cleanup 中断
            set +e 
            bash "$STOP_SCRIPT" "$SERVE_MODEL_NAME" "$SERVE_PORT"
            set -e
        fi
    fi
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "⏱️ Total Duration: ${DURATION}s"
    
    if [ $EXIT_CODE -ne 0 ]; then
        echo "❌ Pipeline FAILED with exit code $EXIT_CODE"
    else
        echo "✅ Pipeline FINISHED successfully."
    fi
    
    exit $EXIT_CODE
}

# 注册 trap
trap cleanup EXIT INT TERM

# --- 2. 启动服务 ---
echo "========================================================"
echo "🚀 Phase 1: Starting vLLM Service..."
echo "📜 Recipe: $SERVE_RECIPE"
echo "========================================================"

# 强制后台模式
export SERVE_MODE="daemon"

# 加载配方
source "$SERVE_RECIPE"

# 再次检查关键变量是否加载成功
if [ -z "$SERVE_PORT" ]; then
    echo "❌ Error: SERVE_PORT not set. Check your Serve Recipe."
    exit 1
fi

# --- 3. 健康检查 (智能版) ---
API_URL="http://localhost:$SERVE_PORT/v1/models"
PID_FILE="/dfs/data/work/Sloop/serve/logs/${SERVE_MODEL_NAME}_${SERVE_PORT}.pid"

echo "⏳ Waiting for service at $API_URL ..."
echo "   Checking PID file: $PID_FILE"

MAX_RETRIES=120
COUNTER=0

while true; do
    # [核心改进] 检查 PID 进程是否还活着
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ! kill -0 $PID 2>/dev/null; then
            echo ""
            echo "❌ CRITICAL: vLLM process (PID $PID) died unexpectedly!"
            echo "   Check logs immediately: /dfs/data/work/Sloop/serve/logs/${SERVE_MODEL_NAME}_${SERVE_PORT}.log"
            exit 1
        fi
    fi

    # 检查服务端口
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL" || echo "000")
    
    if [ "$HTTP_CODE" == "200" ]; then
        echo "✅ Service is UP and READY!"
        break
    fi
    
    sleep 2
    COUNTER=$((COUNTER+1))
    
    if [ $COUNTER -ge $MAX_RETRIES ]; then
        echo ""
        echo "❌ Timeout: Service failed to start within ${MAX_RETRIES}s."
        exit 1
    fi
    echo -n "."
done

# --- 4. 运行评测 ---
echo ""
echo "========================================================"
echo "🧪 Phase 2: Running Evaluation..."
echo "📜 Recipe: $EVAL_RECIPE"
echo "========================================================"

# 运行评测
bash "$EVAL_RECIPE"

# 脚本自然结束，触发 trap cleanup (exit code 0)