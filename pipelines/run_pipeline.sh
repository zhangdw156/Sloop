#!/bin/bash

# ================================================================
# Layer 4: Automated Evaluation Pipeline
# 功能: Serve -> Wait -> Eval -> Stop
# 用法: bash run_pipeline.sh <serve_recipe> <eval_recipe>
# ================================================================

SERVE_RECIPE=$1
EVAL_RECIPE=$2

# --- 0. 参数检查 ---
if [ -z "$SERVE_RECIPE" ] || [ -z "$EVAL_RECIPE" ]; then
    echo "❌ Usage: bash pipelines/run_pipeline.sh <serve_recipe> <eval_recipe>"
    echo "   Ex: bash pipelines/run_pipeline.sh recipes/serve/start_lora.sh recipes/eval/eval_lora.sh"
    exit 1
fi

# 记录开始时间
START_TIME=$(date +%s)

# --- 1. 定义清理函数 (Teardown) ---
# 无论脚本如何退出，都会执行这个函数
cleanup() {
    echo ""
    echo "========================================================"
    echo "🧹 Pipeline Teardown: Stopping Service..."
    
    # 这里的变量来自于下面的 source 操作
    if [ -n "$SERVE_MODEL_NAME" ] && [ -n "$SERVE_PORT" ]; then
        STOP_SCRIPT="/dfs/data/work/Sloop/serve/core/stop_service.sh"
        if [ -f "$STOP_SCRIPT" ]; then
            bash "$STOP_SCRIPT" "$SERVE_MODEL_NAME" "$SERVE_PORT"
        else
            echo "⚠️ Warning: Stop script not found at $STOP_SCRIPT"
            echo "   You may need to manually kill vLLM on port $SERVE_PORT"
        fi
    else
        echo "⚠️ Warning: Service info missing. Manual cleanup might be required."
    fi
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "⏱️ Total Pipeline Duration: ${DURATION}s"
    echo "✅ Pipeline Finished."
    echo "========================================================"
}

# 注册信号捕获: 遇到 EXIT(正常退出), INT(Ctrl+C), TERM(kill) 时执行 cleanup
trap cleanup EXIT INT TERM

# --- 2. 启动服务 (Serve Phase) ---
echo "========================================================"
echo "🚀 Phase 1: Starting vLLM Service..."
echo "📜 Recipe: $SERVE_RECIPE"
echo "========================================================"

# [关键] 强制设置为后台模式，覆盖 Recipe 里的设置
# 这样 source Recipe 时，driver 会在后台启动并写入 PID 文件，而不是卡住当前脚本
export SERVE_MODE="daemon"

# 加载配方 (这会触发 Driver 启动服务)
source "$SERVE_RECIPE"

# --- 3. 健康检查 (Health Check) ---
# 此时 $SERVE_PORT 已经被 source 进来了
API_URL="http://localhost:$SERVE_PORT/v1/models"
echo "⏳ Waiting for service at $API_URL ..."

MAX_RETRIES=120  # 等待 120秒 (模型加载可能慢)
COUNTER=0

while true; do
    # 使用 curl 检查服务状态 (-s 静默, -o 丢弃输出, -w 返回状态码)
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL")
    
    if [ "$HTTP_CODE" == "200" ]; then
        echo "✅ Service is UP and READY!"
        break
    fi
    
    sleep 2
    COUNTER=$((COUNTER+1))
    
    if [ $COUNTER -ge $MAX_RETRIES ]; then
        echo "❌ Timeout: Service failed to start within ${MAX_RETRIES}s."
        echo "   Check logs at: /dfs/data/work/Sloop/serve/logs/"
        exit 1
    fi
    echo -n "."
done

# --- 4. 运行评测 (Eval Phase) ---
echo ""
echo "========================================================"
echo "🧪 Phase 2: Running Evaluation..."
echo "📜 Recipe: $EVAL_RECIPE"
echo "========================================================"

# 运行评测脚本
bash "$EVAL_RECIPE"

# 脚本运行到这里结束，会自动触发 trap cleanup