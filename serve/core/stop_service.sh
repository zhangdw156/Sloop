#!/bin/bash
# ================================================================
# Layer 2 Tool: vLLM Stopper
# 用法: bash stop_service.sh <model_name> <port>
# ================================================================

MODEL_NAME=$1
PORT=$2

# 1. 必须跟 Driver 脚本里的路径保持完全一致
LOG_DIR="/dfs/data/work/Sloop/serve/logs"
PID_FILE="$LOG_DIR/${MODEL_NAME}_${PORT}.pid"

echo "🔍 Checking for PID file: $PID_FILE"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    # 检查进程是否存在
    if kill -0 $PID 2>/dev/null; then
        echo "🛑 Stopping vLLM ($MODEL_NAME) on PID $PID ..."
        
        # 1. 尝试优雅退出 (SIGTERM)
        kill $PID
        
        # 2. 循环等待进程结束 (最多等 20秒)
        for i in {1..20}; do
            if ! kill -0 $PID 2>/dev/null; then
                echo "✅ Service stopped gracefully."
                rm -f "$PID_FILE"
                exit 0
            fi
            sleep 1
            echo -n "."
        done
        
        echo ""
        # 3. 如果还在跑，强制杀死 (SIGKILL)
        echo "⚠️  Timeout! Force killing process $PID..."
        kill -9 $PID
        rm -f "$PID_FILE"
        echo "✅ Service force killed."
    else
        echo "⚠️  Process $PID not found running. Cleaning up stale PID file."
        rm -f "$PID_FILE"
    fi
else
    echo "⚠️  No PID file found. Service might be already stopped."
    
    # [兜底逻辑] 万一 PID 文件丢了，尝试通过端口查找并杀掉 (可选)
    # real_pid=$(lsof -t -i:$PORT)
    # if [ -n "$real_pid" ]; then
    #     echo "Found process on port $PORT: $real_pid. Killing..."
    #     kill -9 $real_pid
    # fi
fi
