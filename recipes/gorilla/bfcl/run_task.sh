#!/bin/bash
set -e  # 遇到任何错误立即退出

# ================= 函数定义 =================
log() {
    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] 🚀 $1"
}

# ================= 0. 检查必要变量 =================
if [ -z "$PROJECT_ROOT" ] || [ -z "$MODEL_NAME" ]; then
    echo "❌ Error: Environment variables are not set. Please run run_task.sh instead."
    exit 1
fi

# ================= 1. 环境加载 =================
log "Loading environment..."

if [ -f "$SETUP_SCRIPT" ]; then
    source "$SETUP_SCRIPT"
else
    echo "⚠️ Warning: Setup script not found at $SETUP_SCRIPT"
fi

if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
else
    echo "❌ Error: Virtualenv not found at $VENV_ACTIVATE"
    exit 1
fi

# 切换到项目目录，确保 bfcl 生成的文件在预期位置
cd "$PROJECT_ROOT" || { echo "❌ Cannot cd to $PROJECT_ROOT"; exit 1; }

# ================= 2. 自动计算 GPU 数量 =================
log "Detecting GPUs..."

if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    echo "🔍 Detected CUDA_VISIBLE_DEVICES set. Using visible GPUs count: $NUM_GPUS"
elif command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "🔍 Detected physical GPUs via nvidia-smi: $NUM_GPUS"
else
    echo "⚠️ nvidia-smi not found. Defaulting to 1 GPU."
    NUM_GPUS=1
fi

# ================= 3. 执行 BFCL Generate =================
log "Running BFCL Generate ($MODEL_NAME)..."

bfcl generate \
  --model "$MODEL_NAME" \
  --test-category "$TEST_CATEGORY" \
  --backend vllm \
  --num-gpus "$NUM_GPUS" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --local-model-path "$LOCAL_MODEL_PATH" \
  --num-threads "$THREADS"

# ================= 4. 执行 BFCL Evaluate =================
log "Running BFCL Evaluate..."

bfcl evaluate \
  --model "$MODEL_NAME" \
  --test-category "$EVAL_CATEGORY" \
  --partial-eval

# ================= 5. 结果归档 =================
log "Archiving results..."

TARGET_DIR="$PROJECT_ROOT/$OUTPUT_DIR_NAME"

# 创建目标文件夹
if [ ! -d "$TARGET_DIR" ]; then
    mkdir -p "$TARGET_DIR"
    echo "Created directory: $TARGET_DIR"
fi

# 移动结果
# 注意：这里加了检查，防止文件夹不存在导致报错
if [ -d "result" ]; then
    # 使用 cp -r 然后 rm 的方式比直接 mv 更安全，特别是跨文件系统时，
    # 但为了保持原逻辑，这里使用 mv。
    # 为了防止覆盖，如果目标里面已经有 result，建议重命名或清除。
    # 这里采用覆盖/合并模式：
    echo "Moving 'result' to $TARGET_DIR..."
    rm -rf "$TARGET_DIR/result" # 清除旧的 result 防止 mv 报错或嵌套
    mv result "$TARGET_DIR/"
else
    echo "⚠️ Warning: 'result' directory not found."
fi

if [ -d "score" ]; then
    echo "Moving 'score' to $TARGET_DIR..."
    rm -rf "$TARGET_DIR/score" # 清除旧的 score
    mv score "$TARGET_DIR/"
else
    echo "⚠️ Warning: 'score' directory not found."
fi

log "Done! All tasks completed successfully."