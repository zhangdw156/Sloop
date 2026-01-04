#!/bin/bash
set -e

# ================= 函数定义 =================
log() {
    echo -e "\n[$(date +'%Y-%m-%d %H:%M:%S')] 🚀 $1"
}

# ================= 0. 检查变量 =================
if [ -z "$PROJECT_ROOT" ] || [ -z "$MODEL_NAME" ] || [ -z "$ARTIFACT_DIR" ]; then
    echo "❌ Error: Necessary variables (PROJECT_ROOT, MODEL_NAME, ARTIFACT_DIR) are missing."
    exit 1
fi

# ================= 1. 环境加载 =================
log "Loading environment..."

if [ -f "$SETUP_SCRIPT" ]; then 
    source "$SETUP_SCRIPT"; 
fi
if [ -f "$VENV_ACTIVATE" ]; then 
    source "$VENV_ACTIVATE"
else
    echo "❌ Error: Virtualenv not found."
    exit 1
fi

# 虽然指定了输出目录，还是建议 cd 过去，防止有些临时文件乱跑
cd "$PROJECT_ROOT" || { echo "❌ Cannot cd to $PROJECT_ROOT"; exit 1; }

# ================= 2. 显卡检测 =================
log "Detecting GPUs..."
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    echo "🔍 Using CUDA_VISIBLE_DEVICES count: $NUM_GPUS"
elif command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "🔍 Using nvidia-smi physical count: $NUM_GPUS"
else
    NUM_GPUS=1
fi

# ================= 3. BFCL Generate =================
log "Running BFCL Generate ($MODEL_NAME)..."

# 确保输出目录存在 (BFCL 可能不会自动创建多级父目录)
mkdir -p "$ARTIFACT_DIR/result"

GEN_CMD=(
    bfcl generate
    --model "$MODEL_NAME"
    --test-category "$TEST_CATEGORY"
    --backend vllm
    --num-gpus "$NUM_GPUS"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --local-model-path "$LOCAL_MODEL_PATH"
    --num-threads "$THREADS"
    --result-dir "$ARTIFACT_DIR/result"  # <--- 直接指定输出目录
)

if [ "$ENABLE_LORA" == "true" ]; then
    log "🧩 Appending LoRA arguments..."
    GEN_CMD+=( --enable-lora )
    if [ -n "$MAX_LORA_RANK" ]; then GEN_CMD+=( --max-lora-rank "$MAX_LORA_RANK" ); fi
    if [ -n "$LORA_MODULES" ]; then GEN_CMD+=( --lora-modules $LORA_MODULES ); fi
fi

echo "Executing Generate Command..."
"${GEN_CMD[@]}"

# ================= 4. BFCL Evaluate =================
log "Running BFCL Evaluate..."

# 确保分数目录存在
mkdir -p "$ARTIFACT_DIR/score"

bfcl evaluate \
  --model "$MODEL_NAME" \
  --test-category "$TEST_CATEGORY" \
  --partial-eval \
  --result-dir "$ARTIFACT_DIR/result" \
  --score-dir "$ARTIFACT_DIR/score"     # <--- 直接指定分数输出目录

log "✅ Done! Results and Scores are located in: $ARTIFACT_DIR"