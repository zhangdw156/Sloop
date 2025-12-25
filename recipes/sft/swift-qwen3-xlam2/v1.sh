#!/bin/bash
export RECIPE_NAME="$(basename "${BASH_SOURCE[0]}" .sh)"

# 只改核心超参
# export LR="2e-5"
# export EPOCHS=4
# export BATCH_SIZE=4

CURRENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "$CURRENT_DIR/run_task.sh"
