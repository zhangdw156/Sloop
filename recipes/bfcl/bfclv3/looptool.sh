#!/bin/bash

source /dfs/data/sbin/setup.sh
source /dfs/data/uv-venv/gorilla/bin/activate

bfcl generate \
    --model Qwen3-8B-LoopTool \
    --test-category all \
    --skip-server-setup \
    --local-tokenizer-path /dfs/data/models/Qwen3-0.6B
