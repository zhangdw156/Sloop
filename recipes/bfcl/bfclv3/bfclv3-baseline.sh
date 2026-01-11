#!/bin/bash

source /dfs/data/sbin/setup.sh
source /dfs/data/uv-venv/gorilla/bin/activate

bfcl generate \
    --model Qwen3-8B \
    --test-category multi_turn_base \
    --skip-server-setup \
    --local-tokenizer-path /dfs/data/models/Qwen3-0.6B

bfcl evaluate \
    --model Qwen3-8B \
    --test-category multi_turn_base
