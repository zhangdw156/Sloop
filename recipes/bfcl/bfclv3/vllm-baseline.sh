#!/bin/bash

source /dfs/data/uv-venv/gorilla/bin/activate

vllm serve /dfs/data/models/Qwen3-8B \
  --served-model-name Qwen3-8B \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --port 1053
