#!/bin/bash

source /dfs/data/uv-venv/gorilla/bin/activate

vllm serve /dfs/data/models/qwen3-8b-looptool-sft \
  --served-model-name Qwen3-8B-LoopTool \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --port 1053