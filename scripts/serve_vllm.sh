#!/usr/bin/env bash
# Serve a model with an OpenAI-compatible API on one GPU.
# Usage: GPU=1 bash scripts/serve_vllm.sh [model] [port]
set -euo pipefail

MODEL=${1:-Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=${2:-8000}

# FlashInfer's sampler JIT-compiles with the system nvcc, which is too old here; use the PyTorch sampler.
export VLLM_USE_FLASHINFER_SAMPLER=0

CUDA_VISIBLE_DEVICES=${GPU:-0} vllm serve "$MODEL" \
  --port "$PORT" \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85
