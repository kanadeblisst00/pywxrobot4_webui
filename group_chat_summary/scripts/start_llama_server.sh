#!/usr/bin/env bash
set -euo pipefail

llama_binary="${LLAMA_SERVER_BIN:-llama-server}"
model_path="${LLAMA_MODEL_PATH:-}"
hf_repo="${LLAMA_HF_REPO:-unsloth/Qwen3.5-9B-GGUF:Q4_K_M}"
model_alias="${LLAMA_MODEL_ALIAS:-qwen3.5-9b}"
llama_port="${LLAMA_PORT:-18080}"
context_size="${LLAMA_CONTEXT_SIZE:-32768}"
gpu_layers="${LLAMA_GPU_LAYERS:-99}"
parallel_slots="${LLAMA_PARALLEL:-1}"

if ! command -v "$llama_binary" >/dev/null 2>&1 && [[ ! -x "$llama_binary" ]]; then
  echo "未找到 llama-server；请安装 llama.cpp 或设置 LLAMA_SERVER_BIN。" >&2
  exit 1
fi

model_args=(--hf-repo "$hf_repo")
if [[ -n "$model_path" ]]; then
  if [[ ! -f "$model_path" ]]; then
    echo "GGUF 模型不存在：$model_path" >&2
    exit 1
  fi
  model_args=(--model "$model_path")
fi

echo "正在启动 llama.cpp：$model_alias -> http://127.0.0.1:$llama_port/v1"
exec "$llama_binary" \
  "${model_args[@]}" \
  --alias "$model_alias" \
  --host 127.0.0.1 \
  --port "$llama_port" \
  --ctx-size "$context_size" \
  --parallel "$parallel_slots" \
  --n-gpu-layers "$gpu_layers" \
  --jinja \
  --no-mmproj
