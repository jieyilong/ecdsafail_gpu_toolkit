#!/usr/bin/env bash
# Runs ON the GPU machine. Reports the GPU/CUDA environment.
export PATH="$PATH:/usr/local/cuda/bin:/opt/cuda/bin"
echo "host: $(hostname 2>/dev/null) | user: $(whoami 2>/dev/null)"
if command -v nvcc >/dev/null; then echo "nvcc: $(nvcc --version | grep -oE 'release [0-9.]+' | head -1)"; else echo "nvcc: NOT FOUND (install CUDA toolkit)"; fi
if command -v nvidia-smi >/dev/null; then
  n=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')
  echo "GPUs: $n"
  nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader | sed 's/^/  /'
else echo "nvidia-smi: NOT FOUND (no NVIDIA driver?)"; fi
