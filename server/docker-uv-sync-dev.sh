#!/usr/bin/env bash
# Fail-closed DEV sync. Skips the missing local seamless path and CUDA/torch
# wheels so the HTTP surface can boot. Never mutates uv.lock.
set -euo pipefail

exec uv sync --python 3.12 --frozen --no-dev \
  --no-install-package seamless-communication \
  --no-install-package simuleval \
  --no-install-package bitarray \
  --no-install-package openai-whisper \
  --no-install-package textgrid \
  --no-install-package torch \
  --no-install-package torchaudio \
  --no-install-package transformers \
  --no-install-package fairseq2 \
  --no-install-package fairseq2n \
  --no-install-package nvidia-cublas-cu12 \
  --no-install-package nvidia-cuda-cupti-cu12 \
  --no-install-package nvidia-cuda-nvrtc-cu12 \
  --no-install-package nvidia-cuda-runtime-cu12 \
  --no-install-package nvidia-cudnn-cu12 \
  --no-install-package nvidia-cufft-cu12 \
  --no-install-package nvidia-cufile-cu12 \
  --no-install-package nvidia-curand-cu12 \
  --no-install-package nvidia-cusolver-cu12 \
  --no-install-package nvidia-cusparse-cu12 \
  --no-install-package nvidia-cusparselt-cu12 \
  --no-install-package nvidia-nccl-cu12 \
  --no-install-package nvidia-nvjitlink-cu12 \
  --no-install-package nvidia-nvshmem-cu12 \
  --no-install-package nvidia-nvtx-cu12 \
  --no-install-package triton \
  --no-install-package torcheval \
  "$@"
