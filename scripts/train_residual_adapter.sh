#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m cineseek_adapters.train \
  --adapter residual_mlp \
  --epochs 5 \
  --batch-size 256 \
  --hidden-dim 768 \
  --lr 1e-4 \
  --output artifacts/checkpoints/residual_mlp.pt

