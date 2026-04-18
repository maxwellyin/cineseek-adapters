#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m cineseek_adapters.train \
  --adapter concat_linear \
  --item-mode title_overview_concat \
  --epochs 5 \
  --batch-size 256 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --title-weight 0.5 \
  --output artifacts/checkpoints/concat_linear.pt \
  --metrics-output experiments/concat_linear_train.json

PYTHONPATH=src python -m cineseek_adapters.evaluate \
  --checkpoint artifacts/checkpoints/concat_linear.pt \
  --split test \
  --batch-size 1024 > experiments/concat_linear_test.json
