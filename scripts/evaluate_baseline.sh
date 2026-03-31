#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src python -m cineseek_adapters.evaluate --adapter identity --split val
PYTHONPATH=src python -m cineseek_adapters.evaluate --adapter identity --split test

