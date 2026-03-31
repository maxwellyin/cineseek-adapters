from __future__ import annotations

from pathlib import Path
import os
import random

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = ROOT_DIR.parent / "retrieval-system" / "data" / "processed" / "msrd_text2item_dataset.pt"
DATASET_PATH = Path(os.environ.get("CINESEEK_DATASET_PATH", DEFAULT_DATASET_PATH)).expanduser().resolve()
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
CHECKPOINT_DIR = ARTIFACTS_DIR / "checkpoints"
EXPERIMENTS_DIR = ROOT_DIR / "experiments"
RANDOM_SEED = 7


def get_device() -> torch.device:
    forced = os.environ.get("CINESEEK_ADAPTER_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_directories() -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

