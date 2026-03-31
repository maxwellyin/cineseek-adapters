from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from cineseek_adapters.config import DATASET_PATH


def load_dataset(path: Path = DATASET_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. Set CINESEEK_DATASET_PATH or keep retrieval-system next to this repo."
        )
    return torch.load(path, map_location="cpu")


def normalize(embeddings: torch.Tensor) -> torch.Tensor:
    return F.normalize(embeddings.float(), dim=-1)


def build_item_embeddings(dataset: dict, mode: str = "title_overview_avg") -> torch.Tensor:
    if mode == "title":
        return normalize(dataset["item_title_embeddings"][1:])
    if mode == "overview":
        return normalize(dataset["item_overview_embeddings"][1:])
    if mode == "title_overview_avg":
        title = normalize(dataset["item_title_embeddings"][1:])
        overview = normalize(dataset["item_overview_embeddings"][1:])
        return normalize((title + overview) / 2.0)
    raise ValueError(f"Unsupported item embedding mode: {mode}")


def get_positive_ids(dataset: dict, split: str) -> list[list[int]]:
    key = f"{split}_positive_ids"
    if key in dataset:
        return [[int(item_id) for item_id in ids] for ids in dataset[key]]
    target_key = f"{split}_target_ids"
    return [[int(item_id)] for item_id in dataset[target_key].tolist()]


def get_query_embeddings(dataset: dict, split: str) -> torch.Tensor:
    return normalize(dataset[f"{split}_query_embeddings"])

