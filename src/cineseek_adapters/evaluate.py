from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from cineseek_adapters.config import DATASET_PATH, get_device
from cineseek_adapters.data import build_item_embeddings, get_positive_ids, get_query_embeddings, load_dataset
from cineseek_adapters.metrics import ranking_metrics
from cineseek_adapters.models import build_adapter, count_trainable_parameters


def load_adapter_from_checkpoint(path: Path, dim: int, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    adapter = build_adapter(
        checkpoint["adapter"],
        dim=dim,
        hidden_dim=checkpoint.get("hidden_dim", 768),
        dropout=checkpoint.get("dropout", 0.0),
        residual_scale=checkpoint.get("residual_scale", 0.2),
    ).to(device)
    adapter.load_state_dict(checkpoint["state_dict"])
    return adapter, checkpoint


@torch.no_grad()
def encode_in_batches(adapter, matrix: torch.Tensor, batch_size: int, device: torch.device) -> torch.Tensor:
    outputs = []
    adapter.eval()
    for start in range(0, matrix.shape[0], batch_size):
        batch = matrix[start : start + batch_size].to(device)
        outputs.append(adapter(batch).cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def evaluate(adapter, query_embeddings, item_embeddings, positive_ids, batch_size: int, device: torch.device, k: int):
    encode_start = time.perf_counter()
    query_repr = encode_in_batches(adapter, query_embeddings, batch_size=batch_size, device=device)
    item_repr = encode_in_batches(adapter, item_embeddings, batch_size=batch_size, device=device)
    encode_seconds = time.perf_counter() - encode_start

    search_start = time.perf_counter()
    scores = query_repr @ item_repr.T
    ranked = torch.topk(scores, k=min(k, item_repr.shape[0]), dim=1).indices + 1
    search_seconds = time.perf_counter() - search_start

    metrics = ranking_metrics([[int(idx) for idx in row.tolist()] for row in ranked], positive_ids)
    metrics["avg_encode_ms"] = 1000.0 * encode_seconds / max(query_embeddings.shape[0], 1)
    metrics["avg_search_ms"] = 1000.0 * search_seconds / max(query_embeddings.shape[0], 1)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--adapter", choices=["identity", "linear", "residual_mlp"], default="identity")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--item-mode", choices=["title", "overview", "title_overview_avg"], default="title_overview_avg")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--residual-scale", type=float, default=0.2)
    args = parser.parse_args()

    device = get_device()
    dataset = load_dataset(args.dataset)
    item_embeddings = build_item_embeddings(dataset, mode=args.item_mode)
    query_embeddings = get_query_embeddings(dataset, args.split)
    positives = get_positive_ids(dataset, args.split)
    dim = int(item_embeddings.shape[1])

    if args.checkpoint:
        adapter, checkpoint = load_adapter_from_checkpoint(args.checkpoint, dim=dim, device=device)
        adapter_name = checkpoint["adapter"]
    else:
        adapter = build_adapter(
            args.adapter,
            dim=dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            residual_scale=args.residual_scale,
        ).to(device)
        adapter_name = args.adapter

    metrics = evaluate(adapter, query_embeddings, item_embeddings, positives, args.batch_size, device, args.k)
    payload = {
        "adapter": adapter_name,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "split": args.split,
        "item_mode": args.item_mode,
        "dataset_items": int(item_embeddings.shape[0]),
        "num_queries": int(query_embeddings.shape[0]),
        "trainable_parameters": count_trainable_parameters(adapter),
        **metrics,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

