from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cineseek_adapters.config import CHECKPOINT_DIR, DATASET_PATH, ensure_directories, get_device, seed_everything
from cineseek_adapters.data import build_item_embeddings, get_positive_ids, get_query_embeddings, load_dataset
from cineseek_adapters.evaluate import evaluate
from cineseek_adapters.models import build_adapter, count_trainable_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--adapter", choices=["linear", "residual_mlp", "concat_linear"], default="residual_mlp")
    parser.add_argument("--item-mode", choices=["title", "overview", "title_overview_avg", "title_overview_concat"], default="title_overview_avg")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--residual-scale", type=float, default=0.2)
    parser.add_argument("--title-weight", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--max-train-examples", type=int, default=0, help="Optional cap for quick experiments.")
    parser.add_argument("--output", type=Path, default=CHECKPOINT_DIR / "residual_mlp.pt")
    parser.add_argument("--metrics-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    ensure_directories()
    seed_everything(args.seed)
    device = get_device()
    dataset = load_dataset(args.dataset)

    item_embeddings = build_item_embeddings(dataset, mode=args.item_mode)
    train_queries = dataset["train_query_embeddings"].float()
    train_targets = dataset["train_target_ids"].long() - 1
    if args.max_train_examples and args.max_train_examples > 0:
        train_queries = train_queries[: args.max_train_examples]
        train_targets = train_targets[: args.max_train_examples]

    query_dim = int(train_queries.shape[1])
    item_dim = int(item_embeddings.shape[1])
    adapter = build_adapter(
        args.adapter,
        dim=query_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        residual_scale=args.residual_scale,
        item_dim=item_dim,
        title_weight=args.title_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(TensorDataset(train_queries, train_targets), batch_size=args.batch_size, shuffle=True)
    item_embeddings_device = item_embeddings.to(device)

    best_val_mrr = -1.0
    best_payload = None
    history = []
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        total_loss = 0.0
        for queries, targets in tqdm(loader, desc=f"epoch {epoch}"):
            queries = queries.to(device)
            targets = targets.to(device)
            query_repr = adapter.encode_query(queries)
            item_repr = adapter.encode_item(item_embeddings_device)
            logits = (query_repr @ item_repr.T) / args.temperature
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())

        val_metrics = evaluate(
            adapter,
            get_query_embeddings(dataset, "val"),
            item_embeddings,
            get_positive_ids(dataset, "val"),
            batch_size=args.eval_batch_size,
            device=device,
            k=100,
        )
        payload = {
            "epoch": epoch,
            "train_loss": total_loss / max(len(loader), 1),
            "val": val_metrics,
            "trainable_parameters": count_trainable_parameters(adapter),
        }
        history.append(payload)
        print(json.dumps(payload, indent=2))

        if val_metrics["mrr"] > best_val_mrr:
            best_val_mrr = val_metrics["mrr"]
            best_payload = payload
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "adapter": args.adapter,
                    "state_dict": adapter.state_dict(),
                    "item_mode": args.item_mode,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                    "residual_scale": args.residual_scale,
                    "item_dim": item_dim,
                    "title_weight": args.title_weight,
                    "temperature": args.temperature,
                    "best": best_payload,
                },
                args.output,
            )

    final_payload = {
        "adapter": args.adapter,
        "item_mode": args.item_mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "residual_scale": args.residual_scale,
        "item_dim": item_dim,
        "title_weight": args.title_weight,
        "temperature": args.temperature,
        "max_train_examples": args.max_train_examples,
        "saved_checkpoint": str(args.output),
        "best": best_payload,
        "history": history,
    }
    if args.metrics_output:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(final_payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final_payload, indent=2))


if __name__ == "__main__":
    main()
