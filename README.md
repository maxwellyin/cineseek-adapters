# CineSeek-Adapters: Lightweight Retrieval Adaptation

CineSeek-Adapters is a small training-focused project built on top of the original CineSeek retrieval dataset. It tests whether lightweight trainable adapters can improve a strong frozen sentence-transformer retrieval baseline without turning the product demo repo back into a training-heavy codebase.

The project is intentionally separate from the deployed CineSeek web demo:

| Project | Role |
| --- | --- |
| CineSeek | Product-facing semantic movie search demo with FAISS, FastAPI, agent reranking, and Docker deployment |
| CineSeek-MM | Multimodal retrieval over movie metadata and posters using CLIP-style embeddings |
| CineSeek-Adapters | Training and ablation project for lightweight retrieval adaptation |

## Research Question

Can a small residual adapter improve frozen sentence-transformer movie retrieval under a realistic small-data setting?

The key comparison is not just final recall. The project tracks whether extra trainable capacity actually helps after controlling for:

- raw frozen embedding baseline
- validation performance
- overfitting risk
- trainable parameter count
- retrieval latency

## Methods

- **Frozen baseline**: title/overview sentence-transformer embeddings from CineSeek.
- **Linear adapter**: one trainable linear residual projection.
- **Residual MLP adapter**: LayerNorm + bottleneck MLP + residual connection.
- **Training objective**: full-catalog contrastive classification over all movie items.
- **Evaluation**: recall@10, recall@50, recall@100, MRR, NDCG.

## Repository Layout

```text
cineseek-adapters/
├── README.md
├── requirements.txt
├── src/cineseek_adapters/
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── metrics.py
│   ├── models.py
│   └── train.py
├── scripts/
│   ├── evaluate_baseline.sh
│   └── train_residual_adapter.sh
├── experiments/
└── artifacts/checkpoints/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

By default, this repo expects the original CineSeek processed dataset at:

```text
../retrieval-system/data/processed/msrd_text2item_dataset.pt
```

You can override it:

```bash
export CINESEEK_DATASET_PATH=/path/to/msrd_text2item_dataset.pt
```

## Run Baseline

```bash
PYTHONPATH=src python -m cineseek_adapters.evaluate --adapter identity --split val
PYTHONPATH=src python -m cineseek_adapters.evaluate --adapter identity --split test
```

This should reproduce the strong raw sentence-transformer baseline from CineSeek.

## Train Residual Adapter

```bash
PYTHONPATH=src python -m cineseek_adapters.train \
  --adapter residual_mlp \
  --epochs 5 \
  --batch-size 256 \
  --hidden-dim 768 \
  --lr 1e-4 \
  --output artifacts/checkpoints/residual_mlp.pt
```

Evaluate the checkpoint:

```bash
PYTHONPATH=src python -m cineseek_adapters.evaluate \
  --checkpoint artifacts/checkpoints/residual_mlp.pt \
  --split test
```

## Smoke Test

For a quick pipeline check without full training:

```bash
PYTHONPATH=src python -m cineseek_adapters.train \
  --adapter residual_mlp \
  --epochs 1 \
  --batch-size 128 \
  --hidden-dim 128 \
  --max-train-examples 512 \
  --output artifacts/checkpoints/smoke_residual_mlp.pt
```

## Expected Interpretation

The raw sentence-transformer baseline is already strong. A useful adapter result is not guaranteed. That is the point of the project: it shows disciplined model adaptation and ablation rather than forcing a more complex model into the deployed CineSeek demo.

Strong outcomes:

- adapter improves recall/MRR without hurting latency much
- adapter only improves recall but hurts MRR, indicating ranking tradeoffs
- adapter fails to beat baseline, showing that the frozen representation is already well matched to the task

Any of these outcomes are valid if they are measured cleanly.

## Resume Framing

**CineSeek-Adapters: Lightweight Retrieval Adaptation**

- Built a PyTorch training and evaluation pipeline for residual adapter tuning over frozen sentence-transformer retrieval embeddings.
- Compared raw embeddings, linear projection, and residual MLP adapters using recall@k, MRR, NDCG, trainable parameter count, and latency.
- Analyzed whether additional trainable capacity improves semantic movie retrieval or overfits a small relevance dataset.
