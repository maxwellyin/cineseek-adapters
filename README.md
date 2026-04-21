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
- **Concat linear adapter**: concatenate title and overview embeddings on the movie side, then learn a linear projection back to the query embedding dimension.
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

## Experiment Results

All models use the same CineSeek text-query-to-movie evaluation split and the same frozen sentence-transformer embedding inputs. Adapter models only train a lightweight transformation head on top of the frozen embeddings.

| Model | Split | R@10 | R@50 | R@100 | MRR | NDCG | Params | Encode ms | Search ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw frozen embeddings | val | 0.944 | 0.970 | 0.976 | 0.828 | 0.862 | 0 | 0.124 | 0.030 |
| Raw frozen embeddings | test | 0.931 | 0.963 | 0.973 | 0.829 | 0.862 | 0 | 0.058 | 0.025 |
| Linear adapter | test | 0.946 | 0.969 | 0.977 | 0.847 | 0.877 | 147,840 | 0.062 | 0.023 |
| Residual MLP adapter | test | 0.951 | 0.969 | 0.978 | 0.870 | 0.895 | 591,744 | 0.084 | 0.022 |
| Concat linear adapter | test | 0.956 | 0.975 | 0.979 | 0.872 | 0.897 | 295,296 | 0.071 | 0.023 |

Validation results for the residual MLP improved consistently over five epochs, with best validation MRR at epoch 5:

| Epoch | Val R@10 | Val MRR | Val NDCG |
| --- | ---: | ---: | ---: |
| 1 | 0.955 | 0.855 | 0.884 |
| 2 | 0.958 | 0.861 | 0.889 |
| 3 | 0.959 | 0.867 | 0.894 |
| 4 | 0.961 | 0.870 | 0.896 |
| 5 | 0.963 | 0.872 | 0.898 |

The concat linear ablation also improved consistently over five epochs, suggesting that preserving title and overview as separate item-side signals before projection is useful:

| Epoch | Val R@10 | Val MRR | Val NDCG |
| --- | ---: | ---: | ---: |
| 1 | 0.958 | 0.863 | 0.891 |
| 2 | 0.960 | 0.873 | 0.898 |
| 3 | 0.963 | 0.877 | 0.902 |
| 4 | 0.963 | 0.880 | 0.904 |
| 5 | 0.965 | 0.882 | 0.906 |

## Key Findings

- The raw sentence-transformer baseline is already strong, but lightweight adaptation still improves ranking quality.
- The linear adapter improves test MRR from 0.829 to 0.847 with only 147K trainable parameters.
- The residual MLP adapter improves test MRR from 0.829 to 0.870 and NDCG from 0.862 to 0.895.
- The concat linear adapter performs slightly better than the residual MLP on this split, reaching test MRR 0.872 and NDCG 0.897 with fewer parameters than the residual MLP.
- This suggests that the strongest gain may come from item-side title/overview fusion structure, not just from adding nonlinear capacity.
- The residual MLP adds less than 0.03 ms per query over the raw baseline in this offline measurement, so the quality gain is not coming from a large inference-cost increase.

## Training Notes

See [Retrieval Training Notes](docs/retrieval_training_notes.md) for the main lessons from comparing the original CineSeek trained dual-tower checkpoint with the adapter ablations, including why `concat -> projection` worked better after changing the objective and preserving the pretrained embedding space.

## Interpretation

The raw sentence-transformer baseline is already strong, but the adapter results show that controlled task adaptation still helps when it preserves the pretrained embedding geometry.

The most important result is the concat linear ablation. It outperforms both the raw baseline and the residual MLP while using fewer parameters than the residual MLP. This suggests that CineSeek benefits more from structured item-side title/overview fusion than from simply adding nonlinear capacity.

The main engineering lesson is that small-data retrieval adaptation should avoid relearning the full semantic space. A stable setup keeps the query encoder frozen, initializes item fusion near the raw baseline, and trains against the full movie catalog so the objective matches retrieval-time ranking.
