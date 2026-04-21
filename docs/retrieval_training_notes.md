# Retrieval Training Notes

These notes summarize the design lessons from comparing the original CineSeek trained dual-tower checkpoint with the lighter adapter experiments in this repository.

## Main Lesson

The original CineSeek training idea was directionally correct: movie items benefit from combining title and overview signals before retrieval. The issue was not the use of concatenation itself. The issue was that the old model changed too much of the representation space at once.

The stronger formulation is:

```text
Keep the pretrained query embedding space stable.
Learn only a small item-side or post-embedding adapter.
Train against the same retrieval universe used at evaluation time.
```

## Original Dual-Tower Formulation

The old CineSeek checkpoint used a randomly initialized dual-tower projection model:

```text
query:
  sentence-transformer embedding, 384d
  -> MLP
  -> 128d normalized vector

movie:
  title embedding, 384d
  overview embedding, 384d
  -> concat, 768d
  -> MLP fusion
  -> MLP projection
  -> 128d normalized vector
```

This is more aggressive than a lightweight adapter. It does not simply learn `concat -> Linear(768 -> 384)`. It moves both queries and items into a new randomly initialized 128-dimensional space.

That design has two risks:

- It can destroy useful pretrained sentence-transformer geometry.
- It asks a small retrieval dataset to relearn a shared semantic space from scratch.

## Adapter Formulation

The adapter experiments keep the pretrained embedding space mostly intact.

### Raw Frozen Baseline

```text
query:
  sentence-transformer embedding, 384d
  -> normalize

movie:
  normalize(title embedding)
  normalize(overview embedding)
  -> average
  -> normalize
```

This is already strong because query and movie text are encoded by the same pretrained sentence-transformer.

### Linear Adapter

```text
embedding, 384d
-> small residual linear projection
-> 384d normalized vector
```

This learns a global calibration of the embedding space.

### Residual MLP Adapter

```text
embedding, 384d
-> LayerNorm
-> MLP
-> residual add
-> 384d normalized vector
```

This learns a nonlinear correction while preserving the original embedding.

### Concat Linear Adapter

```text
query:
  sentence-transformer embedding, 384d
  -> normalize

movie:
  title embedding, 384d
  overview embedding, 384d
  -> concat, 768d
  -> Linear(768 -> 384)
  -> normalize
```

The projection is initialized as a weighted average:

```text
0.5 * title + 0.5 * overview
```

This starts training near the raw frozen baseline instead of from a random retrieval space.

## In-Batch Contrastive vs Full-Catalog Contrastive

Both objectives are contrastive learning. The difference is the negative set.

### In-Batch Contrastive

The old training loop used:

```python
logits = query_repr @ item_repr.T
targets = torch.arange(logits.shape[0])
loss = cross_entropy(logits, targets)
```

If the batch size is 512, each query sees:

```text
1 positive movie + 511 in-batch negative movies
```

This is common in large-scale contrastive learning because the full catalog is often too large to score. It is not wrong, but it optimizes a sampled version of the final retrieval problem.

The final search task is different:

```text
rank the correct movie against the full movie catalog
```

For CineSeek, the catalog is only about 9.7K movies, so full-catalog training is affordable.

### Full-Catalog Contrastive

The adapter training loop scores every batch query against every movie:

```text
query_repr: [batch_size, 384]
item_repr:  [num_movies, 384]
logits:     [batch_size, num_movies]
```

Each query sees:

```text
1 positive movie + all other catalog movies as negatives
```

This better matches FAISS retrieval evaluation because both training and evaluation rank against the same movie universe.

## Temperature Matters

Normalized dot products have a narrow numeric range. Without temperature scaling, the softmax distribution can be too flat and the gradient signal can be weak.

The adapter experiments use:

```text
logits / 0.05
```

This sharpens the contrast between positives and negatives.

## Empirical Takeaway

The old trained checkpoint underperformed the frozen baseline because it combined several difficult choices:

- randomly initialized query and item towers
- dimensionality reduction from 384d to 128d
- in-batch negatives only
- no explicit temperature scaling
- a training objective less aligned with full-catalog retrieval

The concat linear adapter works better because it keeps the useful part of the old idea while removing the unstable parts:

- keep query embeddings frozen
- keep the final retrieval dimension at 384d
- initialize item fusion near the raw baseline
- train with full-catalog contrastive classification
- compare against raw, linear, and residual MLP baselines

The practical lesson is not that training is bad. The lesson is that small-data retrieval adaptation should start from a strong frozen representation and make controlled, minimal changes.
