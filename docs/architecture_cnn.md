# CNN Architecture — Baseline (`WaferCNN`)

Implementation: `models/cnn.py`. Data pipeline: `utils/dataset.py`.

## Overview

A small CNN for 9-class wafer map defect classification on WM-811K, designed to train on **CPU only**.

- **Feature extractor**: 3 conv blocks that learn local patterns (defective dies → clusters, rings, scratches).
- **Classifier head**: flatten → 2 fully connected layers → 9 logits.

## Input

| Property | Value |
| :---- | :---- |
| Shape | `(B, 3, 64, 64)` |
| Encoding | One-hot of wafer values: ch0 = outside wafer, ch1 = good die, ch2 = defective die |
| Dtype | `float32` (values 0 / 1) |

One-hot is used instead of scaling 0/1/2 to [0, 1] because the values are **categories, not intensities** — "defective" is not "twice" a good die.

## Building block

Each block is:

```
Conv2d (3x3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d (2x2)
```

| Layer | Purpose |
| :---- | :---- |
| Conv2d 3×3, padding 1 | Learnable local filters; keeps H×W unchanged |
| BatchNorm2d | Stabilises training, faster convergence (important on CPU) |
| ReLU | Non-linearity |
| MaxPool2d 2×2 | Halves H×W → less compute, larger receptive field for deeper layers |

## Layer-by-layer

| Layer | Output shape | Parameters |
| :---- | :---- | ----: |
| Input (one-hot) | 3 × 64 × 64 | – |
| Block 1: conv 3→16, BN, ReLU, pool | 16 × 32 × 32 | 480 |
| Block 2: conv 16→32, BN, ReLU, pool | 32 × 16 × 16 | 4,704 |
| Block 3: conv 32→64, BN, ReLU, pool | 64 × 8 × 8 | 18,624 |
| Flatten | 4096 | – |
| Dropout (0.5) → Linear 4096→128 → ReLU | 128 | 524,416 |
| Dropout (0.5) → Linear 128→9 | 9 (logits) | 1,161 |
| **Total** | | **549,385** |

Compute: ~12M multiply-adds per image.

## Design decisions

### 1. Flatten instead of global average pooling

Global average pooling (GAP) averages each feature map into one number, discarding **where** a feature occurred. For wafer maps, location is part of the label:

- `Center` vs `Edge-Loc` vs `Loc` — the same kind of defect cluster in different positions.
- `Edge-Ring` vs `Donut` — both rings, differing in radius.

Flattening keeps the 8×8 spatial layout so the classifier can learn position-dependent patterns. The cost is that most parameters sit in the first Linear layer, so dropout is applied there to limit overfitting.

### 2. No softmax in the model

The model outputs **raw logits**. `nn.CrossEntropyLoss` applies log-softmax internally; adding a softmax layer would apply it twice and hurt training. Use `softmax` / `argmax` only at inference time.

### 3. Small channel widths (16 / 32 / 64)

Chosen for CPU training. If the baseline underfits, widen to 32 / 64 / 128 or add a 4th block.

## Training setup

| Item | Choice |
| :---- | :---- |
| Loss | `CrossEntropyLoss` with inverse-frequency class weights (train split only) |
| Optimizer | Adam, lr = 1e-3 |
| Batch size | 128–256 |
| Epochs | ~15–20 |
| Model selection | Best **validation macro-F1** (not accuracy — `none` is ~85% of the data) |
| Augmentation (train only) | Random 90° rotations + horizontal flip |

### Class weights

Computed as `N / (num_classes * count_c)`:

| Class | Weight | Class | Weight |
| :---- | ----: | :---- | ----: |
| Center | 4.48 | Near-full | 129.34 |
| Donut | 34.58 | Random | 22.20 |
| Edge-Loc | 3.70 | Scratch | 16.11 |
| Edge-Ring | 1.99 | none | 0.13 |
| Loc | 5.35 | | |

The ~1000× range between `Near-full` and `none` may make training noisy. If so, try square-root weights.

## CPU considerations

- Measured: ~0.4 s per training step at batch 256 → **~3 min/epoch** (compute only, 6 threads).
- DataLoader workers (one-hot + augmentation per sample) compete with compute threads for the 12 cores — use 2–4 workers and check that loading isn't the bottleneck.
- Machine has ~7 GB RAM; the dataset uses ~700 MB in memory. Don't raise `num_workers` too high.
- Before a full run, overfit a ~500-sample subset to ~100% accuracy as a bug check.
