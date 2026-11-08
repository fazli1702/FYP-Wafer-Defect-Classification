# MobileNetV3 Architecture (Small / Large)

Implementation: `models/mobilenet.py` (wraps `torchvision.models.mobilenet_v3_small` / `mobilenet_v3_large`). Baseline for comparison: `docs/architecture_cnn.md`.

## Usage

```bash
python main.py --overfit --model mobilenet_small [--pretrained]   # sanity check
python main.py --model mobilenet_small                            # from scratch   -> results/mobilenet_small/
python main.py --model mobilenet_small --pretrained               # ImageNet init  -> results/mobilenet_small_pretrained/
python main.py --model mobilenet_large [--pretrained]             # Large variant  -> results/mobilenet_large[_pretrained]/
```

Training loop, loss (weighted cross-entropy), optimizer and metrics are identical to the baseline CNN (`docs/training_loop.md`), so the only difference between runs is the model.

## Why MobileNetV3

MobileNetV3 is designed for **efficient inference on CPUs and mobile devices**, which suits both this project's CPU-only setup and the realistic deployment scenario (fast defect classification on fab inspection hardware). It is a well-known, published architecture, giving a standard reference point against the custom baseline CNN.

## Variant: Small vs Large

Measured on this machine (CPU, 64×64 input, batch 256, compute only):

| Variant | Parameters | Final feature map | Time / epoch |
| :---- | ----: | :---- | ----: |
| **MobileNetV3-Small** | 1.53M | 576 × 2 × 2 | ~3.2 min |
| **MobileNetV3-Large** | 4.21M | 960 × 2 × 2 | ~8.5 min |

Small trains at about the same speed as the baseline CNN. Large has ~2.8× the parameters and is ~2.7× slower per epoch (a 20-epoch run is ~3 hours on CPU) — worth it only if Small underfits. Comparing the two shows whether extra capacity helps on this dataset.

## Key building blocks

| Component | What it does |
| :---- | :---- |
| **Depthwise separable convolution** | Splits a conv into a per-channel 3×3/5×5 (depthwise) + a 1×1 (pointwise) conv. Far fewer parameters and multiply-adds than a standard conv. |
| **Inverted residual block** | 1×1 expand → depthwise conv → 1×1 project, with a skip connection when shapes match. |
| **Squeeze-and-Excitation (SE)** | Learns a per-channel weight from a global summary of the feature map, letting the network emphasise informative channels. |
| **Hardswish activation** | Cheap approximation of Swish; used in the deeper layers. |

## Layer-by-layer — Small (64×64 input)

| Stage | Output shape (C × H × W) | Parameters |
| :---- | :---- | ----: |
| Input (one-hot) | 3 × 64 × 64 | – |
| 0: stem conv 3×3, stride 2 | 16 × 32 × 32 | 464 |
| 1: inverted residual (SE) | 16 × 16 × 16 | 744 |
| 2–3: inverted residual | 24 × 8 × 8 | 9,280 |
| 4–6: inverted residual (SE) | 40 × 4 × 4 | 128,264 |
| 7–8: inverted residual (SE) | 48 × 4 × 4 | 51,768 |
| 9–11: inverted residual (SE) | 96 × 2 × 2 | 680,040 |
| 12: 1×1 conv | 576 × 2 × 2 | 56,448 |
| Global average pool | 576 | – |
| Linear 576→1024 → Hardswish → Dropout (0.2) | 1024 | 590,848 |
| Linear 1024→9 | 9 (logits) | 9,225 |
| **Total** | | **1,527,081** |

## Layer-by-layer — Large (64×64 input)

| Stage | Output shape (C × H × W) | Parameters |
| :---- | :---- | ----: |
| Input (one-hot) | 3 × 64 × 64 | – |
| 0: stem conv 3×3, stride 2 | 16 × 32 × 32 | 464 |
| 1: inverted residual | 16 × 32 × 32 | 464 |
| 2–3: inverted residual | 24 × 16 × 16 | 7,880 |
| 4–6: inverted residual (SE) | 40 × 8 × 8 | 52,312 |
| 7–10: inverted residual | 80 × 4 × 4 | 130,824 |
| 11–12: inverted residual (SE) | 112 × 4 × 4 | 600,544 |
| 13–15: inverted residual (SE) | 160 × 2 × 2 | 2,023,944 |
| 16: 1×1 conv | 960 × 2 × 2 | 155,520 |
| Global average pool | 960 | – |
| Linear 960→1280 → Hardswish → Dropout (0.2) | 1280 | 1,230,080 |
| Linear 1280→9 | 9 (logits) | 11,529 |
| **Total** | | **4,213,561** |

Large is deeper (16 vs 12 stages) and wider (up to 160 vs 96 channels). Both output raw logits (no softmax), as with the baseline.

## Design decisions

### 1. Input kept at 64×64 (not resized to 224×224)

MobileNetV3 was designed for 224×224 ImageNet images and downsamples 32×. At 64×64 the final feature map is only **2×2**. Upsampling to 224×224 would add no information (the maps were already resized from their original sizes) and costs ~12× the compute — not feasible on CPU. Keeping 64×64 also makes the comparison with the baseline fair (same input).

### 2. Global average pooling (differs from the baseline)

MobileNetV3 ends with global average pooling, which the baseline CNN deliberately avoided to keep defect **position** (Center vs Edge-Loc vs Loc). Two things limit the impact here: the final map is only 2×2, and CNNs with zero-padding still encode some absolute position in deep features. Whether this hurts the position-dependent classes is something the comparison will show — check the per-class F1 for `Center`, `Edge-Loc`, `Loc`.

### 3. ImageNet pretraining (`--pretrained`)

- Loads ImageNet weights for the whole network, then replaces only the last Linear (1024→1000) with a new 1024→9 layer.
- The one-hot input conveniently has 3 channels, so the pretrained first conv layer fits without modification.
- ImageNet mean/std normalisation is **not** applied: the inputs are binary masks, not photos.
- Even though wafer maps look nothing like ImageNet photos, the pretrained low-level filters (edges, blobs) give a better starting point. Overfit check (epochs to ~100% on 500 samples): Small ~10 pretrained vs ~30 scratch; Large ~15 pretrained vs ~20 scratch.
- Training both `mobilenet_small` and `mobilenet_small_pretrained` gives a from-scratch vs transfer-learning comparison for the report.

### 4. BatchNorm momentum changed 0.01 → 0.1

torchvision's MobileNetV3 uses BN momentum 0.01, tuned for very long ImageNet training. Here, the running statistics lagged far behind the weights: in the overfit check, train loss reached ~0 but eval-mode predictions were all `none` (accuracy stuck at 83.6% = share of `none`). Setting momentum to 0.1 (the PyTorch default) fixed this — Small and Large, scratch and pretrained, all then reached 100% on the overfit subset.

This is a good example of why the overfit check runs before full training.

## Measured results so far

| Run | Epochs | Val macro-F1 | Test macro-F1 | Time / epoch |
| :---- | ----: | ----: | ----: | ----: |
| `mobilenet_small_pretrained` (pipeline check) | 1 | 0.639 | 0.636 | ~3.5 min |

Full runs pending.
