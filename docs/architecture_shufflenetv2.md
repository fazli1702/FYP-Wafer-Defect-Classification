# ShuffleNetV2 Architecture (x0.5 / x1.0 / x1.5 / x2.0)

Implementation: `models/shufflenet.py` (wraps `torchvision.models.shufflenet_v2_x0_5` … `shufflenet_v2_x2_0`). Compare with: `docs/architecture_cnn.md`, `docs/architecture_mobilenetv3.md`.

## Usage

```bash
python main.py --overfit --model shufflenet_x1_0 [--pretrained]   # sanity check
python main.py --model shufflenet_x0_5 [--pretrained]             # -> results/shufflenet_x0_5[_pretrained]/
python main.py --model shufflenet_x1_0 [--pretrained]
python main.py --model shufflenet_x1_5 [--pretrained]
python main.py --model shufflenet_x2_0 [--pretrained]
```

Training loop, loss, optimizer and metrics are identical to the other models (`docs/training_loop.md`).

## Why ShuffleNetV2

ShuffleNetV2 (Ma et al., 2018) is another efficient architecture for CPUs and mobile devices, like MobileNetV3. Its design comes from measuring **actual speed on hardware** rather than only counting multiply-adds (FLOPs), which makes it a natural fit for a CPU-only project. Comparing it with MobileNetV3 shows whether a different efficient design suits wafer maps better.

## Key building blocks

| Component | What it does |
| :---- | :---- |
| **Channel split** | Each basic unit splits its input channels into two halves: one half passes through unchanged (an identity shortcut), the other goes through the convolutions. |
| **Depthwise separable convolution** | 1×1 conv → 3×3 depthwise conv → 1×1 conv on the processed half. Cheap in parameters and compute. |
| **Concatenation (not addition)** | The two halves are concatenated back together, so the number of channels stays the same. |
| **Channel shuffle** | Mixes channels between the two halves so information flows across groups in the next unit. Without it, the untouched half would never be processed. |
| **Downsampling unit** | At the start of each stage, both branches process the full input with stride 2, halving H×W and doubling the channels. |

## Variants

The variants differ only in **width** (channels per stage). Depth is the same: stages of 4, 8 and 4 units.

Measured on this machine (CPU, 64×64 input, batch 256, compute only):

| Variant | Stage channels (2 / 3 / 4) | conv5 | Parameters | Time / epoch |
| :---- | :---- | ----: | ----: | ----: |
| `shufflenet_x0_5` | 48 / 96 / 192 | 1024 | 351,017 | ~1.6 min |
| `shufflenet_x1_0` | 116 / 232 / 464 | 1024 | 1,262,829 | ~3.1 min |
| `shufflenet_x1_5` | 176 / 352 / 704 | 1024 | 2,487,849 | ~5.0 min |
| `shufflenet_x2_0` | 244 / 488 / 976 | 2048 | 5,363,437 | ~8.4 min |

For reference: baseline CNN 549k params (~3 min/epoch), MobileNetV3-Small 1.53M (~3.2 min), MobileNetV3-Large 4.21M (~8.5 min).

- `x0_5` is the **smallest and fastest model in the project**, smaller than the baseline CNN.
- `x1_0` is a close match to MobileNetV3-Small in both size and speed, which makes it a fair head-to-head comparison.
- `x2_0` is a close match to MobileNetV3-Large.

## Layer-by-layer (x1.0, 64×64 input)

| Stage | Output shape (C × H × W) | Parameters |
| :---- | :---- | ----: |
| Input (one-hot) | 3 × 64 × 64 | – |
| conv1: 3×3, stride 2 + BN + ReLU | 24 × 32 × 32 | 696 |
| maxpool 3×3, stride 2 | 24 × 16 × 16 | – |
| stage2 (4 units) | 116 × 8 × 8 | 30,192 |
| stage3 (8 units) | 232 × 4 × 4 | 244,180 |
| stage4 (4 units) | 464 × 2 × 2 | 501,352 |
| conv5: 1×1 + BN + ReLU | 1024 × 2 × 2 | 477,184 |
| Global average pool | 1024 | – |
| Linear 1024→9 (`fc`) | 9 (logits) | 9,225 |
| **Total** | | **1,262,829** |

Other variants have the same spatial sizes, with the channel counts from the variants table. Output is raw logits (no softmax).

## Design decisions

### 1. Input kept at 64×64

This is the same reasoning as MobileNetV3. ShuffleNetV2 also downsamples 32×, so the final map is **2×2**. Resizing to 224×224 would cost ~12× the compute and add no information.

### 2. Global average pooling

Like MobileNetV3, ShuffleNetV2 ends in global average pooling, unlike the baseline CNN, which flattens to keep defect position. Compare per-class F1 on `Center`, `Edge-Loc` and `Loc` across models.

### 3. Small classifier head

The head is a single Linear layer (no hidden layer, no dropout). MobileNetV3's head has a hidden layer of 1024–1280 units and dropout 0.2, and the baseline CNN's head has a hidden layer of 128 units and dropout 0.5. Almost all of ShuffleNetV2's parameters are in the convolutional feature extractor, so overfitting is controlled mainly by data augmentation and early stopping.

### 4. ImageNet pretraining (`--pretrained`)

- Pretraining is handled the same way as for MobileNetV3. The whole network is loaded with ImageNet weights, and only the final `fc` is replaced (→ 9 classes).
- No ImageNet normalisation is applied, because the inputs are binary masks.
- All four variants have published ImageNet weights.

### 5. No BatchNorm change needed

torchvision's ShuffleNetV2 already uses BN momentum 0.1 (the PyTorch default). MobileNetV3 needed a fix here (`docs/architecture_mobilenetv3.md`, decision 4).

## Overfit check (500 samples, 30 epochs, no augmentation)

| Variant | Scratch: acc @ epoch 10 / 30 | Pretrained: acc @ epoch 10 / 30 |
| :---- | :---- | :---- |
| x0_5 | 0.984 / 0.998 | 0.922 / 0.998 |
| x1_0 | 1.000 / 1.000 | 0.944 / 0.998 |
| x1_5 | 1.000 / 1.000 | 0.998 / 1.000 |
| x2_0 | 1.000 / 1.000 | 0.998 / 0.998 |

All variants memorise the subset (≥ 99.8%), so the pipeline is correct for every one. When accuracy is 0.998, exactly 1 of the 500 samples is wrong. If that sample belongs to a rare class, macro-F1 drops noticeably even though accuracy is near perfect.

Unlike MobileNetV3, pretraining **slowed** memorisation for the two narrowest variants (x0_5, x1_0). The ImageNet features in these small networks may be harder to repurpose for binary wafer masks than random initialisation. Whether this carries over to full training is worth checking: compare `shufflenet_x1_0` against `shufflenet_x1_0_pretrained`.

## Measured results

Full runs pending.
