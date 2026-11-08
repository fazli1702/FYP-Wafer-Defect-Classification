# Training Loop Design

Implementation: `main.py`. Model: `models/cnn.py` (see `docs/architecture_cnn.md`). Data: `utils/dataset.py`.

## Usage

```bash
python main.py --overfit      # sanity check: memorise 500 samples (~1 min)
python main.py --epochs 2     # short run to measure real epoch time
python main.py                # full training (up to 20 epochs) + final test
python main.py --model mobilenet_small|mobilenet_large [--pretrained]   # MobileNetV3 (docs/architecture_mobilenetv3.md)
python main.py --model shufflenet_x0_5|x1_0|x1_5|x2_0 [--pretrained]   # ShuffleNetV2 (docs/architecture_shufflenetv2.md)
```

Each run saves to `results/<run name>/` (e.g. `cnn`, `mobilenet_small_pretrained`, `mobilenet_large`), so runs don't overwrite each other.

Measured on the CPU-only machine: overfit check ~30 s; one full epoch (train + val) ~3 min, so a 20-epoch run is ~1 hour.

## Overview

Each **epoch**:

1. **Train** — one pass over the train set, updating weights.
2. **Validate** — one pass over the val set, no updates, only measuring.
3. **Record** — log metrics; save a checkpoint if val macro-F1 is the best so far.

After training: load the **best** checkpoint and evaluate on the **test set once**.

## Configuration

| Setting | Value | Notes |
| :---- | :---- | :---- |
| Seed | 42 | Same as `preprocess.py`; makes shuffling/augmentation repeatable |
| Device | `cuda` if available, else `cpu` | Current machine is CPU-only |
| Max epochs | 20 | Override with `--epochs` |
| Batch size | 128 | |
| Optimizer | Adam, lr = 1e-3 | |
| Loss | `CrossEntropyLoss` with class weights | Weights from the train split only |
| LR scheduler | `ReduceLROnPlateau(mode="max", factor=0.5, patience=2)` | Watches val macro-F1; halves LR when it stalls |
| Early stopping | patience = 5 | Stop if val macro-F1 doesn't improve for 5 epochs |
| DataLoader workers | 2 | Raise to 3–4 if data loading is the bottleneck |

## Training one epoch

`model.train()` — dropout on, BatchNorm uses batch statistics. Then per batch:

1. **Forward** — `logits = model(X)`
2. **Loss** — `loss = loss_fn(logits, y)`
3. **Clear old gradients** — `optimizer.zero_grad()` (PyTorch accumulates gradients by default)
4. **Backward** — `loss.backward()`
5. **Update** — `optimizer.step()`

Loss is accumulated with `loss.item() * batch_size` — `.item()` avoids keeping the computation graph in memory.

## Evaluation (val and test)

- `model.eval()` — dropout off, BatchNorm uses running statistics learned during training.
- `torch.inference_mode()` — no gradient tracking; faster and less memory.
- Predictions: `logits.argmax(dim=1)` (softmax is unnecessary for argmax).

| Metric | Why |
| :---- | :---- |
| Loss | Tracks improvement / overfitting |
| Accuracy | Easy to read, but misleading: predicting `none` everywhere gives ~85% |
| **Macro-F1** | Mean F1 over the 9 classes, each weighted equally — rare classes matter as much as `none`. **Used for model selection.** |

The same `evaluate()` function is used for validation and test.

## Checkpointing and early stopping

- When val macro-F1 improves, save to `results/<run name>/best.pt`:
  `model_state`, `epoch`, `val_f1`, `classes`.
- Early stopping after 5 epochs without improvement.
- Safe to interrupt (Ctrl+C) — the best checkpoint is always on disk.

## Logging

- One line per epoch: train loss, val loss, val acc, val macro-F1, learning rate, epoch time.
- `results/<run name>/history.csv` is rewritten after every epoch (for training curves in the report).

## Final test (once)

1. Load `results/<run name>/best.pt`.
2. Evaluate on the test loader.
3. Save:
   - `results/<run name>/test_report.txt` — test loss / acc / macro-F1 + per-class precision, recall, F1 (`classification_report`).
   - `results/<run name>/confusion_matrix.png` — row-normalised (each row = recall per true class), readable despite class imbalance.

**Rule:** never tune anything based on test results. All decisions come from validation, otherwise the test score is no longer an unbiased estimate.

## Overfit sanity check (`--overfit`)

- 500 train samples, **no augmentation**, unweighted loss (some rare classes may be absent from 500 samples, which would make inverse-frequency weights infinite).
- 30 epochs; train accuracy on those 500 samples should approach ~100%.
- If it can't memorise a tiny set, something is broken (labels, shapes, missing `zero_grad`, ...) — found in a minute instead of after an hour of training.

## Common pitfalls (handled in the code)

| Pitfall | Effect |
| :---- | :---- |
| Forgetting `model.eval()` | Noisy, lower val/test scores (dropout still active) |
| Forgetting `optimizer.zero_grad()` | Gradients accumulate across batches → unstable training |
| Softmax inside the model | Applied twice with `CrossEntropyLoss` → worse training |
| Storing `loss` instead of `loss.item()` | Memory grows every batch |
| Selecting the model by accuracy | Favours predicting `none`; use macro-F1 |
| Tuning on the test set | Optimistic, invalid test score |
