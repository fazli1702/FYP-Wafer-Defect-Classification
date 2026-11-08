"""Train a model (baseline WaferCNN, MobileNetV3, ShuffleNetV2 or ResNet50) on WM-811K.

See docs/training_loop.md for the design.

python main.py --overfit                          # sanity check: memorise 500 samples
python main.py --epochs 2                         # short run to time an epoch
python main.py                                    # full training + final test (WaferCNN)
python main.py --model mobilenet_small                # MobileNetV3-Small from scratch
python main.py --model mobilenet_small --pretrained   # MobileNetV3-Small, ImageNet weights
python main.py --model mobilenet_large --pretrained   # MobileNetV3-Large, ImageNet weights
python main.py --model shufflenet_x1_0 --pretrained   # ShuffleNetV2 1.0x (also x0_5, x1_5, x2_0)
python main.py --model resnet50 --pretrained          # ResNet50 (~40 min/epoch on CPU)

Results are saved to results/<run name>/, e.g. results/mobilenet_small_pretrained/.
"""

import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader

from models.cnn import WaferCNN
from models.mobilenet import build_mobilenet_v3
from models.resnet import build_resnet50
from models.shufflenet import build_shufflenet_v2
from utils.dataset import NPZ_PATH, WaferDataset, get_dataloaders

SEED = 42
EPOCHS = 50
BATCH_SIZE = 32
LR = 1e-3
LR_PATIENCE = 2           # epochs without val F1 improvement before halving LR
EARLY_STOP_PATIENCE = 5   # epochs without val F1 improvement before stopping
NUM_WORKERS = 6

OVERFIT_SAMPLES = 500
OVERFIT_EPOCHS = 30

MODELS = [
    "cnn",
    "mobilenet_small", "mobilenet_large",
    "shufflenet_x0_5", "shufflenet_x1_0", "shufflenet_x1_5", "shufflenet_x2_0",
    "resnet50",
]
MODEL = MODELS[3]

# each run writes these into results/<run name>/
RESULTS_DIR = "results"
CKPT_FILE = "best.pt"
HISTORY_FILE = "history.csv"
REPORT_FILE = "test_report.txt"
CM_FILE = "confusion_matrix.png"


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(name, num_classes, pretrained=False):
    if name == "cnn":
        if pretrained:
            raise ValueError("--pretrained is not available for cnn")
        return WaferCNN(num_classes=num_classes)
    if name.startswith("mobilenet_"):
        size = name.removeprefix("mobilenet_")
        return build_mobilenet_v3(num_classes=num_classes, size=size, pretrained=pretrained)
    if name.startswith("shufflenet_"):
        size = name.removeprefix("shufflenet_")
        return build_shufflenet_v2(num_classes=num_classes, size=size, pretrained=pretrained)
    if name == "resnet50":
        return build_resnet50(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown model: {name}")


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, loss_fn, device):
    """Returns (loss, accuracy, macro-F1, preds, labels)."""
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.inference_mode():
        for X, y in loader:
            X, y = X.to(device), y.to(device)

            logits = model(X)
            total_loss += loss_fn(logits, y).item() * len(y)

            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(y.cpu())

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    loss = total_loss / len(loader.dataset)
    acc = (preds == labels).mean()
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return loss, acc, macro_f1, preds, labels


def overfit_check(device, model_name, pretrained):
    """Train on a tiny subset with no augmentation; train accuracy should reach ~100%."""
    data = np.load(NPZ_PATH)
    subset = WaferDataset(data["X"], data["y"], data["train_idx"][:OVERFIT_SAMPLES], augment=False)
    loader = DataLoader(subset, batch_size=64, shuffle=True)

    model = build_model(model_name, len(data["classes"]), pretrained).to(device)
    # unweighted: some rare classes may be missing from 500 samples
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"[overfit] {model_name} | {OVERFIT_SAMPLES} samples, {OVERFIT_EPOCHS} epochs")
    for epoch in range(1, OVERFIT_EPOCHS + 1):
        train_loss = train_one_epoch(model, loader, loss_fn, optimizer, device)
        if epoch % 5 == 0 or epoch == 1:
            _, acc, f1, _, _ = evaluate(model, loader, loss_fn, device)
            print(f"epoch {epoch:2d} | train loss {train_loss:.4f} | acc {acc:.4f} | macro-F1 {f1:.4f}")


def save_history(history, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)


def test_best(model, test_loader, loss_fn, classes, device, run_dir):
    """Load the best checkpoint, evaluate on the test set once, save report + confusion matrix."""
    report_path = os.path.join(run_dir, REPORT_FILE)
    cm_path = os.path.join(run_dir, CM_FILE)

    ckpt = torch.load(os.path.join(run_dir, CKPT_FILE), map_location=device)
    model.load_state_dict(ckpt["model_state"])

    loss, acc, f1, preds, labels = evaluate(model, test_loader, loss_fn, device)
    report = classification_report(labels, preds, target_names=classes, digits=4, zero_division=0)

    summary = (
        f"Best checkpoint: epoch {ckpt['epoch']} (val macro-F1 {ckpt['val_f1']:.4f})\n"
        f"Test loss {loss:.4f} | acc {acc:.4f} | macro-F1 {f1:.4f}\n\n{report}"
    )
    print(summary)
    with open(report_path, "w") as f:
        f.write(summary)

    # row-normalised so rare classes are readable next to `none`
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay.from_predictions(
        labels, preds, display_labels=classes, normalize="true",
        values_format=".2f", cmap="Blues", ax=ax, colorbar=False,
    )
    ax.set_title(f"{os.path.basename(run_dir)} | test confusion matrix (row-normalised) | macro-F1 {f1:.4f}")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    print(f"Saved {report_path} and {cm_path}")


def train(device, epochs, model_name, pretrained, run_dir):
    train_loader, val_loader, test_loader, classes, class_weights = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    classes = [str(c) for c in classes]

    model = build_model(model_name, len(classes), pretrained).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=LR_PATIENCE
    )

    best_f1 = -1.0
    epochs_no_improve = 0
    history = []

    print(f"Training {os.path.basename(run_dir)} on {device} | train {len(train_loader.dataset):,} | val {len(val_loader.dataset):,}")
    for epoch in range(1, epochs + 1):
        start = time.time()

        train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, loss_fn, device)

        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_f1)
        elapsed = time.time() - start

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_acc": val_acc, "val_f1": val_f1, "lr": lr, "time_s": elapsed,
        })
        save_history(history, os.path.join(run_dir, HISTORY_FILE))

        improved = val_f1 > best_f1
        if improved:
            best_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_f1": val_f1,
                "classes": classes,
            }, os.path.join(run_dir, CKPT_FILE))
        else:
            epochs_no_improve += 1

        print(
            f"epoch {epoch:2d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
            f"val acc {val_acc:.4f} | val F1 {val_f1:.4f} | lr {lr:.0e} | "
            f"{elapsed // 60:.0f}m{elapsed % 60:02.0f}s" + (" *" if improved else "")
        )

        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping: no val F1 improvement for {EARLY_STOP_PATIENCE} epochs")
            break

    test_best(model, test_loader, loss_fn, classes, device, run_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit", action="store_true", help="run the tiny-subset sanity check")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--model", choices=MODELS, default=MODEL)
    parser.add_argument("--pretrained", action="store_true", help="ImageNet weights (not for cnn)")
    args = parser.parse_args()

    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.overfit:
        overfit_check(device, args.model, args.pretrained)
        return

    run_name = args.model + ("_pretrained" if args.pretrained else "")
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)
    train(device, args.epochs, args.model, args.pretrained, run_dir)


if __name__ == "__main__":
    main()
