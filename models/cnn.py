"""Baseline CNN for WM-811K wafer map defect classification.

See docs/architecture_cnn.md for the design rationale.

Input:  (B, 3, 64, 64) one-hot wafer maps from utils/dataset.py
Output: (B, num_classes) raw logits -- no softmax, CrossEntropyLoss applies it.
"""

from torch import nn


def conv_block(in_channels, out_channels):
    """Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2x2. Halves height and width."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
    )


class WaferCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=9, img_size=64, dropout=0.5):
        super().__init__()

        # 3 x 64 x 64 -> 16 x 32 x 32 -> 32 x 16 x 16 -> 64 x 8 x 8
        self.features = nn.Sequential(
            conv_block(in_channels, 16),
            conv_block(16, 32),
            conv_block(32, 64),
        )

        # flatten (not global average pooling) so the head keeps *where* defects are
        feat_size = img_size // 2**3
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * feat_size * feat_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    """Sanity check: push a batch through the model and print shapes."""
    from utils.dataset import get_dataloaders

    train_loader, _, _, classes, _ = get_dataloaders(num_workers=0)
    X, _ = next(iter(train_loader))

    model = WaferCNN(num_classes=len(classes))
    logits = model(X)

    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"input:  {tuple(X.shape)}")
    print(f"output: {tuple(logits.shape)}")
    print(f"params: {n_params:,}")


if __name__ == "__main__":
    main()
