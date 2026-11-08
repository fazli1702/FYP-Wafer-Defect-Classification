"""ShuffleNetV2 (torchvision) for WM-811K wafer map defect classification.

See docs/architecture_shufflenetv2.md for the design rationale.

Variants (width multipliers): x0_5, x1_0, x1_5, x2_0

Input:  (B, 3, 64, 64) one-hot wafer maps from utils/dataset.py
Output: (B, num_classes) raw logits -- no softmax, CrossEntropyLoss applies it.
"""

from torch import nn
from torchvision.models import (
    ShuffleNet_V2_X0_5_Weights,
    ShuffleNet_V2_X1_0_Weights,
    ShuffleNet_V2_X1_5_Weights,
    ShuffleNet_V2_X2_0_Weights,
    shufflenet_v2_x0_5,
    shufflenet_v2_x1_0,
    shufflenet_v2_x1_5,
    shufflenet_v2_x2_0,
)

VARIANTS = {
    "x0_5": (shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights.IMAGENET1K_V1),
    "x1_0": (shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
    "x1_5": (shufflenet_v2_x1_5, ShuffleNet_V2_X1_5_Weights.IMAGENET1K_V1),
    "x2_0": (shufflenet_v2_x2_0, ShuffleNet_V2_X2_0_Weights.IMAGENET1K_V1),
}


def build_shufflenet_v2(num_classes=9, size="x1_0", pretrained=False):
    """ShuffleNetV2 (`size` in VARIANTS) with the final Linear sized for `num_classes`.

    pretrained=True loads ImageNet weights for the whole network, then replaces
    only the final Linear `fc` (-> 1000) with a new one (-> num_classes).
    BatchNorm already uses PyTorch's default momentum (0.1), unlike MobileNetV3.
    """
    builder, weights = VARIANTS[size]
    if not pretrained:
        return builder(weights=None, num_classes=num_classes)

    model = builder(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    """Sanity check: push a batch through each variant and print shapes."""
    from utils.dataset import get_dataloaders

    train_loader, _, _, classes, _ = get_dataloaders(num_workers=0)
    X, _ = next(iter(train_loader))

    for size in VARIANTS:
        model = build_shufflenet_v2(num_classes=len(classes), size=size, pretrained=True)
        logits = model(X)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"size={size}")
        print(f"  input:    {tuple(X.shape)}")
        print(f"  output:   {tuple(logits.shape)}")
        print(f"  params:   {n_params:,}")


if __name__ == "__main__":
    main()
