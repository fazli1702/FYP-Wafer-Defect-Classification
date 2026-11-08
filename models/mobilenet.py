"""MobileNetV3-Small / -Large (torchvision) for WM-811K wafer map defect classification.

See docs/architecture_mobilenetv3.md for the design rationale.

Input:  (B, 3, 64, 64) one-hot wafer maps from utils/dataset.py
Output: (B, num_classes) raw logits -- no softmax, CrossEntropyLoss applies it.
"""

from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)

VARIANTS = {
    "small": (mobilenet_v3_small, MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "large": (mobilenet_v3_large, MobileNet_V3_Large_Weights.IMAGENET1K_V1),
}


def build_mobilenet_v3(num_classes=9, size="small", pretrained=False, bn_momentum=0.1):
    """MobileNetV3 ("small" or "large") with the last classifier layer sized for `num_classes`.

    pretrained=True loads ImageNet weights for the whole network, then replaces
    only the final Linear (-> 1000) with a new one (-> num_classes).

    bn_momentum: torchvision uses 0.01 (tuned for long ImageNet training), which makes
    BatchNorm running stats lag far behind the weights on short runs -> eval-mode
    predictions collapse to the majority class. 0.1 is the PyTorch default.
    """
    builder, weights = VARIANTS[size]
    if not pretrained:
        model = builder(weights=None, num_classes=num_classes)
    else:
        model = builder(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = bn_momentum
    return model


def main():
    """Sanity check: push a batch through the model and print shapes."""
    from utils.dataset import get_dataloaders

    train_loader, _, _, classes, _ = get_dataloaders(num_workers=0)
    X, _ = next(iter(train_loader))

    for size in VARIANTS:
        model = build_mobilenet_v3(num_classes=len(classes), size=size, pretrained=True)
        logits = model(X)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"size={size}")
        print(f"  input:    {tuple(X.shape)}")
        print(f"  features: {tuple(model.features(X).shape)}")
        print(f"  output:   {tuple(logits.shape)}")
        print(f"  params:   {n_params:,}")


if __name__ == "__main__":
    main()
