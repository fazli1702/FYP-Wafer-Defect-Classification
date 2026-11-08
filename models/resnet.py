"""ResNet50 (torchvision) for WM-811K wafer map defect classification.

See docs/architecture_resnet50.md for the design rationale.

Input:  (B, 3, 64, 64) one-hot wafer maps from utils/dataset.py
Output: (B, num_classes) raw logits -- no softmax, CrossEntropyLoss applies it.
"""

from torch import nn
from torchvision.models import ResNet50_Weights, resnet50


def build_resnet50(num_classes=9, pretrained=False):
    """ResNet50 with the final Linear sized for `num_classes`.

    pretrained=True loads ImageNet weights (IMAGENET1K_V2, the improved torchvision
    recipe) for the whole network, then replaces only the final Linear `fc`
    (2048 -> 1000) with a new one (2048 -> num_classes).
    BatchNorm already uses PyTorch's default momentum (0.1), unlike MobileNetV3.
    """
    if not pretrained:
        return resnet50(weights=None, num_classes=num_classes)

    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    """Sanity check: push a batch through the model and print shapes."""
    from utils.dataset import get_dataloaders

    train_loader, _, _, classes, _ = get_dataloaders(num_workers=0)
    X, _ = next(iter(train_loader))

    model = build_resnet50(num_classes=len(classes), pretrained=True)
    logits = model(X)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"input:  {tuple(X.shape)}")
    print(f"output: {tuple(logits.shape)}")
    print(f"params: {n_params:,}")


if __name__ == "__main__":
    main()
