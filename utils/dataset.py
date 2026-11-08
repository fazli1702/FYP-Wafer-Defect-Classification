"""PyTorch Dataset / DataLoaders for the preprocessed WM-811K data.

Reads `data/wm811k_clean.npz` produced by `preprocess.py`.

Wafer map values:
    0 = outside the wafer
    1 = good die
    2 = defective die

Each map is one-hot encoded into 3 channels (one binary mask per value),
so a sample has shape (3, 64, 64).
"""

"""
HOW TO USE:

from utils.dataset import get_dataloaders

train_loader, val_loader, test_loader, classes, class_weights = get_dataloaders()
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))

for X, y in train_loader:   # X: (B, 3, 64, 64) float32, y: (B,) int64
    ...
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

NPZ_PATH = "data/wm811k_clean.npz"
NUM_VALUES = 3  # 0 / 1 / 2 -> number of one-hot channels


class WaferDataset(Dataset):
    """One split (train / val / test) of the wafer maps.

    Parameters
    ----------
    X : np.ndarray, (N, 64, 64) uint8
        All wafer maps. Kept as uint8 to save memory; converted per sample.
    y : np.ndarray, (N,) int
        All labels.
    indices : np.ndarray
        Rows of X / y that belong to this split.
    augment : bool, default False
        Apply random 90-degree rotations and flips. Use for the train split only.
    """

    def __init__(self, X, y, indices, augment=False):
        self.X = X
        self.y = y
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]

        # (64, 64) -> (64, 64, 3) -> (3, 64, 64)
        wafer = torch.from_numpy(self.X[idx]).long()
        wafer = F.one_hot(wafer, num_classes=NUM_VALUES).permute(2, 0, 1).float()

        # defect patterns don't depend on wafer orientation
        if self.augment:
            k = int(torch.randint(0, 4, (1,)))
            wafer = torch.rot90(wafer, k, dims=(1, 2))
            if torch.rand(1) < 0.5:
                wafer = torch.flip(wafer, dims=(2,))

        label = torch.tensor(self.y[idx], dtype=torch.long)
        return wafer, label


def compute_class_weights(y_train, num_classes):
    """Inverse-frequency weights for CrossEntropyLoss: N / (num_classes * count_c).

    A perfectly balanced dataset gives every class weight 1.0.
    """
    counts = np.bincount(y_train, minlength=num_classes)
    weights = len(y_train) / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(path=NPZ_PATH, batch_size=128, num_workers=2, pin_memory=False):
    """Build the train / val / test DataLoaders from the saved .npz.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    classes : np.ndarray of str
        Class names, indexed by label.
    class_weights : torch.Tensor, (num_classes,)
        Computed from the train split only.
    """
    data = np.load(path)
    X, y, classes = data["X"], data["y"], data["classes"]

    train_ds = WaferDataset(X, y, data["train_idx"], augment=True)
    val_ds = WaferDataset(X, y, data["val_idx"])
    test_ds = WaferDataset(X, y, data["test_idx"])

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    class_weights = compute_class_weights(y[data["train_idx"]], len(classes))

    return train_loader, val_loader, test_loader, classes, class_weights
