import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

PKL_PATH = "data/LSWMD.pkl"
IMG_SIZE = 64
SEED = 42

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "split ratios must sum to 1"

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    print(f"[WM-811K] Loaded {len(df):,} rows")
    return df

def extract_label(val) -> str:
    """failureType is stored as a nested array like [['Center']] or an empty array."""
    while isinstance(val, (np.ndarray, list)):
        if len(val) == 0:
            return "unlabeled"
        val = val[0]
    return str(val).strip() or "unlabeled"

def pad_to_square(m):
    m = np.asarray(m, dtype=np.uint8)
    h, w = m.shape
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return np.pad(m, ((top, s - h - top), (left, s - w - left)), constant_values=0)

def preprocess_map(m, size=IMG_SIZE):
    return cv2.resize(pad_to_square(m), (size, size), interpolation=cv2.INTER_NEAREST)

def split_dataset(df_clean: pd.DataFrame):
    classes = sorted(df_clean["failureType"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    X = np.stack(df_clean["waferMap_64"].values).astype(np.uint8)   # (N, 64, 64)
    y = df_clean["failureType"].map(class_to_idx).values

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_RATIO, stratify=y, random_state=SEED)
    # val is taken from what's left after the test split, so rescale: 0.1 / 0.8 = 0.125
    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train_idx, val_idx = train_test_split(train_idx, test_size=val_size, stratify=y[train_idx], random_state=SEED)

    counts = pd.DataFrame({
        "train": pd.Series(y[train_idx]).value_counts(),
        "val": pd.Series(y[val_idx]).value_counts(),
        "test": pd.Series(y[test_idx]).value_counts(),
    }).reindex(range(len(classes))).fillna(0).astype(int)
    counts.index = classes
    counts.loc["Total"] = counts.sum()

    print(f"TRAIN/VALIDATE/TEST ratio: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(counts.to_string())

    np.savez_compressed(
        "data/wm811k_clean.npz",
        X=X,                       # (N, 64, 64) uint8
        y=y,                       # (N,) int labels
        classes=np.array(classes),
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

def main():
    # df = load_data(PKL_PATH)
    # df["failureType"] = df["failureType"].apply(extract_label)

    # df_labeled = df[df["failureType"] != "unlabeled"].reset_index(drop=True)
    # df_labeled.to_pickle('data/wm811k-labelled.pkl')

    # df_labeled = load_data("data/wm811k-labelled.pkl")
    # df_clean = df_labeled.drop(df_labeled.nsmallest(6, 'dieSize').index)
    # df_clean["waferMap_64"] = df_clean["waferMap"].apply(preprocess_map)
    # df_clean.to_pickle('data/wm811k-clean.pkl')

    # df         - original wm-811k dataset
    # df_labeled - dataset with unlabelled data removed
    # df_clean   - dataset with labelled data scaled to 64x64 

    df_clean = load_data("data/wm811k-clean.pkl")
    # split_dataset(df_clean)



if __name__ == "__main__":
    main()