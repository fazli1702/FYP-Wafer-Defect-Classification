# Downloading Kaggle Dataset

Here is a quick guide on how to download the `WM-811K wafer map` dataset (`qingyi/wm811k-wafer-map`)
- [kaggle](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)

reference:
- [kaggle cli](https://github.com/Kaggle/kaggle-cli)

## Installation
Install `kaggle` using pip
```bash
pip install kaggle
```

Download the dataset
```bash
kaggle datasets download -d qingyi/wm811k-wafer-map
```

You can also download other datasets from kaggle
```bash
kaggle datasets download -d <username>/<dataset-name>
```