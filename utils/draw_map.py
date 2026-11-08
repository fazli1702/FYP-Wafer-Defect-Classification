"""Plot WM-811K wafer maps with matplotlib.

Wafer map values:
    0 = outside the wafer
    1 = good die
    2 = defective die
"""

"""
HOW TO USE:

# by index (needs the DataFrame)
draw_map(0, df=df_labeled)

# by 2D array
draw_map(df_labeled.loc[0, "waferMap"])


# draw multiple plots
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for ax, idx in zip(axes, small_maps.index[:5]):
    draw_map(idx, df=df_labeled, ax=ax)
plt.show()
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

WAFER_CMAP = ListedColormap(["lightgray", "green", "red"])  # 0, 1, 2


def draw_map(wafer, df=None, title=None, ax=None, show=True, map_col="waferMap"):
    """Draw a single wafer map from a 2D array or a DataFrame index.

    Parameters
    ----------
    wafer : int or 2D array-like
        Either a row index label into `df`, or the wafer map array itself.
    df : pandas DataFrame, optional
        Required when `wafer` is an index. The map is read from `df.loc[wafer, map_col]`.
    title : str, optional
        Title shown above the map. If None, a default title is built
        (failure type, shape and die count when available).
    ax : matplotlib Axes, optional
        Axes to draw on. If None, a new figure is created.
    show : bool, default True
        Call plt.show() when a new figure was created. Ignored when ax is given,
        so you can draw several maps into one grid before showing it.
    map_col : str, default "waferMap"
        Column in `df` that holds the wafer map arrays.

    Returns
    -------
    matplotlib Axes
    """
    # Index mode: look the row up in the DataFrame
    if isinstance(wafer, (int, np.integer)):
        if df is None:
            raise ValueError("Pass df=... when using an index instead of an array.")
        row = df.loc[wafer]
        wafer_map = np.asarray(row[map_col])
        if title is None:
            parts = [f"idx={wafer}"]
            if "failureType" in row:
                parts.append(str(row["failureType"]))
            parts.append(f"shape={wafer_map.shape}")
            if "dieSize" in row:
                parts.append(f"dies={int(row['dieSize'])}")
            title = " | ".join(parts)
    # Array mode
    else:
        wafer_map = np.asarray(wafer)
        if title is None:
            title = f"shape={wafer_map.shape}"

    if wafer_map.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {wafer_map.shape}")

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(4, 4))

    ax.imshow(wafer_map, cmap=WAFER_CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")

    if created_fig and show:
        plt.show()

    return ax