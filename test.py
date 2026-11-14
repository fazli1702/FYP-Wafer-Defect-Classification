import matplotlib.pyplot as plt
from preprocess import load_data
from utils.draw_map import draw_map
import random


'''
help to print selected / random wafer maps
use util function draw_map from utils/draw_map.py
'''

def plot_single(idx, df, map_col="waferMap_64", title=None, save_path=None):
    """Plot one wafer map by its df index label, optionally saving it to a file."""
    if idx not in df.index:
        raise KeyError(f"Index {idx} not found in DataFrame")

    if title is None:
        title = f"{df.loc[idx, 'failureType']} (idx={idx})"

    fig, ax = plt.subplots(figsize=(5, 5))
    draw_map(idx, df=df, ax=ax, map_col=map_col, title=title)
    ax.title.set_fontsize(14)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    plt.show()


def main():
    df_clean = load_data("data/wm811k-clean.pkl")

    # dfs_by_type = {ftype: group.reset_index(drop=True)
    #                for ftype, group in df_clean.groupby("failureType")}

    # df_none      = dfs_by_type["none"]
    # df_center    = dfs_by_type["Center"]
    # df_donut     = dfs_by_type["Donut"]
    # df_edge_loc  = dfs_by_type["Edge-Loc"]
    # df_edge_ring = dfs_by_type["Edge-Ring"]
    # df_loc       = dfs_by_type["Loc"]
    # df_near_full = dfs_by_type["Near-full"]
    # df_random    = dfs_by_type["Random"]
    # df_scratch   = dfs_by_type["Scratch"]

    # for ftype, d in dfs_by_type.items():
    #     print(f"{ftype:10s} {len(d):,}")

    # one random sample from each failure type in a 3x3 grid
    # type_dfs = [df_none, df_center, df_donut, df_edge_loc, df_edge_ring,
    #             df_loc, df_near_full, df_random, df_scratch]

    # fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    # for ax, d in zip(axes.flat, type_dfs):
    #     draw_map(random.randint(0, len(d)-1), df=d, ax=ax, map_col="waferMap_64",
    #                 title=d.loc[0, "failureType"])
    # fig.tight_layout()
    # fig.savefig("imgs/one_per_failure_type.png", dpi=150)
    # plt.show()

    # clear examples of each failure type, used in imgs/failure_type_examples.png
    # indices are df_clean index labels (not the reset per-type df indices)
    #   none      142691
    #   Center    27601, 20158
    #   Donut     20310, 20297
    #   Edge-Loc  24689, 22078, 134128, 2705, 62317
    #   Edge-Ring 61391
    #   Loc       101718, 98767
    #   Near-full 96206
    #   Random    9667, 136556, 7171
    #   Scratch   144115, 143632
    # usage: draw_map(27601, df=df_clean, map_col="waferMap_64")

    # fixed sample from each failure type in a 2x5 grid (last slot left blank)
    fixed_idx = {
        "none":      142691,
        "Center":    20158,
        "Donut":     20297,
        "Edge-Loc":  22078,
        "Edge-Ring": 61391,
        "Loc":       101718,
        "Near-full": 96206,
        "Random":    9667,
        "Scratch":   144115,
    }

    # fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    # for ax, (ftype, idx) in zip(axes.flat, fixed_idx.items()):
    #     draw_map(idx, df=df_clean, ax=ax, map_col="waferMap_64", title=ftype)
    #     ax.title.set_fontsize(14)
    # axes.flat[-1].axis("off")
    # fig.tight_layout()
    # fig.savefig("imgs/failure_type_examples2.png", dpi=200)
    # plt.show()

    # single wafer map by df_clean index label
    # plot_single(9667, df_clean)
    plot_single(9667, df_clean, save_path="imgs/random_9667.png")
    plot_single(61391, df_clean, save_path="imgs/edgering_61391.png")
    plot_single(144115, df_clean, save_path="imgs/scratch_144115.png")


if __name__ == "__main__":
    main()