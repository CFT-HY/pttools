import typing as tp

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np


def cmap(
        min_level: float,
        max_level: float,
        diff_level: float,
        cmap_neg_name: str = "Blues",
        cmap_pos_name: str = "Reds") -> tp.Tuple[np.ndarray, tp.List[float]]:
    """Colormap for Matplotlib heatmap plots"""
    n_min = int(min_level / diff_level)
    n_max = int(max_level / diff_level)

    levels = np.linspace(n_min, n_max, n_max - n_min + 1, endpoint=True) * diff_level
    cmap_neg = plt.colormaps[cmap_neg_name]
    cmap_pos = plt.colormaps[cmap_pos_name]

    cols = \
        list(cmap_neg((levels[levels < 0] - diff_level) / (min_level - diff_level))) + \
        list(cmap_pos((levels[levels >= 0] + diff_level) / (max_level + diff_level)))

    return levels, cols


def color_region(
        ax: plt.Axes,
        x: np.ndarray, y: np.ndarray, region: np.ndarray,
        color: str = "red", alpha: float = 1):
    cmp = ListedColormap([color], color, 1)
    # The data type must be supporte by np.isinf()
    region2 = region.copy() if region.dtype is np.float64 else region.astype(np.float64)
    region2[region2 == 0] = np.nan
    # region2[np.isinf(region2)] = np.nan
    return ax.contourf(x, y, region2, cmap=cmp, alpha=alpha)
