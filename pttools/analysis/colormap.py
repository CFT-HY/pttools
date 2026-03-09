"""Utilities for color maps"""

import math
import typing as tp

from matplotlib import cm
from matplotlib.contour import QuadContourSet
from matplotlib.colors import Colormap, ListedColormap
import matplotlib.pyplot as plt
import numpy as np

import pttools.type_hints as th

DEFAULT_CMAP: Colormap = cm.viridis
DEFAULT_CMAP_NEG: Colormap = cm.Blues
DEFAULT_CMAP_POS: Colormap = cm.Reds
DEFAULT_REGION_COLOR = "red"


def get_cmap(cmap: Colormap | str) -> Colormap:
    """Get a color map from its name

    If given a Colormap object, passes it through.
    """
    if isinstance(cmap, str):
        return plt.colormaps[cmap]
    return cmap


def cmap_lines(n, cmap: Colormap | str = DEFAULT_CMAP) -> th.FloatArr1D:
    arr = np.linspace(0, 1, n)
    if isinstance(cmap, str):
        return plt.colormaps[cmap](arr)
    return cmap(arr)


def cmap_plusminus(
        min_level: float,
        max_level: float,
        diff_level: float,
        cmap_neg: Colormap | str = DEFAULT_CMAP_NEG,
        cmap_pos: Colormap | str = DEFAULT_CMAP_POS) -> tuple[th.FloatArr1D, tp.List[float]]:
    """Colormap for Matplotlib heatmap plots with different color schemes for positive and negative values"""
    n_min = math.floor(min_level / diff_level)
    n_max = math.ceil(max_level / diff_level)

    levels = np.linspace(n_min, n_max, n_max - n_min + 1, endpoint=True) * diff_level
    cmap_neg = get_cmap(cmap_neg)
    cmap_pos = get_cmap(cmap_pos)

    colors = \
        list(cmap_neg((levels[levels < 0] - diff_level) / (min_level - diff_level))) + \
        list(cmap_pos((levels[levels >= 0] + diff_level) / (max_level + diff_level)))

    return levels, colors


def color_region(
        ax: plt.Axes,
        x: th.FloatArr1D,
        y: th.FloatArr1D,
        region: th.FloatArr2D,
        color: str = DEFAULT_REGION_COLOR, alpha: float = 1) -> QuadContourSet:
    """Set a region on a plot to a fixed color"""
    cmp = ListedColormap([color], color, 1)
    # The data type must be supported by np.isinf()
    region2 = region.copy() if region.dtype is np.float64 else region.astype(np.float64)
    region2[region2 == 0] = np.nan
    # region2[np.isinf(region2)] = np.nan
    return ax.contourf(x, y, region2, cmap=cmp, alpha=alpha)
