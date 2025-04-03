import typing as tp

from matplotlib.contour import QuadContourSet
from matplotlib.colors import Colormap, ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

CMAP_DEFAULT: Colormap = cm.viridis
CMAP_NEG_DEFAULT: Colormap = cm.Blues
CMAP_POS_DEFAULT: Colormap = cm.Reds
REGION_COLOR_DEFAULT = "red"


def get_cmap(cmap: tp.Union[callable, str]) -> Colormap:
    if isinstance(cmap, str):
        return plt.colormaps[cmap]
    return cmap


def cmap_lines(n, cmap: tp.Union[callable, str] = CMAP_DEFAULT) -> np.ndarray:
    arr = np.linspace(0, 1, n)
    if isinstance(cmap, str):
        return plt.colormaps[cmap](arr)
    return cmap(arr)


def cmap_plusminus(
        min_level: float,
        max_level: float,
        diff_level: float,
        cmap_neg: tp.Union[Colormap, str] = CMAP_NEG_DEFAULT,
        cmap_pos: tp.Union[Colormap, str] = CMAP_POS_DEFAULT) -> tp.Tuple[np.ndarray, tp.List[float]]:
    """Colormap for Matplotlib heatmap plots"""
    n_min = int(min_level / diff_level)
    n_max = int(max_level / diff_level)

    levels = np.linspace(n_min, n_max, n_max - n_min + 1, endpoint=True) * diff_level
    cmap_neg = get_cmap(cmap_neg)
    cmap_pos = get_cmap(cmap_pos)

    colors = \
        list(cmap_neg((levels[levels < 0] - diff_level) / (min_level - diff_level))) + \
        list(cmap_pos((levels[levels >= 0] + diff_level) / (max_level + diff_level)))

    return levels, colors


def color_region(
        ax: plt.Axes,
        x: np.ndarray, y: np.ndarray, region: np.ndarray,
        color: str = REGION_COLOR_DEFAULT, alpha: float = 1) -> QuadContourSet:
    cmp = ListedColormap([color], color, 1)
    # The data type must be supported by np.isinf()
    region2 = region.copy() if region.dtype is np.float64 else region.astype(np.float64)
    region2[region2 == 0] = np.nan
    # region2[np.isinf(region2)] = np.nan
    return ax.contourf(x, y, region2, cmap=cmp, alpha=alpha)
