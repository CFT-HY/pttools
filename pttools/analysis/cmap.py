import typing as tp

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
    cmap_neg = plt.cm.get_cmap(cmap_neg_name)
    cmap_pos = plt.cm.get_cmap(cmap_pos_name)

    cols = \
        list(cmap_neg((levels[levels < 0] - diff_level) / (min_level - diff_level))) + \
        list(cmap_pos((levels[levels >= 0] + diff_level) / (max_level + diff_level)))

    return levels, cols
