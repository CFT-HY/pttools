"""
Entropy comparison
==================

Comparison of the entropies of the old and new solvers
"""

import os.path

from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

from examples.utils import FIG_DIR
# from plot_entropy_old import load
# from pttools.analysis.cmap import cmap
from pttools.analysis.bubble_grid import BubbleGridVWAlpha
from pttools.analysis.plot_entropy import compute
from pttools.logging import setup_logging
from pttools.models.bag import BagModel
from pttools import speedup


def main(relative: bool = True, path: str = None):
    n_points = 10 if speedup.GITHUB_ACTIONS else 20
    v_walls = np.linspace(0.05, 0.95, n_points, endpoint=True)
    alpha_ns = v_walls
    # entropy_ref, v_walls, alpha_ns = load()

    # These should be the same as for the old data
    model = BagModel(g_s=123, g_b=120, V_s=0.9)

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()

    grid_old = BubbleGridVWAlpha(model, v_walls, alpha_ns, compute, use_bag_solver=True)
    entropy_old = grid_old.data[0]

    grid = BubbleGridVWAlpha(model, v_walls, alpha_ns, compute)
    entropy = grid.data[0]
    sn = grid.data[4]
    diff = (entropy_old - entropy)

    if relative:
        diff /= sn

    cs = ax.contourf(v_walls, alpha_ns, diff, locator=ticker.LinearLocator(numticks=20))
    # cs = ax.contourf(v_walls, alpha_ns, np.abs(diff), locator=ticker.LogLocator())

    # levels, cols = cmap(-0.3, 0.4, 0.05)
    # cs = ax.contourf(v_walls, alpha_ns, grid.data.T, levels=levels, colors=cols)
    # cbar = fig.colorbar(cs)
    # cbar.ax.set_ylabel(r'$\Delta s / s_n$')

    cbar = fig.colorbar(cs)
    if relative:
        cbar.ax.set_ylabel(r"$\frac{\Delta s_{new} - \Delta s_{old}}{s_n}$")
    else:
        cbar.ax.set_ylabel(r"$\Delta s_{new} - \Delta s_{old}$")

    ax.grid()
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$\alpha_n$")
    # ax.set_title(r"$\Delta s_{old} - \Delta s_{new}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.legend()

    if path is not None:
        fig.savefig(path)

    return fig


if __name__ == "__main__":
    setup_logging()
    main(path=os.path.join(FIG_DIR, "entropy_comparison.png"))
    plt.show()
