"""
Entropy comparison
==================

Comparison of the entropies of the old and new solvers
"""

from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

from plot_entropy_old import load
# from pttools.analysis.cmap import cmap
from pttools.analysis.bubble_grid import BubbleGridVWAlpha
from pttools.analysis.plot_entropy import compute
from pttools.models.bag import BagModel


def main():
    entropy_old, v_walls, alpha_ns = load()

    # These should be the same as for the old data
    model = BagModel(g_s=123, g_b=120, V_s=0.9)

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()

    grid = BubbleGridVWAlpha(model, v_walls, alpha_ns, compute)
    data = grid.data.T
    diff = (entropy_old - data) / np.abs(data)

    cs = ax.contourf(v_walls, alpha_ns, diff)
    # cs = ax.contourf(v_walls, alpha_ns, diff, locator=ticker.LogLocator())

    # levels, cols = cmap(-0.3, 0.4, 0.05)
    # cs = ax.contourf(v_walls, alpha_ns, grid.data.T, levels=levels, colors=cols)
    # cbar = fig.colorbar(cs)
    # cbar.ax.set_ylabel(r'$\Delta s / s_n$')

    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r"$\frac{\Delta s_{new} - \Delta s_{old}}{| \Delta s_{new} |}$")

    ax.grid()
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$\alpha_n$")
    # ax.set_title(r"$\Delta s_{old} - \Delta s_{new}$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.legend()


if __name__ == "__main__":
    main()
    plt.show()
