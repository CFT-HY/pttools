"""
Parallel
========

Minimal example of parallel bubble solving
"""

import numpy as np

from examples.utils import save_and_show_fig
from pttools.analysis import BubbleGridVWAlpha, VwAlphaPlot
from pttools.bubble import get_kappa_omega
from pttools.models import BagModel


def main() -> VwAlphaPlot:
    """Minimal example of parallel bubble solving"""
    # Create the arrays of v_wall and alpha_n points that will be used for the grid
    v_walls = np.linspace(0.05, 0.95, 20)
    alpha_ns = np.linspace(0.05, 0.3, 20)
    # Create the equation of state
    model = BagModel(a_s=1.1, a_b=1, V_s=1)

    # Parallel computation
    grid = BubbleGridVWAlpha(model, v_walls, alpha_ns, get_kappa_omega)
    # bubbles = grid.bubbles
    kappas = grid.data[0]
    # omegas = grid.data[1]

    # Plotting
    plot = VwAlphaPlot(grid)
    plot.contourf_plusminus(kappas, label=r"$\kappa$")
    plot.chapman_jouguet()
    return plot


if __name__ == "__main__":
    _plot = main()
    save_and_show_fig(_plot.fig, "parallel")
