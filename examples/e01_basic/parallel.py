"""
Parallel
========

Minimal example of parallel bubble solving
"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis import BubbleGridVWAlpha, VwAlphaPlot
from pttools.bubble import get_kappa_omega
from pttools.models import BagModel


def main():
    v_walls = np.linspace(0.05, 0.95, 20)
    alpha_ns = np.linspace(0.05, 0.3, 20)
    model = BagModel(a_s=1.1, a_b=1, V_s=1)

    # Parallel computation
    grid = BubbleGridVWAlpha(model, v_walls, alpha_ns, get_kappa_omega)
    # bubbles = grid.bubbles
    kappas = grid.data[0]
    # omegas = grid.data[1]

    # Plotting
    plot = VwAlphaPlot(grid)
    plot.contourf(kappas, label=r"$\kappa$")
    plot.chapman_jouguet()


if __name__ == "__main__":
    main()
    plt.show()
