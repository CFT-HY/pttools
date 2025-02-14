"""
Parallel
========

Minimal example of parallel bubble solving
"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis import BubbleGridVWAlpha, VwAlphaPlot
from pttools.bubble import Bubble
from pttools.models import BagModel


def compute(bubble: Bubble):
    if bubble.no_solution_found or bubble.solver_failed:
        return np.nan, np.nan
    return bubble.kappa, bubble.omega

compute.fail_value = (np.nan, np.nan)
compute.return_type = (float, float)


def main():
    v_walls = np.linspace(0.05, 0.95, 20)
    alpha_ns = np.linspace(0.05, 0.3, 20)
    model = BagModel(a_s=1.1, a_b=1, V_s=1)

    # Parallel computation
    grid = BubbleGridVWAlpha(model, v_walls, alpha_ns, compute)
    # bubbles = grid.bubbles
    kappas = grid.data[0]
    # omegas = grid.data[1]

    # Plotting
    plot = VwAlphaPlot(grid)
    plot.colorbar()


if __name__ == "__main__":
    main()
    plt.show()
