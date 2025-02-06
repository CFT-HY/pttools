"""
Parallel
========

Minimal example of parallel bubble solving
"""

import numpy as np

from pttools.analysis import BubbleGridVWAlpha
from pttools.bubble import Bubble
from pttools.models import BagModel


def compute(bubble: Bubble):
	if bubble.no_solution_found or bubble.solver_failed:
		return np.nan, np.nan
	return bubble.kappa, bubble.omega

compute.return_type = (float, float)


def main():
    v_walls = np.linspace(0.1, 0.9, 5)
    alpha_ns = np.linspace(0.1, 0.3, 5)
    model = BagModel(a_s=1.1, a_b=1, V_s=1)
    grid = BubbleGridVWAlpha(model, v_walls, alpha_ns, compute)
    bubbles = grid.bubbles
    kappas = grid.data[0]
    omegas = grid.data[1]
    print(bubbles.shape)
    print(kappas.shape)
    print(omegas.shape)


if __name__ == "__main__":
    main()
