"""
Parallel
========

Minimal example of parallel bubble solving
"""

import numpy as np

from pttools.models import BagModel
from pttools.analysis.parallel import create_bubbles


def main():
    model = BagModel(g_s=123, g_b=120, V_s=0.9)

    bubbles = create_bubbles(
        model=model,
        v_walls=np.linspace(0.05, 0.95, 20),
        alpha_ns=np.linspace(model.alpha_n_min+0.01, 0.5, 20)
    )

    print(bubbles[0, 0].sol_type)


if __name__ == "__main__":
    main()
