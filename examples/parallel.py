"""
Parallel
========

Minimal example of parallel bubble solving
"""

import numpy as np

from pttools.models import BagModel
from pttools.analysis.parallel import create_bubbles


def main():
    model = BagModel(a_s=1.5, a_b=1, V_s=1)

    bubbles = create_bubbles(
        model=model,
        v_walls=np.linspace(0.1, 0.9, 3),
        alpha_ns=np.linspace(model.alpha_n_min+0.01, 0.3, 3)
    )

    print(bubbles[0, 0].sol_type)


if __name__ == "__main__":
    main()
