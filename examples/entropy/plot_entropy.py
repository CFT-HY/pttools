"""
Entropy
=======

Plot the relative change in entropy density
"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.logging import setup_logging
from pttools.analysis.plot_entropy import gen_and_plot_entropy
from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel


def main():
    n_points = 20
    models = [
        # BagModel(a_s=1.1, a_b=1, V_s=1),
        # ConstCSModel(css2=1/3-0.01, csb2=1/3-0.02, a_s=1.5, a_b=1, V_s=1)
        BagModel(g_s=123, g_b=120, V_s=0.9),
        ConstCSModel(css2=1/3 - 0.01, csb2=1/3 - 0.011, g_s=123, g_b=120, V_s=0.9)
    ]
    alpha_n_min = np.max([model.alpha_n_min for model in models]) + 0.01

    gen_and_plot_entropy(
        models=models,
        v_walls=np.linspace(0.05, 0.95, n_points),
        alpha_ns=np.linspace(alpha_n_min, 0.95, n_points),
        # v_walls=np.linspace(0.3, 0.8, 9),
        # alpha_ns=np.linspace(0.12, 0.5, 9),
        min_level=-0.3,
        max_level=0.4,
        diff_level=0.05,
        # use_bag_solver=True
    )


if __name__ == "__main__":
    setup_logging()
    main()
    plt.show()
