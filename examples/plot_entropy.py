"""
Entropy
=======

Plot the relative change in entropy density
"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.plot_entropy import plot_entropy
from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel


def main():
    plot_entropy(
        # model=ConstCSModel(css2=1/3 - 0.01, csb2=1/3 - 0.02, a_s=1.5, a_b=1, V_s=1),
        # model=BagModel(a_s=1.1, a_b=1, V_s=1),
        model=BagModel(g_s=123, g_b=120, V_s=1),
        v_walls=np.linspace(0.12, 0.8, 9),
        alpha_ns=np.linspace(0.12, 0.8, 9),
        # v_walls=np.linspace(0.3, 0.8, 9),
        # alpha_ns=np.linspace(0.12, 0.5, 9)
    )


if __name__ == "__main__":
    main()
    plt.show()
