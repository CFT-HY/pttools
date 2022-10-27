"""Figure of the effective degrees of freedom in the Standard Model"""

import matplotlib.pyplot as plt

from pttools.models.sm import StandardModel
from pttools.analysis.g_cs2 import plot_g_cs2


def main():
    sm = StandardModel()
    plot_g_cs2(sm)


if __name__ == "__main__":
    main()
    plt.show()
