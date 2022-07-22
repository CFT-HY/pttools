"""Figure of the effective degrees of freedom in the Standard Model"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble.boundary import Phase
from pttools.models.sm import StandardModel


def main():
    fig: plt.Figure
    axs: np.ndarray
    fig, axs = plt.subplots(3, 1, sharex=True)
    fig.subplots_adjust(hspace=0)

    sm = StandardModel()
    temp = np.logspace(sm.GEFF_DATA_LOG_TEMP[0], sm.GEFF_DATA_LOG_TEMP[-1], 100)
    axs[0].plot(temp, sm.ge_gs_ratio(temp), label=r"$g_e/g_s(T)$, spline")
    axs[0].scatter(sm.GEFF_DATA_TEMP, sm.GEFF_DATA_GE_GS_RATIO, label=r"$g_e/g_s(T)$, data")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(temp, sm.ge(temp, Phase.SYMMETRIC), label=r"$g_e(T)$, spline", color="blue")
    axs[1].plot(temp, sm.gs(temp, Phase.SYMMETRIC), label=r"$g_s(T)$, spline", color="red")
    axs[1].scatter(sm.GEFF_DATA_TEMP, sm.GEFF_DATA_GE, label=r"$g_e(T)$, data")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(temp, sm.cs2(temp, Phase.SYMMETRIC), label="$c_s^2$, spline")
    axs[2].scatter(sm.GEFF_DATA_TEMP, sm.cs2_full(sm.GEFF_DATA_TEMP, Phase.SYMMETRIC), label="$c_s^2$ from g-splines")
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_xlabel("T (MeV)")
    axs[2].set_xscale("log")


if __name__ == "__main__":
    main()
    plt.show()
