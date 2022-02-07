"""Figure of the effective degrees of freedom in the Standard Model"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble import geff


def main():
    fig: plt.Figure
    axs: np.ndarray
    fig, axs = plt.subplots(2, 1, sharex=True)
    fig.subplots_adjust(hspace=0)

    temp = np.logspace(geff.DATA[0, 0], geff.DATA[0, -1], 100)
    axs[0].plot(temp, geff.grho_gs_ratio(temp), label=r"$g_\rho/g_s(T)$, spline")
    axs[0].scatter(geff.DATA_TEMP, geff.DATA_GRHO_GS_RATIO, label=r"$g_\rho/g_s(T)$, data")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(temp, geff.g_rho(temp), label=r"$g_\rho(T)$, spline", color="blue")
    axs[1].plot(temp, geff.g_s(temp), label=r"$g_s(T)$, spline", color="red")
    axs[1].scatter(geff.DATA_TEMP, geff.DATA_G_RHO, label=r"$g_\rho(T)$, data")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("T (MeV)")
    axs[1].grid(True)
    axs[1].legend()


if __name__ == "__main__":
    main()
    plt.show()
