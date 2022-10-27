import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble.boundary import Phase
from pttools.models.thermo import ThermoModel


def plot_g_cs2(thermo: ThermoModel, phase: Phase = Phase.SYMMETRIC, fig: plt.Figure = None) -> plt.Figure:
    if fig is None:
        fig = plt.figure()
    axs: np.ndarray
    axs = fig.subplots(nrows=3, ncols=1, sharex=True)
    fig.subplots_adjust(hspace=0)

    temp = np.logspace(thermo.GEFF_DATA_LOG_TEMP[0], thermo.GEFF_DATA_LOG_TEMP[-1], 100)
    axs[0].plot(temp, thermo.ge_gs_ratio(temp, phase), label=r"$g_e/g_s(T)$, spline")
    if hasattr(thermo, "GEFF_DATA_GE_GS_RATIO"):
        axs[0].scatter(thermo.GEFF_DATA_TEMP, thermo.GEFF_DATA_GE_GS_RATIO, label=r"$g_e/g_s(T)$, data")
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(temp, thermo.ge(temp, phase), label=r"$g_e(T)$, spline", color="blue")
    axs[1].plot(temp, thermo.gs(temp, phase), label=r"$g_s(T)$, spline", color="red")
    if hasattr(thermo, "GEFF_DATA_GE"):
        axs[1].scatter(thermo.GEFF_DATA_TEMP, thermo.GEFF_DATA_GE, label=r"$g_e(T)$, data")
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(temp, thermo.cs2(temp, phase), label="$c_s^2$, spline")
    axs[2].scatter(thermo.GEFF_DATA_TEMP, thermo.cs2_full(thermo.GEFF_DATA_TEMP, phase), label="$c_s^2$ from g-splines")
    axs[2].grid(True)
    axs[2].legend()
    axs[2].set_xlabel("T (MeV)")
    axs[2].set_xscale("log")

    return fig
