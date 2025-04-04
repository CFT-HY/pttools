r"""Plot $g_\text{eff}$ and $c_s^2$"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble.boundary import Phase
from pttools.models.thermo import ThermoModel


def plot_g_cs2(thermo: ThermoModel, phase: Phase = Phase.SYMMETRIC, fig: plt.Figure = None) -> plt.Figure:
    if fig is None:
        fig = plt.figure()
    axs = fig.subplots(nrows=3, ncols=1, sharex=True)
    ax1: plt.Axes = axs[0]
    ax2: plt.Axes = axs[1]
    ax3: plt.Axes = axs[2]
    fig.subplots_adjust(hspace=0)

    temp = np.logspace(thermo.GEFF_DATA_LOG_TEMP[0], thermo.GEFF_DATA_LOG_TEMP[-1], 100)
    ax1.plot(temp, thermo.ge_gs_ratio(temp, phase), label=r"$g_e/g_s(T)$, spline")
    if hasattr(thermo, "GEFF_DATA_GE_GS_RATIO"):
        ax1.scatter(thermo.GEFF_DATA_TEMP, thermo.GEFF_DATA_GE_GS_RATIO, label=r"$g_e/g_s(T)$, data")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(temp, thermo.ge(temp, phase), label=r"$g_e(T)$, spline", color="blue")
    ax2.plot(temp, thermo.gs(temp, phase), label=r"$g_s(T)$, spline", color="red")
    if hasattr(thermo, "GEFF_DATA_GE"):
        ax2.scatter(thermo.GEFF_DATA_TEMP, thermo.GEFF_DATA_GE, label=r"$g_e(T)$, data")
    ax2.grid(True)
    ax2.legend()

    ax3.plot(temp, thermo.cs2(temp, phase), label="$c_s^2$, spline")
    ax3.scatter(thermo.GEFF_DATA_TEMP, thermo.cs2_full(thermo.GEFF_DATA_TEMP, phase), label="$c_s^2$ from g-splines")
    ax3.grid(True)
    ax3.legend()
    ax3.set_xlabel("T (MeV)")
    ax3.set_xscale("log")

    return fig
