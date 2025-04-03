r"""Compare the properties of multiple thermodynamic models"""

import matplotlib.pyplot as plt
import numpy as np

from pttools import bubble, models
from pttools.analysis import utils


class ThermoModelsPlot:
    def __init__(self, temp: np.ndarray):
        self.temp = temp

        self.fig: plt.Figure = plt.figure()
        self.axs: np.ndarray = self.fig.subplots(nrows=2, ncols=2)

        self.ax_cs2 = self.axs[0, 0]
        self.ax_g = self.axs[0, 1]
        self.ax_dg_dT = self.axs[1, 0]
        self.ax_dx_dT = self.axs[1, 1]

        self.ax_cs2.set_ylabel("$c_s^2$")
        self.ax_g.set_ylabel("g")
        self.ax_dg_dT.set_ylabel(r"$\frac{dg}{dT}$")
        self.ax_dx_dT.set_ylabel(r"$\frac{dx}{dT}$")
        self.ax_dx_dT.set_yscale("log")

        for ax in self.axs.flat:
            ax.set_xscale("log")
            ax.set_xlabel("T")

        self.fig.tight_layout()

    def add(self, model: models.ThermoModel, phase: bubble.Phase, **kwargs):
        label = utils.model_phase_label(model, phase)
        self.ax_cs2.plot(self.temp, model.cs2(self.temp, phase), label=label, **kwargs)
        self.ax_cs2.plot(self.temp, model.cs2_full(self.temp, phase), label=f"{label}, full", **kwargs)

        self.ax_g.plot(self.temp, model.ge(self.temp, phase), label=f"$g_e$, {label}", **kwargs)
        self.ax_g.plot(self.temp, model.gs(self.temp, phase), label=f"$g_s$, {label}", **kwargs)

        self.ax_dg_dT.plot(self.temp, model.dge_dT(self.temp, phase), label=rf"$\frac{{dg_e}}{{dT}}$" + label, **kwargs)
        self.ax_dg_dT.plot(self.temp, model.dgs_dT(self.temp, phase), label=rf"$\frac{{dg_s}}{{dT}}$" + label, **kwargs)

        self.ax_dx_dT.plot(self.temp, model.de_dt(self.temp, phase), label=label, **kwargs)
        self.ax_dx_dT.plot(self.temp, model.dp_dt(self.temp, phase), label=label, **kwargs)

    def process(self):
        ax: plt.Axes
        for ax in self.axs.flat:
            utils.legend(ax, fontsize="x-small")
