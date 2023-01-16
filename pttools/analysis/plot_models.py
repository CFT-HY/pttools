r"""Compare the thermodynamic quantities of multiple models"""

import matplotlib.pyplot as plt
import numpy as np

from pttools import bubble, models
from pttools.analysis import utils


class ModelsPlot:
    def __init__(self, temp: np.ndarray):
        self.temp = temp

        self.fig: plt.Figure = plt.figure(figsize=(11.69, 8.27))
        self.axs: np.ndarray = self.fig.subplots(nrows=3, ncols=3)
        self.ax_p = self.axs[0, 0]
        self.ax_s = self.axs[0, 1]
        self.ax_w = self.axs[0, 2]
        self.ax_e = self.axs[1, 0]
        self.ax_temp = self.axs[1, 1]
        self.ax_cs2 = self.axs[1, 2]
        self.ax_alpha_n = self.axs[2, 0]
        self.ax_alpha_plus = self.axs[2, 1]

        for ax in [self.ax_p, self.ax_s, self.ax_w, self.ax_e, self.ax_alpha_n, self.ax_alpha_plus]:
            ax.set_yscale("log")

        for ax in self.axs.flat:
            ax.set_xlabel("T")
            ax.set_xscale("log")

        self.ax_p.set_ylabel("p")
        self.ax_s.set_ylabel("s")
        self.ax_w.set_ylabel("w")
        self.ax_e.set_ylabel("e")

        self.ax_temp.set_xlabel("w")
        self.ax_temp.set_ylabel("T")

        self.ax_cs2.set_ylabel("$c_s^2$")
        self.ax_cs2.set_ylim(0, 1)

        self.ax_alpha_n.set_ylabel(r"$\alpha_n$")
        self.ax_alpha_plus.set_ylabel(r"$\alpha_+ \left( T_-=\frac{T_+}{2} \right)$")

        self.fig.tight_layout()

    def add(self, model: models.Model, phase: bubble.Phase, ls: str = "-", **kwargs):
        label = utils.model_phase_label(model, phase)
        w = model.w(self.temp, phase)

        self.ax_p.plot(self.temp, model.p_temp(self.temp, phase), label=label, ls=ls, **kwargs)
        self.ax_s.plot(self.temp, model.s_temp(self.temp, phase), label=label, ls=ls, **kwargs)
        self.ax_w.plot(self.temp, w, label=label, ls=ls, **kwargs)
        self.ax_e.plot(self.temp, model.e_temp(self.temp, phase), label=label, ls=ls, **kwargs)

        self.ax_temp.plot(w, model.temp(w, phase), label=label, ls=ls, **kwargs)
        self.ax_cs2.plot(self.temp, model.cs2(w, phase), label=label, ls=ls, **kwargs)
        self.ax_alpha_n.plot(self.temp, model.alpha_n(w), label=label, ls=ls, **kwargs)
        self.ax_alpha_plus.plot(
            self.temp,
            model.alpha_plus(w, model.w(self.temp/2, phase)), label=label, ls=ls, **kwargs)

    def process(self):
        for ax in self.axs.flat:
            ax.legend(fontsize="x-small")
