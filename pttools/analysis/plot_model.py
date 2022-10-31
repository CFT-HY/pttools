import matplotlib.pyplot as plt
import numpy as np

from pttools import models
from pttools.bubble import Phase


class ModelPlot:
    def __init__(self, model: models.Model, critical_temp_guess: float = 10):
        self.model = model

        self.fig: plt.Figure = plt.figure(figsize=(11.69, 8.27))
        self.axs: np.ndarray = self.fig.subplots(nrows=3, ncols=3)
        self.ax_p = self.axs[0, 0]
        self.ax_s = self.axs[0, 1]
        self.ax_w = self.axs[0, 2]
        self.ax_e = self.axs[1, 0]
        self.ax_cs2 = self.axs[1, 2]

        self.crit = model.critical_temp(critical_temp_guess)
        self.temps_b = np.linspace(0.7 * self.crit, self.crit)
        self.temps_s = np.linspace(self.crit, 1.3 * self.crit)

        self.plot(self.ax_p, self.model.p_temp, "p")
        self.plot(self.ax_s, self.model.s_temp, "s")
        self.plot(self.ax_w, self.model.w, "w")
        self.plot(self.ax_e, self.model.e_temp, "e")
        self.plot(self.ax_cs2, self.model.cs2_temp, label="c_s^2", label_s="$c_{s,s}^2$", label_b="$c_{s,b}^2$")

        self.fig.tight_layout()

    def plot(self, ax: plt.Axes, func: callable, label: str = None, label_s: str = None, label_b: str = None):
        if label_s is None and label is not None:
            label_s = f"${label}_s$"
        if label_b is None and label is not None:
            label_b = f"${label}_b$"
        ax.plot(self.temps_b, func(self.temps_b, Phase.BROKEN), color="b", label=label_s)
        ax.plot(self.temps_s, func(self.temps_s, Phase.BROKEN), color="b", ls=":")
        ax.plot(self.temps_b, func(self.temps_b, Phase.SYMMETRIC), color="r", ls=":")
        ax.plot(self.temps_s, func(self.temps_s, Phase.SYMMETRIC), color="r", label=label_b)

        ax.axvline(self.crit, ls=":", label=r"$T_{crit}$")
        ax.set_xlabel("$T$")
        ax.set_ylabel(f"${label}$")

        ax.legend()
