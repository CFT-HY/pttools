"""Plot thermodynamic quantities for a model"""

import logging

import matplotlib.pyplot as plt
import numpy as np

from pttools import models
from pttools.bubble import Phase

logger = logging.getLogger(__name__)


class ModelPlot:
    def __init__(
            self,
            model: models.Model,
            t_min: float = None, t_max: float = None,
            t_log_range: float = 2,
            t_log: bool = True,
            y_log: bool = True,
            n_points: int = 20):
        self.model = model
        self.t_log = t_log

        self.fig: plt.Figure = plt.figure(figsize=(11.69, 8.27))
        self.axs: np.ndarray = self.fig.subplots(nrows=3, ncols=3)
        self.ax_p = self.axs[0, 0]
        self.ax_s = self.axs[0, 1]
        self.ax_w = self.axs[0, 2]
        self.ax_e = self.axs[1, 0]
        self.ax_cs2 = self.axs[1, 1]
        # self.ax_theta = self.axs[2, 0]

        if t_log:
            self.t_min = max(model.t_min, 10**(-t_log_range)*model.t_crit) if t_min is None else t_min
            self.t_max = min(model.t_max, 10**t_log_range*model.t_crit) if t_max is None else t_max
            self.temps_b = np.logspace(np.log10(self.t_min), np.log10(model.t_crit), n_points)
            self.temps_s = np.logspace(np.log10(model.t_crit), np.log10(self.t_max), n_points)
        else:
            self.t_min = max(model.t_min, 0.7 * model.t_crit) if t_min is None else t_min
            self.t_max = min(model.t_max, 1.3 * model.t_crit) if t_max is None else t_max
            self.temps_b = np.linspace(self.t_min, model.t_crit, n_points)
            self.temps_s = np.linspace(model.t_crit, self.t_max, n_points)

        self.plot(self.ax_p, self.model.p_temp, "p", y_log=y_log)
        self.plot(self.ax_s, self.model.s_temp, "s", y_log=y_log)
        self.plot(self.ax_w, self.model.w, "w", y_log=y_log)
        self.plot(self.ax_e, self.model.e_temp, "e", y_log=y_log)
        self.plot(
            self.ax_cs2, self.model.cs2_temp,
            label="c_s^2", label_s="$c_{s,s}^2$", label_b="$c_{s,b}^2$", y_lim=False, y_log=False)

        self.fig.tight_layout()

    def plot(
            self,
            ax: plt.Axes, func: callable,
            label: str = None, label_s: str = None, label_b: str = None,
            y_lim: bool = True, y_log: bool = True):
        if label_s is None and label is not None:
            label_s = f"${label}_s$"
        if label_b is None and label is not None:
            label_b = f"${label}_b$"
        vals_s_low = func(self.temps_b, Phase.SYMMETRIC)
        vals_s_high = func(self.temps_s, Phase.SYMMETRIC)
        vals_b_low = func(self.temps_b, Phase.BROKEN)
        vals_b_high = func(self.temps_s, Phase.BROKEN)

        if y_log:
            vals_s_low_nan = vals_s_low < 0
            vals_b_low_nan = vals_b_low < 0
            if np.any((vals_s_low_nan, vals_b_low_nan)):
                logger.warning("Detected points p < 0. Setting these to nan due to logarithmic axes.")
                vals_s_low[vals_s_low_nan] = np.nan
                vals_b_low[vals_b_low_nan] = np.nan

        ax.plot(self.temps_b, vals_s_low, color="r", ls=":")
        ax.plot(self.temps_s, vals_s_high, color="r", label=label_s)
        ax.plot(self.temps_b, vals_b_low, color="b", label=label_b)
        ax.plot(self.temps_s, vals_b_high, color="b", ls=":")

        ax.axvline(self.model.t_crit, ls=":", label=r"$T_{crit}$")
        ax.set_xlabel("$T$")
        ax.set_ylabel(f"${label}$")

        ax.set_xlim(self.t_min, self.t_max)
        if y_lim:
            vals_s_min: float = np.nanmin((vals_s_low, vals_s_high))
            vals_b_min: float = np.nanmin((vals_b_low, vals_b_high))
            vals_min = min(vals_s_min, vals_b_min)
            vals_s_max: float = np.nanmax((vals_s_low, vals_s_high))
            vals_b_max: float = np.nanmax((vals_b_low, vals_b_high))
            vals_max = max(vals_s_max, vals_b_max)
            if 1.1 * vals_s_min > vals_b_max or 1.1 * vals_b_min > vals_s_max:
                ax.set_ylim(vals_min, vals_max)
            else:
                ax.set_ylim(vals_min, min(vals_s_max, vals_b_max))

        if self.t_log:
            ax.set_xscale("log")
        if y_log:
            ax.set_yscale("log")
        ax.legend()
