r"""Compare the thermodynamic quantities of multiple models."""

from __future__ import annotations

import typing as tp

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from pttools.plot import utils
import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from pttools import bubble, models

__all__ = ["ModelsPlot"]


class ModelsPlot:
    """A plot of the thermodynamic quantities of multiple models."""

    def __init__(self, temp: th.FloatArr1D):
        """:param temp: $T$, temperatures, used as the x-axis"""
        self.temp: th.FloatArr1D = temp

        self.fig: Figure = Figure(figsize=(11.69, 8.27))
        self.axs: th.AxesArr2D = self.fig.subplots(nrows=3, ncols=3)
        self.ax_p: Axes = self.axs[0, 0]
        self.ax_s: Axes = self.axs[0, 1]
        self.ax_w: Axes = self.axs[0, 2]
        self.ax_e: Axes = self.axs[1, 0]
        self.ax_temp: Axes = self.axs[1, 1]
        self.ax_cs2: Axes = self.axs[1, 2]
        self.ax_alpha_n: Axes = self.axs[2, 0]
        self.ax_alpha_plus: Axes = self.axs[2, 1]

        ax: Axes
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

    def add(self, model: models.Model, phase: bubble.Phase, ls: str = "-", **kwargs: tp.Any) -> None:
        """Add a model to the plot."""
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
            model.alpha_plus(
                w,
                model.w(self.temp/2, phase),
                error_on_invalid=False, nan_on_invalid=False, log_invalid=False),
            label=label, ls=ls, **kwargs
        )

    def process(self) -> None:
        """Process the plot."""
        ax: Axes
        for ax in self.axs.flat:
            utils.legend(ax, fontsize="x-small")
