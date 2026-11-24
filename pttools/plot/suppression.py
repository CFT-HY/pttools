"""Analysis utilities for the suppression factor."""

from __future__ import annotations

import logging

from matplotlib.tri import Triangulation
import numpy as np

from pttools.bubble import CS0
from pttools.plot.vw_alpha import VwAlphaPlot
from pttools.ssm.suppression import Suppression, alpha_n_max_bag_approx
from pttools.ssm.suppression import alpha_n_max_dataset as alpha_n_max_func
import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

__all__ = ["SuppressionPlot", "plot_alpha_n_max"]

logger: logging.Logger = logging.getLogger(__name__)


class SuppressionPlot(VwAlphaPlot):
    r"""Plot the suppression data as a 2D contour plot.

    :gowling_2021:`\ ` fig. 10
    """

    def __init__(
            self,
            sup: Suppression,
            fig: Figure | None = None,
            ax: Axes | None = None,
            v_wall_min: float = 0,
            v_wall_max: float = 1,
            alpha_n_min: float = 0,
            alpha_n_max: float = 1,
            cs: float | None = CS0,
            cmap: str = "plasma",
            levels: th.FloatArr1D | None = None,
            title: str | None = None,
            alpha_n_max_lines: bool = True,
            points: bool = True,
            v_cj: bool = True,
            figsize: tuple[float, float] = (5, 4),
        ):
        r"""Create the contour plot of the suppression factor $\Sigma$.

        :param sup: suppression data
        :param fig: figure to plot in. If None, a new figure is created.
        :param ax: axes to plot in. If None, new axes are created.
        :param v_wall_min: minimum $v_\text{wall}$ of the x-axis
        :param v_wall_max: maximum $v_\text{wall}$ of the x-axis
        :param alpha_n_min: minimum $\alpha_n$ of the y-axis
        :param alpha_n_max: maximum $\alpha_n$ of the y-axis
        :param cs: $c_s$, speed of sound, plotted as a vertical line. If None, the line is not plotted.
        :param cmap: colormap for the contour plot
        :param levels: contour levels. If None, they are chosen automatically.
        :param title: title of the plot. If None, the label of the suppression data is used.
        :param alpha_n_max_lines: whether to plot the exact and approximate $\alpha_{n,\text{max}}$ curves
        :param points: whether to plot the points of the dataset
        :param v_cj: whether to plot the Chapman-Jouguet speed $v_\text{CJ}$ of the bag model
        :param figsize: size of the figure if a new figure is created
        """
        fig_was_none = fig is None
        super().__init__(
            fig=fig, ax=ax,
            v_wall_min=v_wall_min, v_wall_max=v_wall_max, alpha_n_min=alpha_n_min, alpha_n_max=alpha_n_max,
            title=sup.label if title is None else title, figsize=figsize
        )
        self.sup: Suppression = sup
        self.tri: Triangulation = Triangulation(self.sup.v_walls, self.sup.alpha_ns)

        # The type hints of Matplotlib do not accept levels=None, even though it's the default value.
        tricontourf = self.ax.tricontourf(  # pyrefly: ignore[no-matching-overload]
            self.tri, self.sup.suppressions, cmap=cmap, levels=levels
        )
        self.cbar: Colorbar = self.ax.figure.colorbar(tricontourf, ax=self.ax)
        self.cbar.ax.set_ylabel(r"$\Sigma$")

        if alpha_n_max_lines:
            plot_alpha_n_max(self.ax)
        if cs is not None:
            self.sound_speed(cs)
        if v_cj:
            self.chapman_jouguet_bag()
        if points:
            self.ax.scatter(self.sup.v_walls, self.sup.alpha_ns, label="dataset", s=10)

        self.legend()
        if fig_was_none:
            self.fig.tight_layout()

        # logger.info("Peak of suppression: v_wall=%s, alpha_n=%s, suppression=%s", *self.sup.peak())


def plot_alpha_n_max(ax: Axes, n_points: int = 20) -> list[Line2D]:
    r"""Plot the $\alpha_{n,\text{max}}({v}_\text{wall})$ curves of the suppression dataset and of the bag model.

    See :py:func:`pttools.ssm.suppression.suppression.alpha_n_max_dataset`
    and :py:func:`pttools.ssm.suppression.suppression.alpha_n_max_bag_approx`.

    :param ax: axes to plot in
    :param n_points: number of points of the curves
    :return: the plotted lines
    """
    v_walls = np.linspace(0, 1, n_points, endpoint=False)
    return \
        ax.plot(v_walls, alpha_n_max_func(v_walls), label=r"$\alpha_{n,\text{max,dataset}}$") + \
        ax.plot(v_walls, alpha_n_max_bag_approx(v_walls), label=r"$\alpha_{n,\text{max,bag}}$")
