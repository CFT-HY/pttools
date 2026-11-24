r"""A plot with $v_\text{wall}$ on the x-axis and $\alpha_n$ on the y-axis."""

from __future__ import annotations

import typing as tp

import numpy as np

from pttools.bubble.chapman_jouguet import v_chapman_jouguet_bag
from pttools.plot import colormap
from pttools.plot.colormap import DEFAULT_REGION_COLOR
from pttools.plot.utils import create_fig_ax
import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.colors import Colormap
    from matplotlib.contour import QuadContourSet
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D

__all__ = ["VwAlphaPlot"]


class VwAlphaPlot:
    r"""A plot with $v_\text{wall}$ on the x-axis and $\alpha_n$ on the y-axis.

    2D data, such as that of :py:meth:`contourf_plusminus` and :py:meth:`color_region`,
    is given on the grid defined by :py:attr:`v_walls` and :py:attr:`alpha_ns`, indexed as ``[alpha_n, v_wall]``.
    """

    def __init__(
            self,
            fig: Figure | None = None,
            ax: Axes | None = None,
            v_walls: th.FloatArr1D | None = None,
            alpha_ns: th.FloatArr1D | None = None,
            v_wall_min: float | None = None,
            v_wall_max: float | None = None,
            alpha_n_min: float | None = None,
            alpha_n_max: float | None = None,
            title: str | None = None,
            alpha_label: str = r"$\alpha_n$",
            figsize: tuple[float, float] | None = None):
        r"""Create the axes for the plot. Add data to it with the other methods.

        :param fig: figure to plot in. If None, a new figure is created.
        :param ax: axes to plot in. If None, new axes are created.
        :param v_walls: $v_\text{wall}$, the x-coordinates of the grid of the 2D data
        :param alpha_ns: $\alpha_n$, the y-coordinates of the grid of the 2D data
        :param v_wall_min: minimum $v_\text{wall}$ of the x-axis.
            If None, the minimum of ``v_walls`` is used, or 0 if ``v_walls`` is None.
        :param v_wall_max: maximum $v_\text{wall}$ of the x-axis.
            If None, the maximum of ``v_walls`` is used, or 1 if ``v_walls`` is None.
        :param alpha_n_min: minimum $\alpha_n$ of the y-axis.
            If None, the minimum of ``alpha_ns`` is used, or 0 if ``alpha_ns`` is None.
        :param alpha_n_max: maximum $\alpha_n$ of the y-axis.
            If None, the maximum of ``alpha_ns`` is used, or 1 if ``alpha_ns`` is None.
        :param title: title of the plot
        :param alpha_label: label of the y-axis
        :param figsize: size of the figure if a new figure is created
        """
        self.fig: Figure
        self.ax: Axes
        self.fig, self.ax = create_fig_ax(fig, ax, figsize=figsize)
        #: $v_\text{wall}$, the x-coordinates of the grid of the 2D data
        self.v_walls: th.FloatArr1D | None = v_walls
        #: $\alpha_n$, the y-coordinates of the grid of the 2D data
        self.alpha_ns: th.FloatArr1D | None = alpha_ns

        if v_wall_min is None:
            v_wall_min = 0 if v_walls is None else v_walls.min()
        if v_wall_max is None:
            v_wall_max = 1 if v_walls is None else v_walls.max()
        if alpha_n_min is None:
            alpha_n_min = 0 if alpha_ns is None else alpha_ns.min()
        if alpha_n_max is None:
            alpha_n_max = 1 if alpha_ns is None else alpha_ns.max()

        self.ax.set_xlabel(r"$v_\text{wall}$")
        self.ax.set_ylabel(alpha_label)
        self.ax.set_xlim(v_wall_min, v_wall_max)
        self.ax.set_ylim(alpha_n_min, alpha_n_max)
        self.ax.grid()
        if title is not None:
            self.ax.set_title(title)

    def _grid(self) -> tuple[th.FloatArr1D, th.FloatArr1D]:
        if self.v_walls is None or self.alpha_ns is None:
            raise ValueError("Plotting 2D data requires the grid coordinates v_walls and alpha_ns.")
        return self.v_walls, self.alpha_ns

    def chapman_jouguet_bag(
            self,
            n_points: int = 20,
            color: str | None = None,
            ls: str = "--",
            label: str = r"$v_\text{CJ}$") -> list[Line2D]:
        r"""Add the Chapman-Jouguet speed $v_\text{CJ}(\alpha_n)$ of the bag model to the plot.

        :param n_points: number of points of the curve
        :param color: color of the curve. If None, the next color of the color cycle is used.
        :param ls: line style of the curve
        :param label: label of the curve
        :return: the plotted lines
        """
        alpha_n_min, alpha_n_max = self.ax.get_ylim()
        alpha_ns = np.linspace(alpha_n_min, alpha_n_max, n_points)
        return self.ax.plot(v_chapman_jouguet_bag(alpha_ns), alpha_ns, color=color, ls=ls, label=label)

    def color_region(
            self,
            region: th.BoolArr | th.FloatArr,
            color: str = DEFAULT_REGION_COLOR,
            alpha: float = 1) -> QuadContourSet:
        """Color a region with a fixed color."""
        v_walls, alpha_ns = self._grid()
        return colormap.color_region(
            ax=self.ax, x=v_walls, y=alpha_ns,
            region=region, color=color, alpha=alpha
        )

    def contourf_plusminus(
            self,
            data: th.FloatArr2D,
            label: str,
            diff_level: float | None = None,
            cmap_neg: Colormap | str = colormap.DEFAULT_CMAP_NEG,
            cmap_pos: Colormap | str = colormap.DEFAULT_CMAP_POS) -> Colorbar:
        """Add a contour plot with different color maps for the positive and negative values to the figure.

        :param data: 2D data on the grid of :py:attr:`v_walls` and :py:attr:`alpha_ns`
        :param label: label of the color bar
        :param diff_level: difference between the contour levels. If None, 0.1 is used.
        :param cmap_neg: color map of the negative values
        :param cmap_pos: color map of the positive values
        :return: the color bar
        """
        v_walls, alpha_ns = self._grid()
        if diff_level is None:
            diff_level = 0.1
        levels, colors = colormap.cmap_plusminus(
            min_level=np.nanmin(data), max_level=np.nanmax(data), diff_level=diff_level,
            cmap_neg=cmap_neg, cmap_pos=cmap_pos
        )
        cs: QuadContourSet = self.ax.contourf(v_walls, alpha_ns, data, levels=levels, colors=colors)
        cbar = self.fig.colorbar(cs, ax=self.ax)
        cbar.ax.set_ylabel(label)
        return cbar

    def legend(self, *args: tp.Any, **kwargs: tp.Any) -> Legend:
        """Add a legend to the plot."""
        return self.ax.legend(*args, **kwargs)

    def sound_speed(self, cs: float, color: str = "k", ls: str = ":", label: str = "$c_s$") -> Line2D:
        """Add a vertical line at the speed of sound to the plot.

        :param cs: $c_s$, speed of sound
        :param color: color of the line
        :param ls: line style of the line
        :param label: label of the line
        :return: the plotted line
        """
        return self.ax.axvline(x=cs, label=label, ls=ls, c=color)
