"""Comparison of two suppression factor datasets."""

from __future__ import annotations

import typing as tp

import numpy as np

from pttools.bubble import CS0
from pttools.plot.suppression import plot_alpha_n_max
from pttools.plot.vw_alpha import VwAlphaPlot
from pttools.ssm.suppression import Suppression, SuppressionMethod
import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.axes import Axes
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure

__all__ = ["SuppressionComparisonPlot"]


class SuppressionComparisonPlot(VwAlphaPlot):
    r"""Plot the absolute or relative difference of two suppression factor datasets as a 2D contour plot.

    The absolute difference is $\Sigma_1 - \Sigma_2$, and the relative difference is
    $(\Sigma_1 - \Sigma_2) / \Sigma_2$.
    """

    def __init__(
            self,
            sup1: Suppression,
            sup2: Suppression,
            relative: bool = False,
            method1: SuppressionMethod = SuppressionMethod.NO_EXT,
            method2: SuppressionMethod = SuppressionMethod.NO_EXT,
            fig: Figure | None = None,
            ax: Axes | None = None,
            v_wall_min: float = 0,
            v_wall_max: float = 1,
            alpha_n_min: float = 0,
            alpha_n_max: float = 1,
            n_points: int = 200,
            diff_level: float | None = None,
            cs: float | None = CS0,
            title: str | None = None,
            alpha_n_max_lines: bool = True,
            points: bool = True,
            v_cj: bool = True,
            figsize: tuple[float, float] = (5, 4),
        ):
        r"""Interpolate the suppression factors to a grid, and plot their difference.

        :param sup1: $\Sigma_1$, the suppression dataset that is compared
        :param sup2: $\Sigma_2$, the reference suppression dataset
        :param relative: whether to plot the relative difference instead of the absolute difference
        :param method1: extrapolation method of ``sup1``.
            With the default :py:attr:`pttools.ssm.suppression.suppression.SuppressionMethod.NO_EXT`,
            the difference is plotted only within the convex hull of the data points.
        :param method2: extrapolation method of ``sup2``
        :param fig: figure to plot in. If None, a new figure is created.
        :param ax: axes to plot in. If None, new axes are created.
        :param v_wall_min: minimum $v_\text{wall}$ of the x-axis and the grid
        :param v_wall_max: maximum $v_\text{wall}$ of the x-axis and the grid
        :param alpha_n_min: minimum $\alpha_n$ of the y-axis and the grid
        :param alpha_n_max: maximum $\alpha_n$ of the y-axis and the grid
        :param n_points: number of grid points in each direction
        :param diff_level: difference between the contour levels.
            If None, 0.05 is used for the absolute difference and 0.1 for the relative difference.
        :param cs: $c_s$, speed of sound, plotted as a vertical line. If None, the line is not plotted.
        :param title: title of the plot. If None, it is generated from the labels of the datasets.
        :param alpha_n_max_lines: whether to plot the exact and approximate $\alpha_{n,\text{max}}$ curves
        :param points: whether to plot the points of the datasets
        :param v_cj: whether to plot the Chapman-Jouguet speed $v_\text{CJ}$ of the bag model
        :param figsize: size of the figure if a new figure is created
        """
        fig_was_none = fig is None
        if title is None:
            title = f"{'Relative' if relative else 'Absolute'} difference\n{sup1.label} vs. {sup2.label}"
        v_walls = np.linspace(v_wall_min, v_wall_max, n_points)
        alpha_ns = np.linspace(alpha_n_min, alpha_n_max, n_points)
        super().__init__(fig=fig, ax=ax, v_walls=v_walls, alpha_ns=alpha_ns, title=title, figsize=figsize)
        self.sup1: Suppression = sup1
        self.sup2: Suppression = sup2
        self.relative: bool = relative

        #: $\Sigma_1$ on the grid, indexed as ``[alpha_n, v_wall]``
        self.sup1_grid: th.FloatArr2D = tp.cast(
            th.FloatArr2D, sup1.suppression(v_walls, alpha_ns, method=method1))
        #: $\Sigma_2$ on the grid, indexed as ``[alpha_n, v_wall]``
        self.sup2_grid: th.FloatArr2D = tp.cast(
            th.FloatArr2D, sup2.suppression(v_walls, alpha_ns, method=method2))
        #: The plotted difference on the grid, indexed as ``[alpha_n, v_wall]``
        self.diff: th.FloatArr2D = self.sup1_grid - self.sup2_grid
        if relative:
            self.diff /= self.sup2_grid
            label = r"$(\Sigma_1 - \Sigma_2) / \Sigma_2$"
        else:
            label = r"$\Sigma_1 - \Sigma_2$"
        if diff_level is None:
            diff_level = 0.1 if relative else 0.05
        self.cbar: Colorbar = self.contourf_plusminus(self.diff, label=label, diff_level=diff_level)

        if alpha_n_max_lines:
            plot_alpha_n_max(self.ax)
        if cs is not None:
            self.sound_speed(cs)
        if v_cj:
            self.chapman_jouguet_bag()
        if points:
            self.ax.scatter(sup1.v_walls, sup1.alpha_ns, label=r"$\Sigma_1$ dataset", s=10, marker="o")
            self.ax.scatter(sup2.v_walls, sup2.alpha_ns, label=r"$\Sigma_2$ dataset", s=10, marker="x")

        self.legend()
        if fig_was_none:
            self.fig.tight_layout()
