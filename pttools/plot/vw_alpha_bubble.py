r"""A plot of a grid of bubbles with $v_\text{wall}$ on the x-axis and $\alpha_n$ on the y-axis."""

from __future__ import annotations

from pttools.bubble.chapman_jouguet import v_chapman_jouguet
from pttools.plot.vw_alpha import VwAlphaPlot
import pttools.type_hints as th

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    from pttools.analysis.bubble_grid import BubbleGridVWAlpha

__all__ = ["VwAlphaBubblePlot"]


class VwAlphaBubblePlot(VwAlphaPlot):
    r"""A plot of a grid of bubbles with $v_\text{wall}$ on the x-axis and $\alpha_n$ on the y-axis."""

    def __init__(
            self,
            grid: BubbleGridVWAlpha,
            fig: Figure | None = None,
            ax: Axes | None = None,
            title: str | None = None,
            alpha_label: str = r"$\alpha_n$"):
        r"""Create the axes for the plot. Add data to it with the other methods.

        :param grid: grid of bubbles, whose $v_\text{wall}$ and $\alpha_n$ values define the grid and the axis limits
        :param fig: figure to plot in. If None, a new figure is created.
        :param ax: axes to plot in. If None, new axes are created in the figure. Requires fig.
        :param title: title of the plot
        :param alpha_label: label of the y-axis
        :raises ValueError: if ax is given without fig
        """
        if fig is None and ax is not None:
            raise ValueError("Cannot provide ax without fig")
        super().__init__(
            fig=fig, ax=ax, v_walls=grid.v_walls, alpha_ns=grid.alpha_ns, title=title, alpha_label=alpha_label)
        self.grid: BubbleGridVWAlpha = grid

    def chapman_jouguet(self, color: str = "black", ls: str = "--", label: str = "$v_{CJ}$") -> list[Line2D]:
        """Add the Chapman-Jouguet speed curve of the model of the grid to the plot."""
        return self.ax.plot(
            v_chapman_jouguet(self.grid.model, self.grid.alpha_ns),
            self.grid.alpha_ns, color=color, ls=ls, label=label
        )
