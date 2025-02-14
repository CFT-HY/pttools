from matplotlib.contour import QuadContourSet
import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.bubble_grid import BubbleGridVWAlpha
from pttools.analysis import colormap
from pttools.analysis.colormap import REGION_COLOR_DEFAULT
from pttools.bubble.chapman_jouguet import v_chapman_jouguet


class VwAlphaPlot:
    def __init__(
            self,
            grid: BubbleGridVWAlpha,
            fig: plt.Figure = None,
            ax: plt.Axes = None,
            title: str = None,
            alpha_label: str = r"$\alpha_n$"):
        self.grid = grid

        if fig is None:
            if ax is not None:
                raise ValueError("Cannot provide ax without fig")
            self.fig = plt.figure()
        else:
            self.fig = fig

        self.ax = self.fig.add_subplot() if ax is None else ax
        self.ax.set_xlabel(r"$v_\text{wall}$")
        self.ax.set_ylabel(alpha_label)
        self.ax.set_xlim(grid.v_walls.min(), grid.v_walls.max())
        self.ax.set_ylim(grid.alpha_ns.min(), grid.alpha_ns.max())
        # self.ax.set_xlim(0, 1)
        # self.ax.set_ylim(0, 1)
        self.ax.grid()
        if title is not None:
            self.ax.set_title(title)
        # self.ax.legend()

    def contourf(
            self,
            data: np.ndarray,
            label: str,
            diff_level: float = None,
            cmap_neg_name: str = colormap.CMAP_NEG_DEFAULT,
            cmap_pos_name: str = colormap.CMAP_POS_DEFAULT):
        if diff_level is None:
            diff_level = 0.1
        levels, colors = colormap.cmap(
            min_level=np.nanmin(data), max_level=np.nanmax(data), diff_level=diff_level,
            cmap_neg_name=cmap_neg_name, cmap_pos_name=cmap_pos_name
        )
        cs: QuadContourSet = self.ax.contourf(
            self.grid.v_walls, self.grid.alpha_ns, data, levels=levels, colors=colors)
        cbar = self.fig.colorbar(cs)
        cbar.ax.set_ylabel(label)

    def color_region(self, region: np.ndarray, color: str = REGION_COLOR_DEFAULT, alpha: float = 1):
        colormap.color_region(
            ax=self.ax, x=self.grid.v_walls, y=self.grid.alpha_ns,
            region=region, color=color, alpha=alpha
        )

    def chapman_jouguet(self, ls: str = "k--", label: str = "$v_{CJ}$"):
        self.ax.plot(
            v_chapman_jouguet(self.grid.model, self.grid.alpha_ns), self.grid.alpha_ns, ls=ls, label=label)
