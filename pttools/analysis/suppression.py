import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
import numpy as np

from pttools.analysis.utils import create_fig_ax
from pttools.bubble import v_chapman_jouguet_bag
from pttools.omgw0.suppression import Suppression, alpha_n_max_approx
from pttools.omgw0.suppression import alpha_n_max as alpha_n_max_func


class SuppressionPlot:
    r"""Plot the suppression data as a 2D contour plot

    :gowling_2021:`\ ` fig. 10
    """
    def __init__(
            self,
            sup: Suppression,
            fig: plt.Figure = None,
            ax: plt.Axes = None,
            v_wall_min: float = 0,
            v_wall_max: float = 1,
            alpha_n_min: float = 0,
            alpha_n_max: float = 1,
            title: str = None,
            alpha_n_max_lines: bool = True,
            v_cj: bool = True,
            figsize: tuple[float, float] = (4, 3),
            # levels: np.ndarray[int, np.float64] = np.array([0.01, 0.03, 0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 2.5])
        ):
        fig_was_none = fig is None
        self.fig, self.ax = create_fig_ax(fig, ax, figsize=figsize)
        self.sup = sup
        self.tri = Triangulation(self.sup.v_walls, self.sup.alpha_ns)

        tricontourf = self.ax.tricontourf(self.tri, self.sup.suppressions, cmap="plasma")  #, levels=levels)
        self.cbar = self.ax.figure.colorbar(tricontourf, ax=self.ax)
        self.cbar.ax.set_ylabel(r"$\Sigma$")

        if alpha_n_max_lines:
            self.line_v_walls = np.linspace(0, 1, 20, endpoint=False)
            self.ax.plot(self.line_v_walls, alpha_n_max_func(self.line_v_walls), label=r"$\alpha_{n,\text{max}}$")
            self.ax.plot(self.line_v_walls, alpha_n_max_approx(self.line_v_walls), label=r"$\alpha_{n,\text{max,approx}}$")
        if v_cj:
            alpha_ns = np.linspace(alpha_n_min, alpha_n_max, 20)
            self.ax.plot(v_chapman_jouguet_bag(alpha_ns), alpha_ns, label=r"$v_\text{CJ}$", ls="--")

        self.ax.set_xlabel(r"$v_\text{wall}$")
        self.ax.set_ylabel(r"$\alpha_n$")
        self.ax.set_xlim(v_wall_min, v_wall_max)
        self.ax.set_ylim(alpha_n_min, alpha_n_max)
        self.ax.set_title(self.sup.name if title is None else title)
        self.ax.legend()
        if fig_was_none:
            self.fig.tight_layout()
