import logging
import typing as tp

from matplotlib.contour import QuadContourSet
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis import cmap
from pttools.analysis.bubble_grid import BubbleGridVWAlpha
from pttools.bubble.bubble import Bubble
from pttools.bubble.chapman_jouguet import v_chapman_jouguet

if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


def compute(bubble: Bubble):
    try:
        if bubble.no_solution_found or bubble.solver_failed:
            return np.nan
        return bubble.entropy_density_relative
    except IndexError as e:
        logger.exception("Fail", exc_info=e)
        return np.nan


def gen_and_plot_entropy(
        model: "Model",
        v_walls: np.ndarray,
        alpha_ns: np.ndarray,
        min_level: float,
        max_level: float,
        diff_level: float,
        use_bag_solver: bool = False) -> tp.Tuple[plt.Figure, plt.Axes]:
    grid = BubbleGridVWAlpha(model, v_walls, alpha_ns, compute, use_bag_solver=use_bag_solver)
    fig, ax = plot_entropy(
        grid.data.T, v_walls, alpha_ns,
        min_level=min_level, max_level=max_level, diff_level=diff_level)

    cmap.color_region(ax, v_walls, alpha_ns, grid.unphysical_alpha_plus().T, color="red", alpha=0.5)
    cmap.color_region(ax, v_walls, alpha_ns, grid.numerical_error().T, color="blue", alpha=0.5)
    cmap.color_region(ax, v_walls, alpha_ns, grid.solver_failed().T, color="green", alpha=0.5)

    ax.plot(v_chapman_jouguet(model, alpha_ns), alpha_ns, 'k--', label=r'$v_{CJ}$')
    ax.set_title(rf"$\Delta s / s_n$ for {model.label_latex}")
    ax.legend()

    plot_kappa_omega_sum(grid)

    return fig, ax


def plot_entropy(
        data: np.ndarray,
        v_walls: np.ndarray, alpha_ns: np.ndarray,
        min_level: float, max_level: float, diff_level: float,
        fig: plt.Figure = None,
        ax: plt.Axes = None) -> tp.Tuple[plt.Figure, plt.Axes]:
    if fig is None or ax is None:
        fig: plt.Figure = plt.figure()
        ax: plt.Axes = fig.add_subplot()

    levels, cols = cmap.cmap(min_level, max_level, diff_level)
    cs: QuadContourSet = ax.contourf(v_walls, alpha_ns, data, levels=levels, colors=cols)
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r'$\Delta s / s_n$')

    ax.grid()
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$\alpha_n$")
    ax.set_title(rf"$\Delta s / s_n$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    # ax.legend()

    return fig, ax


def plot_kappa_omega_sum(grid: BubbleGridVWAlpha, fig: plt.Figure = None, ax: plt.Axes = None):
    if fig is None or ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()

    kappa_omega_sum: np.ndarray = grid.kappa().astype(np.float_) + grid.omega().astype(np.float_)
    kappa_omega_sum = np.abs(kappa_omega_sum - 1)
    cs: QuadContourSet = ax.contourf(grid.v_walls, grid.alpha_ns, kappa_omega_sum.T, locator=ticker.LogLocator())
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r"$|\kappa + \omega - 1|$")

    ax.grid()
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$\alpha_n$")

    return fig, ax
