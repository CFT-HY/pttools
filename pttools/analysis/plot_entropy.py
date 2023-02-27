import logging
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis import cmap, parallel
from pttools.analysis.bubble_grid import BubbleGridVWAlpha
from pttools.bubble.boundary import Phase
from pttools.bubble.bubble import Bubble
from pttools.bubble.chapman_jouguet import v_chapman_jouguet
# from pttools.speedup import parallel

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


def plot_entropy(model: "Model", v_walls: np.ndarray, alpha_ns: np.ndarray):
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()

    grid = BubbleGridVWAlpha(model, v_walls, alpha_ns, compute)

    levels, cols = cmap.cmap(-0.3, 0.4, 0.05)
    cs = ax.contourf(v_walls, alpha_ns, grid.data.T, levels=levels, colors=cols)
    cbar = fig.colorbar(cs)
    cbar.ax.set_ylabel(r'$\Delta s / s_n$')

    ax.contour(v_walls, alpha_ns, grid.unphysical_alpha_plus())

    ax.plot(v_chapman_jouguet(model, alpha_ns), alpha_ns, 'k--', label=r'$v_{CJ}$')

    ax.grid()
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$\alpha_n$")
    ax.set_title(rf"$\Delta s / s_n$ for {model.label_latex}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    return fig
