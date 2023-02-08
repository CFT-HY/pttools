import logging
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis import parallel
from pttools.bubble.boundary import Phase
from pttools.bubble.bubble import Bubble
# from pttools.speedup import parallel

if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


def compute(bubble: Bubble):
    try:
        if hasattr(bubble, "failed") and bubble.failed:
            return np.nan
        # if bubble.unphysical:
        #     return np.nan
        if bubble.no_solution_found:
            return np.nan
        return bubble.entropy_density / bubble.model.s(bubble.wn, Phase.SYMMETRIC)
    except IndexError as e:
        logger.exception("Fail", exc_info=e)
        return np.nan


def plot_entropy(model: "Model", v_walls: np.ndarray, alpha_ns: np.ndarray):
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()

    entropy_densities = parallel.create_bubbles(model, v_walls, alpha_ns, compute)
    print(entropy_densities.ndim)

    # bubbles = np.empty((v_walls.size, alpha_ns.size), dtype=object)
    # for i_v_wall, v_wall in enumerate(v_walls):
    #     for i_alpha_n, alpha_n in enumerate(alpha_ns):
    #         bubbles[i_v_wall, i_alpha_n] = Bubble(model, v_wall, alpha_n)
    #
    # entropy_densities = parallel.run_parallel(compute, bubbles)

    cs = ax.contourf(v_walls, alpha_ns, entropy_densities.T)
    fig.colorbar(cs)
    ax.set_xlabel(r"$v_w$")
    ax.set_ylabel(r"$\alpha_n$")
    ax.set_title(rf"$\Delta s / s_n$ for {model.label_latex}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, np.max(alpha_ns))

    return fig
