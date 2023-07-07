r"""
Plot $\kappa(\xi)$ for various models
=====================================
"""

import concurrent.futures as cf
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from examples import utils
from pttools.logging import setup_logging
from pttools.bubble.boundary import Phase
from pttools.bubble.bubble import Bubble
from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel

logger = logging.getLogger(__name__)


def alpha_thetabarn_to_alpha_n(model: ConstCSModel, alpha_thetabarn: float, wn: float):
    return alpha_thetabarn - 1/wn * (1 - 1/(3*model.csb2))*(model.p(wn, Phase.SYMMETRIC) - model.p(wn, Phase.BROKEN))


def kappa_vec(model: ConstCSModel, v_walls: np.ndarray, alpha_n: float) -> np.ndarray:
    kappas = np.empty_like(v_walls)
    for i, v_wall in enumerate(v_walls):
        try:
            bubble = Bubble(model, v_wall=v_wall, alpha_n=alpha_n)
            bubble.solve()
            kappas[i] = bubble.kappa
        except (IndexError, ValueError):
            continue
    return kappas


def main():
    t_start = time.perf_counter()
    alpha_ns = np.array([0.01, 0.03, 0.1, 0.3, 1, 3])
    colors = ["blue", "orange", "red", "green", "purple", "grey"]
    v_walls = np.linspace(0.2, 0.95, 10)
    lines = ["--", "-", "--", "-"]
    V_s = 1
    a_s = 1.1
    a_b = 1
    allow_invalid = True
    logger.debug("Loading models.")
    models = [
        BagModel(a_s=a_s, a_b=a_b, V_s=V_s, allow_invalid=allow_invalid),
        ConstCSModel(a_s=a_s, a_b=a_b, css2=1/3, csb2=1/4, V_s=V_s, allow_invalid=allow_invalid),
        ConstCSModel(a_s=a_s, a_b=a_b, css2=1/4, csb2=1/3, V_s=V_s, allow_invalid=allow_invalid),
        ConstCSModel(a_s=a_s, a_b=a_b, css2=1/4, csb2=1/4, V_s=V_s, allow_invalid=allow_invalid)
    ]
    logger.debug("Models loaded.")
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    futs = np.zeros((alpha_ns.size, len(models)), dtype=object)
    with cf.ProcessPoolExecutor(max_workers=len(os.sched_getaffinity(0))) as ex:
        for i_alpha, alpha_n in enumerate(alpha_ns):
            for i_model, model in enumerate(models):
                 futs[i_alpha, i_model] = ex.submit(kappa_vec, model, v_walls, alpha_n)
        for i_alpha, color in enumerate(colors):
            for i_model, ls in enumerate(lines):
                ax.plot(v_walls, futs[i_alpha, i_model].result(), color=color, ls=ls)

    ax.set_ylim(0.01, 10)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\xi_w$")
    ax.set_ylabel(r"$\kappa$")

    logger.info(f"Elapsed time: {time.perf_counter() - t_start}")
    return fig


if __name__ == "__main__":
    setup_logging()
    fig = main()
    utils.save_and_show(fig, "plot_xi_kappa.png")
