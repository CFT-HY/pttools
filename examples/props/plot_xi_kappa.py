r"""
Plot κ(ξ) for various models
=================================
"""

import logging
import time

import matplotlib.pyplot as plt
import numpy as np

from examples import utils
from pttools.logging import setup_logging
from pttools.bubble.boundary import Phase
from pttools.bubble.bubble import Bubble
from pttools.models.const_cs import ConstCSModel
from pttools.bubble.fluid_reference import ref
from pttools.speedup.parallel import run_parallel

logger = logging.getLogger(__name__)


def alpha_thetabarn_to_alpha_n(model: ConstCSModel, alpha_thetabarn: float, wn: float):
    return alpha_thetabarn - 1/wn * (1 - 1/(3*model.csb2))*(model.p(wn, Phase.SYMMETRIC) - model.p(wn, Phase.BROKEN))


def kappa_vec(params: np.ndarray, v_walls: np.ndarray) -> np.ndarray:
    model, alpha_n = params
    kappas = np.ones_like(v_walls) * np.nan
    v_wall: float
    for i, v_wall in enumerate(v_walls):
        try:
            bubble = Bubble(model, v_wall=v_wall, alpha_n=alpha_n)
            kappas[i] = bubble.kappa
        except (IndexError, ValueError, RuntimeError):
            continue
    return kappas


def main():
    ref()
    t_start = time.perf_counter()
    alpha_ns = np.array([0.01, 0.03, 0.1, 0.3, 1, 3])
    colors = ["blue", "orange", "red", "green", "purple", "grey"]
    v_walls = np.linspace(0.2, 0.95, 10)
    lines = ["--", "-", "--", "-"]
    V_s = 1
    a_s = 1.1
    a_b = 1
    allow_invalid = True
    logger.debug("Loading models for plot_xi_kappa.")
    models = [
        ConstCSModel(a_s=a_s, a_b=a_b, css2=1/3, csb2=1/3, V_s=V_s, allow_invalid=allow_invalid),
        ConstCSModel(a_s=a_s, a_b=a_b, css2=1/3, csb2=1/4, V_s=V_s, allow_invalid=allow_invalid),
        ConstCSModel(a_s=a_s, a_b=a_b, css2=1/4, csb2=1/3, V_s=V_s, allow_invalid=allow_invalid),
        ConstCSModel(a_s=a_s, a_b=a_b, css2=1/4, csb2=1/4, V_s=V_s, allow_invalid=allow_invalid)
    ]
    logger.debug("Models loaded for plot_xi_kappa.")
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    params = np.zeros((len(models), alpha_ns.size, 2), dtype=np.object_)
    for i_model, model in enumerate(models):
        for i_alpha, alpha_n in enumerate(alpha_ns):
            params[i_model, i_alpha, :] = (model, alpha_n)
    kappas = run_parallel(
        func=kappa_vec, params=params, multiple_params=True, args=(v_walls, ),
        return_arr_shape=(v_walls.size, ), output_dtypes=(np.float64, )
    )
    for i_alpha, color in enumerate(colors):
        for i_model, (model, ls) in enumerate(zip(models, lines)):
            ax.plot(
                v_walls, kappas[i_model, i_alpha, :], color=color, ls=ls,
                label=f"{alpha_ns[i_alpha]}, css2={model.css2:.3f}, csb={model.csb2:.3f}"
            )

    ax.set_ylim(0.01, 10)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\xi_w$")
    ax.set_ylabel(r"$\kappa$")
    ax.legend()

    logger.info(f"Elapsed time: {time.perf_counter() - t_start:.2f}")
    return fig


if __name__ == "__main__":
    setup_logging()
    fig = main()
    utils.save_and_show(fig, "plot_xi_kappa.png")
