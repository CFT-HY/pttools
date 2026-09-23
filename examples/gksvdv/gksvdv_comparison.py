r"""
Comparison of Giese et al. and PTtools solvers
==============================================
"""

import logging
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from examples.utils import save_and_show_figs
from pttools.analysis.parallel import create_bubbles
from pttools.bubble.bubble import get_kappa
from pttools.models import GKSVDV_ALPHA_N, gksvdv_models, gksvdv_v_wall
from pttools.type_hints import FloatArr1D
from pttools.utils import testing_or_ci

logger: logging.Logger = logging.getLogger(__name__)


def main(
        colors: tp.Sequence[str] = ("b", "y", "r", "g", "purple", "grey"),
        alpha_ns: FloatArr1D = GKSVDV_ALPHA_N) -> tuple[plt.Figure, plt.Figure]:
    """Comparison of Giese et al. and PTtools solvers"""
    models = gksvdv_models()
    v_walls = gksvdv_v_wall(n=10 if testing_or_ci() else 50)
    logger.info("Minimum alpha_ns: %s", [model.alpha_n_min for model in models])
    for model in models:
        logger.info("Model parameters: %s", model.params_str())

    figsize_x = 10
    fig1: plt.Figure = plt.figure(figsize=(figsize_x, 6))
    fig2: plt.Figure = plt.figure(figsize=(figsize_x, 6))
    axs1 = fig1.subplots(1, 2)
    ax1 = axs1[0]
    ax2 = axs1[1]
    ax3 = fig2.add_subplot()

    kappas_pttools = np.empty((len(models), alpha_ns.size, v_walls.size))
    kappas_giese = np.empty((len(models), alpha_ns.size, v_walls.size))
    for i_model, model in enumerate(models):
        ls = "--" if i_model in [2, 3] else "-"
        _bubbles_pttools, kappas_pttools[i_model, :, :] = create_bubbles(
            model=model, v_walls=v_walls, alpha_ns=alpha_ns, func=get_kappa,
            bubble_kwargs={"allow_invalid": False}, allow_bubble_failure=True
        )
        _bubbles_giese, kappas_giese[i_model, :, :] = create_bubbles(
            model=model, v_walls=v_walls, alpha_ns=alpha_ns, func=get_kappa,
            bubble_kwargs={"allow_invalid": False, "use_giese_solver": True}, allow_bubble_failure=True
        )
        for i_alpha_n, (_alpha_n, color) in enumerate(zip(alpha_ns, colors, strict=False)):
            kpt = kappas_pttools[i_model, i_alpha_n, :]
            ax1.plot(v_walls, kpt, ls=ls, color=color, alpha=0.5)
            kg = kappas_giese[i_model, i_alpha_n, :]
            ax2.plot(v_walls, kg, ls=ls, color=color, alpha=0.5)
            ax3.plot(v_walls, (kpt - kg)/kg, color=color, alpha=0.5)

    for ax in axs1.flat:
        ax.set_xlabel(r"$v_\text{wall}$")
        ax.set_ylabel(r"$\kappa$")
        ax.set_xlim(v_walls.min(), v_walls.max())
        ax.set_yscale("log")
    ax1.set_title("PTtools")
    ax2.set_title("Giese et al.")
    ax3.set_xlim(v_walls.min(), v_walls.max())
    ax3.set_yscale("log")

    return fig1, fig2


if __name__ == "__main__":
    _fig1, _fig2 = main()
    save_and_show_figs({
        "giese_comparison": _fig1,
        "giese_comparison_diff": _fig2
    })
