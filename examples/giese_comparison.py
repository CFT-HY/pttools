r"""
Comparison of Giese and PTtools solvers
=======================================
"""


import logging


import matplotlib.pyplot as plt
import numpy as np

from examples.utils import save
from pttools.analysis.parallel import create_bubbles
from pttools.bubble import Bubble
from pttools.models import ConstCSModel
from pttools.speedup import GITHUB_ACTIONS

logger = logging.getLogger(__name__)


def get_kappa(bubble: Bubble) -> float:
    if (not bubble.solved) or bubble.no_solution_found or bubble.solver_failed or bubble.numerical_error:
        return np.nan
    return bubble.kappa

get_kappa.return_type = float
get_kappa.fail_value = np.nan


def main():
    alpha_ns = np.array([0.01, 0.03, 0.1, 0.3, 1, 3])
    colors = ["b", "y", "r", "g", "purple", "grey"]
    n_v_walls = 20 if GITHUB_ACTIONS else 50
    v_walls = np.linspace(0.2, 0.95, n_v_walls)
    a_s = 5
    a_b = 1
    V_s = 1
    models = [
        ConstCSModel(css2=1 / 3, csb2=1 / 3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_ns[0]),
        ConstCSModel(css2=1 / 3, csb2=1 / 4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_ns[0]),
        ConstCSModel(css2=1 / 4, csb2=1 / 3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_ns[0]),
        ConstCSModel(css2=1 / 4, csb2=1 / 4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_ns[0])
    ]
    logger.info(f"Minimum alpha_ns: %s", [model.alpha_n_min for model in models])
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
        bubbles_pttools, kappas_pttools[i_model, :, :] = create_bubbles(
            model=model, v_walls=v_walls, alpha_ns=alpha_ns, func=get_kappa,
            bubble_kwargs={"allow_invalid": False}, allow_bubble_failure=True
        )
        bubbles_giese, kappas_giese[i_model, :, :] = create_bubbles(
            model=model, v_walls=v_walls, alpha_ns=alpha_ns, func=get_kappa,
            bubble_kwargs={"allow_invalid": False, "use_giese_solver": True}, allow_bubble_failure=True
        )
        for i_alpha_n, (alpha_n, color) in enumerate(zip(alpha_ns, colors)):
            kpt = kappas_pttools[i_model, i_alpha_n, :]
            kg = kappas_giese[i_model, i_alpha_n, :]
            ax1.plot(v_walls, kpt, ls=ls, color=color, alpha=0.5)
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
    fig, fig2 = main()
    save(fig, "giese_comparison")
    save(fig, "giese_comparison_diff")
    plt.show()
