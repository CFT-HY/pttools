r"""
Giese LISA fig. 2
=================

Reproduction of :giese_2021:`\ `, fig. 2
"""

import logging
import time
import typing as tp

import matplotlib.pyplot as plt
import numpy as np
from numba.cuda.cudadrv.runtime import Runtime

from examples.utils import save_and_show
from pttools.analysis import A4_PAPER_SIZE
from pttools.analysis.parallel import create_bubbles
from pttools.analysis.utils import A4_PAPER_SIZE
from pttools.bubble import Bubble, Phase
from pttools.models import Model, ConstCSModel
from pttools.speedup import run_parallel, GITHUB_ACTIONS

logger = logging.getLogger(__name__)

try:
    logger.info("Giese code imported successfully.")
    from giese.lisa import kappaNuMuModel
except ImportError:
    logger.info("Giese could not be imported.")
    kappaNuMuModel: tp.Optional[callable] = None

def get_kappa(bubble: Bubble) -> float:
    if (not bubble.solved) or bubble.no_solution_found or bubble.solver_failed or bubble.numerical_error:
        return np.nan
    return bubble.kappa_giese


get_kappa.return_type = float
get_kappa.fail_value = np.nan


def kappa_giese(params: np.ndarray, model: ConstCSModel) -> float:
    v_wall, alpha_tbn_giese = params
    try:
        kappa, _, _, _, _ = kappaNuMuModel(
            cs2s=model.cs2(model.w_crit, Phase.SYMMETRIC),
            cs2b=model.cs2(model.w_crit, Phase.BROKEN),
            al=alpha_tbn_giese,
            vw=v_wall
        )
    except ValueError:
        return np.nan
    return kappa


def kappas_giese(
        model: ConstCSModel,
        v_walls: np.ndarray,
        alpha_ns: np.ndarray,
        theta_bar: bool = False) -> np.ndarray:
    if theta_bar:
        alpha_tbns = alpha_ns
    else:
        alpha_tbns = np.empty((alpha_ns.size,))
        for i, alpha_n in enumerate(alpha_ns):
            try:
                wn = model.w_n(alpha_n, theta_bar=theta_bar)
                alpha_tbns[i] = model.alpha_n_from_alpha_theta_bar_n(alpha_theta_bar_n=alpha_n, wn=wn)
            except (ValueError, RuntimeError):
                alpha_tbns[i] = np.nan

    params = np.empty((alpha_ns.size, v_walls.size, 2))
    for i_alpha_tbn, alpha_tbn in enumerate(alpha_tbns):
        for i_v_wall, v_wall in enumerate(v_walls):
            params[i_alpha_tbn, i_v_wall, 0] = v_wall
            params[i_alpha_tbn, i_v_wall, 1] = alpha_tbn

    kappas = run_parallel(
        func=kappa_giese,
        params=params,
        multiple_params=True,
        # output_dtypes=(float, ),
        # max_workers=max_workers,
        log_progress_percentage=True,
        kwargs={"model": model}
    )
    return kappas


def create_figure(
        ax: plt.Axes,
        models: tp.List[Model],
        alpha_ns: np.ndarray,
        colors: tp.List[str],
        v_walls: np.ndarray,
        theta_bar: bool = False,
        giese: bool = False):
    for i, model in enumerate(models):
        ls = "--" if i in [2, 3] else "-"
        if giese:
            if kappaNuMuModel is None:
                continue
            kappas = kappas_giese(model=model, v_walls=v_walls, alpha_ns=alpha_ns, theta_bar=theta_bar)
        else:
            bubbles, kappas = create_bubbles(
                model=model, v_walls=v_walls, alpha_ns=alpha_ns, func=get_kappa,
                bubble_kwargs={"theta_bar": theta_bar, "allow_invalid": False}, allow_bubble_failure=True
            )
        for j, color in enumerate(colors):
            try:
                i_max = np.nanargmax(kappas[j])
            except ValueError:
                print(f"Could not produce bubbles with alpha_n={alpha_ns[j]} for {model.label_unicode}")
                continue
            kwargs = {}
            if ls == "-":
                kwargs["label"] = rf"$\alpha={alpha_ns[j]}$"
            ax.plot(v_walls, kappas[j], ls=ls, color=color, alpha=0.5, **kwargs)
            print(
                f"alpha_n={alpha_ns[j]}, kappa_max={kappas[j, i_max]}, i_max={i_max}, "
                f"v_wall={v_walls[i_max]}, color={color}, ls={ls}, {model.label_unicode}"
            )

    ax.set_xlabel(r"$\xi_w$")
    ax.set_ylabel(r"$\kappa_{\bar{\theta}_n}$")
    ax.set_yscale("log")
    ax.set_ylim(bottom=10**-2.5, top=1)
    ax.set_xlim(v_walls.min(), v_walls.max())

    title = ""
    if giese:
        title += "Giese"
    else:
        title += "PTtools"
    title += ", "
    if theta_bar:
        title += r"$\alpha_{\bar{\theta}_n}$"
    else:
        title += r"$\alpha_n$"
    ax.set_title(title)
    # ax.legend()


def main():
    alpha_thetabar_ns = np.array([0.01, 0.03, 0.1, 0.3, 1, 3])
    colors = ["b", "y", "r", "g", "purple", "grey"]
    n_v_walls = 20 if GITHUB_ACTIONS else 50
    v_walls = np.linspace(0.2, 0.95, n_v_walls)
    a_s = 5
    a_b = 1
    V_s = 1
    models = [
        ConstCSModel(css2=1/3, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_thetabar_ns[0]),
        ConstCSModel(css2=1/3, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_thetabar_ns[0]),
        ConstCSModel(css2=1/4, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_thetabar_ns[0]),
        ConstCSModel(css2=1/4, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_thetabar_ns[0])
    ]
    logger.info(f"Minimum alpha_ns: %s", [model.alpha_n_min for model in models])
    for model in models:
        logger.info("Model parameters: %s", model.params_str())

    fig: plt.Figure = plt.figure(figsize=(10, 8))
    axs = fig.subplots(2, 2)

    start_time = time.perf_counter()
    create_figure(
        axs[1, 0], models, alpha_ns=alpha_thetabar_ns, colors=colors, v_walls=v_walls, theta_bar=True, giese=True)
    create_figure(
        axs[1, 1], models, alpha_ns=alpha_thetabar_ns, colors=colors, v_walls=v_walls, theta_bar=False, giese=True)
    giese_time = time.perf_counter()
    logger.info(f"Creating Giese kappa figures took {giese_time - start_time:.2f} s.")
    create_figure(axs[0, 0], models, alpha_ns=alpha_thetabar_ns, colors=colors, v_walls=v_walls, theta_bar=True)
    create_figure(axs[0, 1], models, alpha_ns=alpha_thetabar_ns, colors=colors, v_walls=v_walls, theta_bar=False)
    logger.info(f"Creating PTtools kappa figures took {time.perf_counter() - giese_time:.2f} s.")
    fig.tight_layout()

    print("v_walls")
    print(v_walls)
    return fig


if __name__ == "__main__":
    fig = main()
    save_and_show(fig, "giese_lisa_fig2")
