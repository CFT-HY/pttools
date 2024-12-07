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

from examples.utils import save
from pttools.analysis.parallel import create_bubbles
# from pttools.analysis.utils import A4_PAPER_SIZE
from pttools.bubble import Bubble, Phase
from pttools.models import ConstCSModel
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
        kappa, v_arr, wow_arr, xi_arr, mode, vp, vm = kappaNuMuModel(
            # cs2s=model.cs2(model.w_crit, Phase.SYMMETRIC),
            # cs2b=model.cs2(model.w_crit, Phase.BROKEN),
            cs2s=model.css2,
            cs2b=model.csb2,
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
                alpha_tbns[i] = model.alpha_theta_bar_n_from_alpha_n(alpha_n=alpha_n, wn=wn)
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
        axs: tp.Iterable[plt.Axes],
        models: tp.List[ConstCSModel],
        alpha_ns: np.ndarray,
        colors: tp.List[str],
        lss: tp.List[str],
        v_walls: np.ndarray,
        theta_bar: bool = False,
        giese: bool = False):
    kappas = np.empty((len(models), alpha_ns.size, v_walls.size))
    for i_model, (model, ls) in enumerate(zip(models, lss)):
        if giese:
            if kappaNuMuModel is None:
                kappas[i_model, :, :] = np.nan
            kappas[i_model, :, :] = kappas_giese(model=model, v_walls=v_walls, alpha_ns=alpha_ns, theta_bar=theta_bar)
        else:
            bubbles, kappas[i_model, :, :] = create_bubbles(
                model=model, v_walls=v_walls, alpha_ns=alpha_ns, func=get_kappa,
                bubble_kwargs={"theta_bar": theta_bar, "allow_invalid": False}, allow_bubble_failure=True
            )
        for i_alpha_n, (alpha_n, color) in enumerate(zip(alpha_ns, colors)):
            try:
                i_max = np.nanargmax(kappas[i_model, i_alpha_n, :])
            except ValueError:
                logger.error(f"Could not produce bubbles with alpha_n={alpha_n} for {model.label_unicode}")
                continue
            kwargs = {}
            # if ls == "-":
            #     kwargs["label"] = rf"$\alpha={alpha_ns[i_alpha_n]}$"
            for ax in axs:
                ax.plot(v_walls, kappas[i_model, i_alpha_n, :], ls=ls, color=color, alpha=0.5, **kwargs)
            logger.info(
                f"alpha_n={alpha_n}, kappa_max={kappas[i_model, i_alpha_n, i_max]}, i_max={i_max}, "
                f"v_wall={v_walls[i_max]}, color={color}, ls={ls}, {model.label_unicode}"
            )
            failed_inds = np.argwhere(np.isnan(kappas[i_model, i_alpha_n, :]))
            if failed_inds.size:
                logger.info(
                    "Failed v_walls: %s",
                    v_walls[failed_inds].flatten())
    title = ""
    if giese:
        title += "Giese et al."
    else:
        title += "PTtools"
    title += ", "
    if theta_bar:
        title += r"$\alpha_{\bar{\theta}_n}$"
    else:
        title += r"$\alpha_n$"

    for ax in axs:
        ax.set_ylabel(r"$\kappa_{\bar{\theta}_n}$")
        ax.set_ylim(bottom=10 ** -2.5, top=1)
        ax.set_xlabel(r"$v_\text{wall}$")
        ax.set_yscale("log")
        ax.set_xlim(v_walls.min(), v_walls.max())
        ax.set_title(title)

    return kappas


def create_diff_figure(
        ax: plt.Axes,
        kappas_pttools: np.ndarray,
        kappas_giese: np.ndarray,
        models: tp.List[ConstCSModel],
        v_walls: np.ndarray,
        colors: tp.List[str],
        lss: tp.List[str],
        theta_bar: bool,
        title: bool = True):
    rel_diffs = np.abs(kappas_pttools - kappas_giese) / kappas_giese
    if theta_bar:
        title_str = r"$\alpha_{\bar{\theta}_n}$"
    else:
        title_str = r"$\alpha_n$"
    print(title_str)
    for i_model, (model, ls) in enumerate(zip(models, lss)):
        for i_alpha in range(kappas_pttools.shape[1]):
            ax.plot(
                v_walls,
                rel_diffs[i_model, i_alpha, :],
                color=colors[i_alpha],
                ls=ls,
                # label=f"Model {i_model}, alpha {i_alpha}",
            )
        print(model.label_unicode)
        print(np.nanmax(rel_diffs[i_model, :, :], axis=1))
    ax.set_xlabel(r"$v_\text{wall}$")
    ax.set_ylabel(
        r"$|\kappa_{\bar{\theta}_n,\text{PTtools}} - \kappa_{\bar{\theta}_n,\text{ref}}|"
        r" / \kappa_{\bar{\theta}_n,\text{ref}}$")
    ax.set_yscale("log")
    ax.set_ylim(10**(-6), 1)
    ax.set_xlim(v_walls.min(), v_walls.max())
    if title:
        ax.set_title(title_str)


def main():
    alpha_thetabar_ns = np.array([0.01, 0.03, 0.1, 0.3, 1, 3])
    colors = ["b", "y", "r", "g", "purple", "grey"]
    n_v_walls = 20 if GITHUB_ACTIONS else 50
    v_walls = np.linspace(0.2, 0.95, n_v_walls)
    lss = ["-", "--", ":", "-."]
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

    figsize_x = 8
    fig1: plt.Figure = plt.figure(figsize=(figsize_x, 6))
    fig2: plt.Figure = plt.figure(figsize=(figsize_x, 4))
    fig3: plt.Figure = plt.figure(figsize=(figsize_x, 4))
    fig4: plt.Figure = plt.figure(figsize=(figsize_x, 4))
    axs1 = fig1.subplots(2, 2)
    axs2 = fig2.subplots(1, 2)
    axs3 = fig3.subplots(1, 2)
    ax4 = fig4.add_subplot()

    start_time = time.perf_counter()
    kappas_giese_atbn = create_figure(
        axs=(axs1[1, 0], ),
        models=models, alpha_ns=alpha_thetabar_ns, colors=colors, lss=lss, v_walls=v_walls,
        theta_bar=True, giese=True
    )
    kappas_giese_an = create_figure(
        axs=(axs1[1, 1], axs3[1]),
        models=models, alpha_ns=alpha_thetabar_ns, colors=colors, lss=lss, v_walls=v_walls,
        theta_bar=False, giese=True
    )
    giese_time = time.perf_counter()
    logger.info(f"Creating Giese kappa figures took {giese_time - start_time:.2f} s.")
    kappas_pttools_atbn = create_figure(
        axs=(axs1[0, 0], ),
        models=models, alpha_ns=alpha_thetabar_ns, colors=colors, lss=lss, v_walls=v_walls,
        theta_bar=True
    )
    kappas_pttools_an = create_figure(
        axs=(axs1[0, 1], axs3[0]),
        models=models, alpha_ns=alpha_thetabar_ns, colors=colors, lss=lss, v_walls=v_walls,
        theta_bar=False
    )
    print("v_walls")
    print(v_walls)
    create_diff_figure(
        ax=axs2[0],
        kappas_pttools=kappas_pttools_atbn, kappas_giese=kappas_giese_atbn,
        models=models, colors=colors, lss=lss, v_walls=v_walls, theta_bar=True
    )
    create_diff_figure(
        ax=axs2[1],
        kappas_pttools=kappas_pttools_an, kappas_giese=kappas_giese_an,
        models=models, colors=colors, lss=lss, v_walls=v_walls, theta_bar=False
    )
    create_diff_figure(
        ax=ax4,
        kappas_pttools=kappas_pttools_an, kappas_giese=kappas_giese_an,
        models=models, colors=colors, lss=lss, v_walls=v_walls, theta_bar=False, title=False
    )
    logger.info(f"Creating PTtools kappa figures took {time.perf_counter() - giese_time:.2f} s.")
    fig1.tight_layout()
    fig2.tight_layout()
    fig3.tight_layout()
    fig4.tight_layout()
    return fig1, fig2, fig3, fig4


if __name__ == "__main__":
    fig, fig2, fig3, fig4 = main()
    save(fig, "giese_lisa_fig2")
    save(fig2, "giese_lisa_fig2_diff")
    save(fig3, "giese_lisa_fig2_alpha_n")
    save(fig4, "giese_lisa_fig2_alpha_n_diff")
    plt.show()
