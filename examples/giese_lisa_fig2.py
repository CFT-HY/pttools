r"""
Giese LISA fig. 2
=================

Reproduction of :giese_2021:`\ `, fig. 2
"""

import logging
import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from examples.utils import save_and_show
from pttools.analysis.parallel import create_bubbles
from pttools.analysis.utils import A4_PAPER_SIZE
from pttools.bubble import Bubble, Phase
from pttools.models import Model, ConstCSModel

try:
    from giese.lisa import kappaNuMuModel
except ImportError:
    kappaNuMuModel: tp.Optional[callable] = None

logger = logging.getLogger(__name__)


def get_kappa(bubble: Bubble) -> float:
    if not bubble.solved:
        return np.nan
    return bubble.kappa


get_kappa.return_type = float
get_kappa.fail_value = np.nan


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
            kappas = np.empty((alpha_ns.size, v_walls.size))
            for j, alpha_n in enumerate(alpha_ns):
                if theta_bar:
                    alpha_tbn_giese = alpha_n
                else:
                    try:
                        wn_guess = model.w_n(alpha_n, error_on_invalid=False)
                        alpha_tbn_giese = model.alpha_n_from_alpha_theta_bar_n(alpha_theta_bar_n=alpha_n, wn_guess=wn_guess)
                        logger.info("Creating Giese plot for alpha_theta_bar_n: %s", alpha_tbn_giese)
                    except RuntimeError:
                        kappas[j, :] = np.nan
                        continue
                if kappaNuMuModel is None:
                    kappas[j, :] = np.nan
                else:
                    for k, v_wall in enumerate(v_walls):
                        try:
                            kappas[j, k], _, _, _, _ = kappaNuMuModel(
                                cs2s=model.cs2(model.w_crit, Phase.SYMMETRIC),
                                cs2b=model.cs2(model.w_crit, Phase.BROKEN),
                                al=alpha_tbn_giese,
                                vw=v_wall
                            )
                        except ValueError:
                            kappas[j, k] = np.nan
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
            ax.plot(v_walls, kappas[j], ls=ls, color=color, alpha=0.5)
            print(
                f"alpha_n={alpha_ns[j]}, kappa_max={kappas[j, i_max]}, i_max={i_max}, "
                f"v_wall={v_walls[i_max]}, color={color}, ls={ls}, {model.label_unicode}"
            )

    ax.set_xlabel(r"$\xi_w$")
    ax.set_ylabel(r"$\kappa$")
    ax.set_yscale("log")
    ax.set_ylim(bottom=10**-2.5, top=1)

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


def main():
    alpha_thetabar_ns = np.array([0.01, 0.03, 0.1, 0.3, 1, 3])
    colors = ["b", "y", "r", "g", "purple", "grey"]
    v_walls = np.linspace(0.2, 0.95, 50)
    a_s = 5
    a_b = 1
    V_s = 1
    models = [
        ConstCSModel(css2=1/3, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_thetabar_ns[0]),
        ConstCSModel(css2=1/3, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_thetabar_ns[0]),
        ConstCSModel(css2=1/4, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_thetabar_ns[0]),
        ConstCSModel(css2=1/4, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_thetabar_ns[0])
    ]
    # print(f"Minimum alpha_ns: {[model.alpha_n_min for model in models]}")
    for model in models:
        print(
            f"css2={model.css2:.3f}, csb2={model.csb2:.3f}, alpha_n_min={model.alpha_n_min:.3f} "
            f"(a_s={model.a_s:.3f}, a_b={model.a_b:.3f}, V_s={model.V_s:.3f}, V_b={model.V_b:.3f})"
        )

    fig: plt.Figure = plt.figure(figsize=A4_PAPER_SIZE)
    axs = fig.subplots(2, 2)

    create_figure(axs[0, 0], models, alpha_ns=alpha_thetabar_ns, colors=colors, v_walls=v_walls, theta_bar=True)
    create_figure(axs[0, 1], models, alpha_ns=alpha_thetabar_ns, colors=colors, v_walls=v_walls, theta_bar=False)
    create_figure(axs[1, 0], models, alpha_ns=alpha_thetabar_ns, colors=colors, v_walls=v_walls, theta_bar=True, giese=True)
    create_figure(axs[1, 1], models, alpha_ns=alpha_thetabar_ns, colors=colors, v_walls=v_walls, theta_bar=False, giese=True)

    fig.tight_layout()
    return fig


if __name__ == "__main__":
    fig = main()
    save_and_show(fig, "giese_lisa_fig2.png")
