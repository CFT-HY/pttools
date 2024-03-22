"""
Giese testing 2
=============

Compare Giese fluid profiles with PTtools
"""

import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from examples import utils
from pttools.analysis.utils import A4_PAPER_SIZE
from pttools.bubble import Bubble, Phase, lorentz
from pttools.models import Model, BagModel, ConstCSModel

try:
    from giese.lisa import kappaNuMuModel
except ImportError:
    kappaNuMuModel: tp.Optional[callable] = None


def main():
    alpha_theta_bar_n = 0.1
    v_walls = np.array([0.4, 0.6, 0.8])
    colors = ["r", "g", "b"]
    a_s = 5
    a_b = 1
    V_s = 1
    models = [
        # BagModel(a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_theta_bar_n)
        ConstCSModel(css2=1/3, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_theta_bar_n),
        ConstCSModel(css2=1/3, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=0.3*alpha_theta_bar_n),
        ConstCSModel(css2=1/4, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_theta_bar_n),
        ConstCSModel(css2=1/4, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_theta_bar_n)
    ]

    fig: plt.Figure = plt.figure(figsize=A4_PAPER_SIZE)
    axs = fig.subplots(2, 2)
    axs_flat = np.ravel(axs)

    for ax, model in zip(axs_flat, models):
        for v_wall, color in zip(v_walls, colors):
            bubble = Bubble(model=model, v_wall=v_wall, alpha_n=alpha_theta_bar_n, theta_bar=True)
            ax.plot(bubble.xi, bubble.v, label=f"$v_w={v_wall}$", c=color)

            kappa, v, w, xi, mode = kappaNuMuModel(
                cs2s=model.cs2(model.w_crit, Phase.SYMMETRIC),
                cs2b=model.cs2(model.w_crit, Phase.BROKEN),
                al=alpha_theta_bar_n,
                vw=v_wall
            )
            ax.plot(xi, v, ls=":", c=color)

        xi_mu = np.linspace(np.sqrt(model.csb2), 1, 20)
        v_mu = lorentz(xi_mu, np.sqrt(model.csb2))
        ax.plot(xi_mu, v_mu, ls=":", c="k")

        ax.set_title(model.label_latex)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel("$v$")
        ax.legend()

    fig.tight_layout()
    utils.save(fig, "giese_testing2.png")
    return fig


if __name__ == "__main__":
    main()
    plt.show()
