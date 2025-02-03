"""
Giese testing 2
===============

Compare Giese fluid profiles with PTtools
"""

import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from examples import utils
from pttools.analysis.utils import A4_PAPER_SIZE
from pttools.bubble import Bubble, Phase, lorentz, v_chapman_jouguet_const_cs
from pttools.models import Model, BagModel, ConstCSModel

try:
    from giese.lisa import kappaNuMuModel
except ImportError:
    kappaNuMuModel: tp.Optional[callable] = None


def main():
    alpha_n = 0.3
    theta_bar = False
    # v_walls = np.array([0.5, 0.6, 0.65])
    v_walls = np.array([0.8122449, 0.82755102, 0.84285714, 0.85816327])
    colors = ["r", "g", "b", "orange"]
    a_s = 5
    a_b = 1
    V_s = 1
    models = [
        # BagModel(a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_theta_bar_n)
        ConstCSModel(css2=1/3, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n),
        ConstCSModel(css2=1/3, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n),
        # ConstCSModel(css2=1/4, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n),
        ConstCSModel(css2=1/4, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s, alpha_n_min=alpha_n)
    ]
    print("Models")
    for model in models:
        alpha_theta_bar_n = model.alpha_theta_bar_n_from_alpha_n(alpha_n)
        print("Model:", model.params_str(), model.tn(alpha_n=alpha_n, theta_bar=theta_bar)/model.critical_temp())
        print("v_cj:", v_chapman_jouguet_const_cs(model, alpha_theta_bar_plus=alpha_theta_bar_n))

    fig: plt.Figure = plt.figure(figsize=A4_PAPER_SIZE)
    axs = fig.subplots(2, 2)
    axs_flat = np.ravel(axs)

    print("Solving bubbles")
    for ax, model in zip(axs_flat, models):
        print("Model:", model.params_str())
        for v_wall, color in zip(v_walls, colors):
            bubble = Bubble(model=model, v_wall=v_wall, alpha_n=alpha_n, theta_bar=theta_bar)
            print("Bubble:", f"css2={model.css2}, csb2={model.csb2}, v_wall={v_wall}, sol_type={bubble.sol_type}")
            ax.plot(bubble.xi, bubble.v, label=f"$v_w={v_wall}$", c=color)

            if not theta_bar:
                alpha_theta_bar_n = model.alpha_theta_bar_n_from_alpha_n(alpha_n=alpha_n)
                print("alpha_theta_bar_n_giese", alpha_theta_bar_n)
            else:
                alpha_theta_bar_n = alpha_n

            # If the Giese code has not been loaded
            if kappaNuMuModel is None:
                continue

            kappa, v, w, xi, mode, vp, vm = kappaNuMuModel(
                cs2s=model.cs2(model.w_crit, Phase.SYMMETRIC),
                cs2b=model.cs2(model.w_crit, Phase.BROKEN),
                al=alpha_theta_bar_n,
                vw=v_wall
            )
            ax.plot(xi, v, ls=":", c=color)
            if np.isclose(v_wall, 0.8):
                print(np.nanmax(v))

        # The dotted line can be computed directly
        xi_mu = np.linspace(np.sqrt(model.csb2), 1, 20)
        v_mu = lorentz(xi=xi_mu, v=np.sqrt(model.csb2))
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
