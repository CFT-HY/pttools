r"""
Giese LISA fig. 2
=================

Reproduction of :giese_2021:`\ `, fig. 2
"""

import matplotlib.pyplot as plt
import numpy as np

from examples.utils import save_and_show
from pttools.analysis.parallel import create_bubbles
from pttools.bubble.bubble import Bubble
from pttools.models.const_cs import ConstCSModel


def get_kappa(bubble: Bubble) -> float:
    if not bubble.solved:
        return np.nan
    return bubble.kappa


get_kappa.return_type = float
get_kappa.fail_value = np.nan


def main():
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()

    a_s = 5
    a_b = 1
    V_s = 1
    models = [
        ConstCSModel(css2=1/3, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s),
        ConstCSModel(css2=1/3, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s),
        ConstCSModel(css2=1/4, csb2=1/3, a_s=a_s, a_b=a_b, V_s=V_s),
        ConstCSModel(css2=1/4, csb2=1/4, a_s=a_s, a_b=a_b, V_s=V_s)
    ]
    alpha_thetabar_ns = np.array([0.01, 0.03, 0.1, 0.3, 1, 3])
    colors = ["b", "y", "r", "g", "purple", "grey"]
    xi_ws = np.linspace(0.2, 0.95, 50)
    for i, model in enumerate(models):
        ls = "--" if i in [2, 3] else "-"
        bubbles, kappas = create_bubbles(
            model=model, v_walls=xi_ws, alpha_ns=alpha_thetabar_ns, func=get_kappa,
            bubble_kwargs={"theta_bar": True, "allow_invalid": True}, allow_bubble_failure=True
        )
        for i, color in enumerate(colors):
            ax.plot(xi_ws, kappas[i], ls=ls, color=color, alpha=0.5)

    ax.set_xlabel(r"$\xi_w$")
    ax.set_ylabel(r"$\kappa$")
    ax.set_yscale("log")
    ax.set_ylim(top=1)
    return fig


if __name__ == "__main__":
    fig = main()
    save_and_show(fig, "giese_lisa_fig2.png")
