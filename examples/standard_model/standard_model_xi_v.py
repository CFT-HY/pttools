r"""
Standard Model xi, v
====================

Example that the Standard Model can be used with the bubble solver
"""

import matplotlib.pyplot as plt

from examples.utils import save_and_show_fig
from pttools.bubble.phase import Phase
from pttools.bubble.bubble import Bubble
from pttools.logging import setup_logging
from pttools.models.full import FullModel
from pttools.models.sm import StandardModel


def main() -> plt.Figure:
    sm = StandardModel(V_s=5e12, g_mult_s=1 + 1e-9, silence_temp=True)
    model = FullModel(sm, T_crit_guess=100e3)
    wn = model.wn(alpha_n=0.1)
    tn = model.temp(wn, Phase.SYMMETRIC)
    print(wn, tn)
    bubble = Bubble(model, v_wall=0.3, alpha_n=0.05)


    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()

    ax.plot(bubble.xi, bubble.v)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("$v$")
    return fig


if __name__ == "__main__":
    setup_logging()
    _fig = main()
    save_and_show_fig(_fig, "standard_model_xi_v")
