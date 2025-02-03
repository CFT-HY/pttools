"""
Giese testing 3
===============

Testing to find the properties of a single bubble with PTtools and Giese solvers
"""

import typing as tp

import matplotlib.pyplot as plt

from pttools.bubble import Bubble
from pttools.models import ConstCSModel
from pttools.bubble.props import find_phase
from pttools.bubble.thermo import kappa, kinetic_energy_density, va_trace_anomaly_diff

try:
    from giese.lisa import kappaNuMuModel
except ImportError:
    kappaNuMuModel: tp.Optional[callable] = None


def main():
    css2 = 1/4
    csb2 = 1/4
    # This is a problematic point
    v_wall = 0.56734694
    # v_wall = 0.85
    # v_wall = 0.86
    alpha_n = 0.01
    model = ConstCSModel(a_s=5, a_b=1, css2=css2, csb2=csb2, V_s=1, alpha_n_min=0.01)
    alpha_tbn = model.alpha_theta_bar_n_from_alpha_n(alpha_n)

    bubble = Bubble(model, v_wall=v_wall, alpha_n=alpha_n)
    bubble_fig = bubble.plot()

    # If the Giese code has not been loaded
    if kappaNuMuModel is None:
        return

    kappa_tbn_giese, v, wow, xi, mode, vp, vm = kappaNuMuModel(cs2b=csb2, cs2s=css2, al=alpha_tbn, vw=v_wall)

    phase_pttools = find_phase(bubble.xi, bubble.v_wall)
    phase_giese = find_phase(xi, v_wall)

    w = wow * bubble.wn
    # if bubble.sol_type == SolutionType.SUB_DEF:
    #     w = wow * bubble.wn
    #     # w = np.concatenate([bubble.w[0], w])
    #     w[xi <= v_wall] = bubble.w[0]
    #     phase_giese[1] = 1
    # elif bubble.sol_type == SolutionType.DETON:
    #     w = wow * bubble.w.max()
    #     w[xi >= v_wall] = bubble.w[-1]
    # elif bubble.sol_type == SolutionType.HYBRID:
    #     w = wow * bubble.wn
    #     # w[xi < v_wall] *= bubble.w[0] / wow[0]
    # else:
    #     raise RuntimeError

    det_pttools = va_trace_anomaly_diff(model, bubble.w, bubble.xi, v_wall, phase=phase_pttools)
    det_giese = va_trace_anomaly_diff(model, w, xi, v_wall, phase=phase_giese)
    print("va_trace_anomaly_diff", det_pttools, det_giese)

    kappa_pttools = kappa(model=model, v=bubble.v, w=bubble.w, xi=bubble.xi, v_wall=v_wall, delta_e_theta=det_pttools)
    kappa_giese = kappa(model=model, v=v, w=w, xi=xi, v_wall=v_wall, delta_e_theta=det_giese)
    print("kappa", kappa_pttools, kappa_giese)

    ek_pttools = kinetic_energy_density(bubble.v, bubble.w, bubble.xi, v_wall)
    ek_giese = kinetic_energy_density(v, w, xi, v_wall)
    print("e_K", ek_pttools, ek_giese)

    kappa_tbn_pttools = bubble.kappa_giese
    print("kappa_tbn", kappa_tbn_pttools, kappa_tbn_giese)

    print("csb", model.csb)
    print("v_mu2", bubble.v_mu)

    # print(phase_pttools)
    # print(phase_giese)
    bubble_fig.axes[0].plot(phase_pttools)
    bubble_fig.axes[0].plot(phase_giese, ls="--")

    # bubble_fig.axes[0].plot(xi, v, ls="--")
    # bubble_fig.axes[1].plot(xi, w, ls="--")
    bubble_fig.axes[0].scatter(xi, v, c="r")
    bubble_fig.axes[1].scatter(xi, w, c="r")
    for ax in bubble_fig.axes:
        ax.set_xlim(0, 1)

    # save(bubble_fig, "const_cs_bubble.png")


if __name__ == "__main__":
    main()
    plt.show()
