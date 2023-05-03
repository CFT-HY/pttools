"""
Bag model reference properties
==============================

Plot the parameters of fluid_reference
"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble import fluid_reference
from pttools.bubble.transition import SolutionType


def main():
    ref = fluid_reference.ref()

    fig: plt.Figure = plt.figure()
    axs: np.ndarray = fig.subplots(nrows=2, ncols=3)

    ax_vp: plt.Axes = axs[0, 0]
    ax_vm: plt.Axes = axs[1, 0]
    ax_vp_tilde: plt.Axes = axs[0, 1]
    ax_vm_tilde: plt.Axes = axs[1, 1]
    ax_wp: plt.Axes = axs[0, 2]
    ax_wm: plt.Axes = axs[1, 2]

    ax_vp.contourf(ref.v_wall, ref.alpha_n, ref.vp)
    ax_vm.contourf(ref.v_wall, ref.alpha_n, ref.vm)
    ax_vp_tilde.contourf(ref.v_wall, ref.alpha_n, ref.vp_tilde)
    ax_vm_tilde.contourf(ref.v_wall, ref.alpha_n, ref.vm_tilde)
    ax_wp.contourf(ref.v_wall, ref.alpha_n, ref.wp)
    ax_wm.contourf(ref.v_wall, ref.alpha_n, ref.wm)

    ax_vp.set_title("$v_+$")
    ax_vm.set_title("$v_-$")
    ax_vp_tilde.set_title(r"$\tilde{v}_+$")
    ax_vm_tilde.set_title(r"$\tilde{v}_-$")
    ax_wp.set_title("$w_+$")
    ax_wm.set_title("$w_-$")

    for ax in axs.flat:
        ax.set_xlabel("$v_w$")
        ax.set_ylabel(r"$\alpha_n$")

    fig.tight_layout()

    print(ref.get(v_wall=0.25, alpha_n=0.8, sol_type=SolutionType.SUB_DEF))
    print(ref.get(v_wall=0.7, alpha_n=0.25, sol_type=SolutionType.HYBRID))
    print(ref.get(v_wall=0.1, alpha_n=0.95, sol_type=SolutionType.DETON))


if __name__ == "__main__":
    main()
    plt.show()
