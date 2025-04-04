"""
Old-new comparison
==================

Comparison of old and new solvers
"""

import matplotlib.pyplot as plt
import numpy as np

from examples import utils
from pttools.analysis.utils import A3_PAPER_SIZE
from pttools.bubble import boundary
from pttools.bubble.boundary import Phase, SolutionType
from pttools.bubble.bubble import Bubble
from pttools.bubble import fluid_bag
from pttools.bubble import relativity
from pttools.models.model import Model
from pttools.models.bag import BagModel
from pttools.ssmtools.spectrum import SSMSpectrum, power_gw_scaled_bag, spec_den_v_bag, power_v_bag
from tests.paper.plane import xiv_plane
from tests.paper.plot_plane_paper import plot_plane


def validate(model: Model, v: np.ndarray, w: np.ndarray, xi: np.ndarray, sol_type: SolutionType):
    if sol_type == SolutionType.SUB_DEF:
        validate_def(model, v, w, xi, sol_type)
    elif sol_type == SolutionType.HYBRID:
        validate_def(model, v, w, xi, sol_type)
        validate_shock(model, v, w, xi, sol_type)
    elif sol_type == SolutionType.DETON:
        validate_shock(model, v, w, xi, sol_type)


def validate_def(model: Model, v: np.ndarray, w: np.ndarray, xi: np.ndarray, sol_type: SolutionType):
    i_wall = np.argmax(v)
    v_wall = xi[i_wall]
    v1p = v[i_wall-1]
    v2p = v[i_wall]
    v1w = -relativity.lorentz(v1p, v_wall)
    v2w = -relativity.lorentz(v2p, v_wall)
    w1 = w[i_wall-1]
    w2 = w[i_wall]
    validate2(model, v1p, v2p, v1w, v2w, w1, w2, Phase.BROKEN, Phase.SYMMETRIC, sol_type)


def validate_shock(model: Model, v: np.ndarray, w: np.ndarray, xi: np.ndarray, sol_type: SolutionType):
    v_wall = xi[-2]
    v1p = v[-3]
    v2p = 0
    v1w = -relativity.lorentz(v1p, v_wall)
    v2w = -relativity.lorentz(v2p, v_wall)
    w1 = w[-3]
    w2 = w[-2]
    if sol_type == SolutionType.DETON:
        phase1 = Phase.BROKEN
        phase2 = Phase.SYMMETRIC
    else:
        phase1 = Phase.SYMMETRIC
        phase2 = Phase.SYMMETRIC
    validate2(model, v1p, v2p, v1w, v2w, w1, w2, phase1, phase2, sol_type)


def validate2(
        model: Model,
        v1p: float, v2p: float,
        v1w: float, v2w: float,
        w1: float, w2: float,
        phase1: Phase, phase2: Phase,
        sol_type: SolutionType):
    dev = boundary.junction_conditions_solvable(np.array([v2w, w2]), model, v1w, w1, phase1, phase2)
    print(f"sol_type={sol_type}, v1p={v1p}, v2p={v2p}, v1w={v1w}, v2w={v2w}, w1={w1}, w2={w2}, dev={dev}")


def main():
    bag = BagModel(a_s=1.1, a_b=1, V_s=2)

    v_walls = [0.5, 0.7, 0.77]
    alpha_ns = [0.578, 0.151, 0.091]
    sol_types = [SolutionType.SUB_DEF, SolutionType.HYBRID, SolutionType.DETON]

    spectra = [
        SSMSpectrum(
            Bubble(bag, v_wall=v_walls[i], alpha_n=alpha_ns[i], sol_type=sol_types[i])
        )
        for i in range(len(v_walls))
    ]
    z = spectra[0].y

    data = xiv_plane(separate_phases=False)
    fig: plt.Figure = plt.figure(figsize=A3_PAPER_SIZE)
    axs = fig.subplots(2, 2)
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]
    plot_plane(ax=ax1, data_s=data, selected_solutions=False)

    print("Solving & plotting old bubbles")
    for v_wall, alpha_n, sol_type in zip(v_walls, alpha_ns, sol_types):
        v, w, xi = fluid_bag.sound_shell_bag(v_wall=v_wall, alpha_n=alpha_n)
        ax1.plot(xi, v, color="blue", label=rf"$v_w={v_wall}, \alpha_n={alpha_n}$")
        validate(bag, v, w, xi, sol_type)

        label = rf"old, $v_w={v_wall}, \alpha_n={alpha_n}$"
        sdv = spec_den_v_bag(z, (v_wall, alpha_n))
        ax2.plot(z, sdv, label=label)

        pow_v = power_v_bag(z, (v_wall, alpha_n))
        ax3.plot(z, pow_v, label=label)

        gw = power_gw_scaled_bag(z, (v_wall, alpha_n))
        ax4.plot(z, gw, label=label)

    print("Plotting new bubbles")
    for spectrum in spectra:
        bubble = spectrum.bubble
        ax1.plot(bubble.xi, bubble.v, ls=":", color="red")
        validate(bag, bubble.v, bubble.w, bubble.xi, bubble.sol_type)

        label = rf"new, $v_w={bubble.v_wall}, \alpha_n={bubble.alpha_n}$"
        ax2.plot(spectrum.y, spectrum.spec_den_v, label=label)
        ax3.plot(spectrum.y, spectrum.pow_v, label=label)
        ax4.plot(spectrum.y, spectrum.pow_gw, label=label)

    ax2.set_ylabel("spec_den_v")
    ax3.set_ylabel("pow_v")
    ax4.set_ylabel(r"$\mathcal{P}_{\text{gw}}(z)$")

    for ax in (ax2, ax3, ax4):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$z = kR*$")

    for ax in axs.flat:
        ax.legend()

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    fig = main()
    utils.save_and_show(fig, "old_new.png")
