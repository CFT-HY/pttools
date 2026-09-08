"""
LISA noise
==========

Plot LISA instrument and astrophysical noise as a function of frequency
"""

from matplotlib import pyplot as plt
import numpy as np

from examples.utils import save_and_show_fig
from pttools.analysis.plot_spectra import (
    F_LABEL,
    NOISE_EB_LABEL,
    NOISE_GB_LABEL,
    NOISE_INS_LABEL,
    NOISE_LABEL,
    POW_GW0_H2_LABEL,
)
from pttools.analysis.utils import A4_PAPER_SIZE, legend
from pttools.omgw0 import noise


def main() -> plt.Figure:
    """Plot LISA instrument and astrophysical noise as a function of frequency"""
    fig: plt.Figure = plt.figure(figsize=A4_PAPER_SIZE)
    axs = fig.subplots(2, 2)

    f = np.logspace(-5, -1, 50)
    ax1 = axs[0, 0]
    P_oms = noise.P_oms()
    ax1.plot([f.min(), f.max()], [P_oms, P_oms], label=r"$P_\text{oms}$")
    ax1.plot(f, noise.P_acc(f), label=r"$P_\text{acc}$")
    ax1.set_ylabel("$P(f)$")

    ax2 = axs[0, 1]
    ax2.plot(f, noise.S_AE(f), label="$S_{A,E}$")
    ax2.plot(f, noise.S_AE_approx(f), label=r"$S_{A,E,\text{approx}}$")
    ax2.plot(f, noise.S_gb(f), label=r"$S_\text{gb}$")
    ax2.set_ylabel("$S(f)$")
    ax2.set_ylim(1e-40, 1e-34)

    ax3 = axs[1, 0]
    ax3.plot(f, noise.omega_ins_h2(f), label=NOISE_INS_LABEL)
    ax3.plot(f, noise.omega_eb_h2(f), label=NOISE_EB_LABEL)
    ax3.plot(f, noise.omega_gb_h2(f), label=NOISE_GB_LABEL)
    ax3.plot(f, noise.omega_noise_h2(f), label=NOISE_LABEL)
    ax3.set_ylabel(POW_GW0_H2_LABEL)
    ax3.set_ylim(1e-14, 1e-7)

    for ax in axs.flat:
        ax.set_xlabel(F_LABEL)
        ax.set_xlim(f[0], f[-1])
        ax.set_xscale("log")
        ax.set_yscale("log")
        legend(ax, loc="upper right")
    fig.tight_layout()

    return fig


if __name__ == '__main__':
    _fig = main()
    save_and_show_fig(_fig, "noise")
