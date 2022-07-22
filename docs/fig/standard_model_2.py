import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble.boundary import Phase
from pttools.models.sm import StandardModel


def main():
    thermo = StandardModel()
    temp = np.logspace(thermo.GEFF_DATA_LOG_TEMP[0], thermo.GEFF_DATA_LOG_TEMP[-1])

    fig: plt.Figure = plt.figure()
    ax1: plt.Axes
    ax2: plt.Axes
    ((ax1, ax2), (ax3, ax4)) = fig.subplots(2, 2)

    alpha = 0.5
    ax1.plot(temp, thermo.cs2(temp, Phase.SYMMETRIC), label="symmetric", alpha=alpha)
    ax1.plot(temp, thermo.cs2(temp, Phase.BROKEN), label="broken", alpha=alpha)
    ax1.plot(temp, thermo.cs2_full(temp, Phase.SYMMETRIC), label="symmetric (full)", alpha=alpha)
    ax1.plot(temp, thermo.cs2_full(temp, Phase.SYMMETRIC), label="broken (full)", alpha=alpha)
    ax1.set_xlabel("T(GeV)")
    ax1.set_ylabel("$c_s^2$")
    ax1.set_xscale("log")
    ax1.legend()

    ax2.plot(temp, thermo.ge(temp, Phase.SYMMETRIC), label=r"$g_e(\phi=s)$", alpha=alpha)
    ax2.plot(temp, thermo.ge(temp, Phase.BROKEN), label=r"$g_e(\phi=b)$", alpha=alpha)
    ax2.plot(temp, thermo.gs(temp, Phase.SYMMETRIC), label=r"$g_s(\phi=s)$", alpha=alpha)
    ax2.plot(temp, thermo.gs(temp, Phase.BROKEN), label=r"$g_s(\phi=b)$", alpha=alpha)
    ax2.set_xlabel("T(GeV)")
    ax2.set_ylabel("g")
    ax2.set_xscale("log")
    ax2.legend()

    ax3.plot(temp, thermo.dge_dT(temp, Phase.SYMMETRIC), label=r"$\frac{dg_e}{dT}(\phi=s)$", alpha=alpha)
    ax3.plot(temp, thermo.dge_dT(temp, Phase.BROKEN), label=r"$\frac{dg_e}{dT}(\phi=b)$", alpha=alpha)
    ax3.plot(temp, thermo.dgs_dT(temp, Phase.SYMMETRIC), label=r"$\frac{dg_s}{dT}(\phi=s)$", alpha=alpha)
    ax3.plot(temp, thermo.dgs_dT(temp, Phase.BROKEN), label=r"$\frac{dg_s}{dT}(\phi=b)$", alpha=alpha)
    ax3.set_xlabel("T(GeV)")
    ax3.set_ylabel(r"$\frac{dg}{dT}$")
    ax3.set_xscale("log")
    ax3.legend()

    ax4.plot(temp, thermo.de_dt(temp, Phase.SYMMETRIC), label=r"$\frac{de}{dT}(\phi=s)$", alpha=alpha)
    ax4.plot(temp, thermo.de_dt(temp, Phase.BROKEN), label=r"$\frac{de}{dT}(\phi=b)$", alpha=alpha)
    ax4.plot(temp, thermo.dp_dt(temp, Phase.SYMMETRIC), label=r"$\frac{dp}{dT}(\phi=s)$", alpha=alpha)
    ax4.plot(temp, thermo.dp_dt(temp, Phase.BROKEN), label=r"$\frac{dp}{dT}(\phi=b)$", alpha=alpha)
    ax4.set_xlabel("T(GeV)")
    ax4.set_ylabel(r"$\frac{dx}{dT}$")
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.legend()

    fig.tight_layout()


if __name__ == "__main__":
    main()
    plt.show()
