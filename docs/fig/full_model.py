import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble.boundary import Phase
from pttools.models import FullModel, StandardModel


def main():
    thermo = StandardModel()
    model = FullModel(thermo=thermo, V_s=0)

    fig: plt.Figure = plt.figure()
    ax1: plt.Axes
    ax2: plt.Axes
    ax3: plt.Axes
    ax4: plt.Axes
    ((ax1, ax2), (ax3, ax4)) = fig.subplots(2, 2)

    alpha = 0.5

    temp = np.linspace(thermo.GEFF_DATA_TEMP[0], thermo.GEFF_DATA_TEMP[-1], 100)
    ax1.plot(temp, model.w(temp, Phase.SYMMETRIC), label="symmetric", alpha=alpha)
    ax1.plot(temp, model.w(temp, Phase.BROKEN), label="broken", alpha=alpha)
    ax1.set_xlabel("T")
    ax1.set_ylabel("w(T)")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.legend()

    w = np.logspace(1, 23, 100)
    ax2.plot(w, model.temp(w, Phase.SYMMETRIC), label="symmetric", alpha=alpha)
    ax2.plot(w, model.temp(w, Phase.BROKEN), label="broken", alpha=alpha)
    ax2.plot(model.w(thermo.GEFF_DATA_TEMP, Phase.SYMMETRIC), thermo.GEFF_DATA_TEMP, label="symmetric(full)")
    ax2.plot(model.w(thermo.GEFF_DATA_TEMP, Phase.BROKEN), thermo.GEFF_DATA_TEMP, label="broken(full)")
    ax2.set_xlabel("w")
    ax2.set_ylabel("T(w)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.legend()

    ax3.plot(w, model.cs2(w, Phase.SYMMETRIC), label="symmetric", alpha=alpha)
    ax3.plot(w, model.cs2(w, Phase.BROKEN), label="broken", alpha=alpha)
    ax3.plot(w, thermo.cs2_full(model.temp(w, Phase.SYMMETRIC), Phase.SYMMETRIC), label="symmetric (full)", alpha=alpha)
    ax3.plot(w, thermo.cs2_full(model.temp(w, Phase.BROKEN), Phase.BROKEN), label="broken (full)", alpha=alpha)
    ax3.set_xlabel("w")
    ax3.set_ylabel(r"$c_s^2$")
    # ax3.set_ylim(0, 1)
    ax3.set_xscale("log")
    ax3.legend()

    ax4.plot(w, model.theta(w, Phase.SYMMETRIC), label="symmetric", alpha=alpha)
    ax4.plot(w, model.theta(w, Phase.BROKEN), label="broken", alpha=alpha)
    ax4.set_xlabel("w")
    ax4.set_ylabel(r"$\theta(w,\phi)$")
    ax4.set_xscale("log")
    ax4.set_yscale("log")
    ax4.legend()

    # ax3.plot(w, model.alpha_n(w, allow_negative=True), label=r"$\alpha_n$", alpha=alpha)
    # ax3.plot(w, model.alpha_plus(w, 0.5*w, allow_negative=True), label=r"$\alpha_+(w_-=0.5w_+)$", alpha=alpha)
    # ax3.set_xlabel("w_n, w_+")
    # ax3.set_ylabel(r"$\alpha$")
    # ax3.legend()

    fig.tight_layout()


if __name__ == "__main__":
    main()
    plt.show()
