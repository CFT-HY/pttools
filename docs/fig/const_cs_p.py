r"""$p(T)$ figure for the constant sound speed model"""

import matplotlib.pyplot as plt
import numpy as np

from pttools.bubble.boundary import Phase
from pttools.models.const_cs import ConstCSModel


def main():
    model = ConstCSModel(a_s=1.1, a_b=1, css2=0.4**2, csb2=1/3, V_s=1, V_b=0)
    crit = model.critical_temp(guess=10)
    temps_b = np.linspace(0.5 * crit, crit)
    temps_s = np.linspace(crit, 1.2 * crit)
    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()
    ax.plot(temps_b, model.p_temp(temps_b, Phase.BROKEN), color="b", label="$p_b$")
    ax.plot(temps_s, model.p_temp(temps_s, Phase.BROKEN), color="b", ls=":")
    ax.plot(temps_b, model.p_temp(temps_b, Phase.SYMMETRIC), color="r", ls=":")
    ax.plot(temps_s, model.p_temp(temps_s, Phase.SYMMETRIC), color="r", label="$p_s$")
    ax.axvline(crit, ls=":", label=r"$T_{crit}$")
    ax.set_xlabel("T")
    ax.set_ylabel("p")
    ax.grid()
    ax.legend()


if __name__ == "__main__":
    main()
    plt.show()
