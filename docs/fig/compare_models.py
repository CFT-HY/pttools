import matplotlib.pyplot as plt
import numpy as np

from pttools import models


def main():
    model_bag = models.BagModel(a_s=1.1, a_b=1, V_s=1)
    model_const_cs_like_bag = models.ConstCSModel(a_s=1.1, a_b=1, css2=1/3, csb2=1/3, V_s=1)

    fig: plt.Figure = plt.figure()
    ax: plt.Axes = fig.add_subplot()

    # w = np.linspace(1, 2)
    # w2 = np.linspace(1.2, 1.3)
    phase = np.linspace(0, 1)
    alpha = phase
    ax.plot(model_bag.w_n(alpha), alpha=0.5, label="bag")
    ax.plot(model_const_cs_like_bag.w_n(alpha), alpha=0.5, label="const_cs")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("$w_n$")
    ax.legend()


if __name__ == "__main__":
    main()
    plt.show()
