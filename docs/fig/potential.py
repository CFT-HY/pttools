import matplotlib.pyplot as plt
import numpy as np
# import sympy as sp
# from sympy import abc

from examples.utils import save_and_show
import pttools.type_hints as th


def potential(x: th.FloatOrArr, a: float, b: float, c: float):
    return a*x**4 + b*x**3 + c*x**2


def b_of_det_zero(a: float, c: float) -> float:
    return np.sqrt(32/9 * a * c)


# def solver():
#     a, b, c = abc.a, abc.b, abc.c
#     sol = sp.solve((-b+sp.sqrt(b**2 - 4*a*c))/(2*a) - (-3*b + sp.sqrt(9*b**2 - 32*a*c))/(8*a), b)
#     print(sol)


def main():
    x = np.linspace(0, 2, 50)

    fig: plt.Figure = plt.figure(figsize=(3.6, 3.2))
    ax: plt.Axes = fig.add_subplot()
    a = 1
    c = 1.15
    b = np.array([0, -b_of_det_zero(a, c), -2*np.sqrt(a*c), -2.25, -2.403])
    colors = ["red", "forestgreen", "limegreen", "lime", "blue"]
    labels = ["$T>T_c$", "$T=T_1$", "$T=T_c$", "$T=T_2$", "$T=0$"]
    # b = np.linspace(-1, -2.4, 5)
    for bi, color, label in zip(b, colors, labels):
        ax.plot(x, potential(x, a, bi, c), color=color, label=label)
    ax.set_ylim(-0.5, 0.5)
    ax.set_ylabel("$V_T$")
    ax.set_xlabel(r"$|\phi|$")
    # ax.tick_params(left=False, bottom=False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.text(1.31, -0.48, f"$+v$")
    ax.legend()
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    # solver()
    fig = main()
    save_and_show(fig, "potential")
