import typing as tp

import matplotlib.pyplot as plt

from pttools.analysis import entropy
from pttools.bubble.bubble import Bubble


def plot_entropy(bubbles: tp.Iterable[Bubble], colors: tp.Iterable[str], fig: plt.Figure = None, ax: plt.Axes = None):
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot()

    for bubble, color in zip(bubbles, colors):
        s, s_ref = entropy.compute_entropy(bubble)
        ax.plot(bubble.xi, s, c=color)
        ax.plot(bubble.xi, s_ref, c=color, ls=":")

    ax.set_xlim(0, 1)
    ax.set_ylim(-5, 50)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("$s$")
    ax.set_title("Entropy")

    return fig
