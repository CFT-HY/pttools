"""Utilities for plotting the entropy of bubbles"""

import typing as tp

import matplotlib.pyplot as plt

from pttools.analysis import entropy
from pttools.bubble.bubble import Bubble


def plot_entropy(bubbles: tp.Iterable[Bubble], colors: tp.Iterable[str], fig: plt.Figure = None):
    if fig is None:
        fig = plt.figure()
    ax1 = fig.add_subplot()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax2 = fig.add_subplot(1, 2, 2, sharex=ax1)

    for bubble, color in zip(bubbles, colors):
        s = entropy.compute_entropy(bubble)
        ax1.plot(bubble.xi, s, c=color)
        ax1.plot(bubble.xi, bubble.s, c=color, ls=":")

        # ax2.plot(bubble.xi, bubble.v, c=color)
        # ax2.plot(bubble.xi, bubble.phase, c=color)

    ax1.set_xlim(0, 1)
    # ax.set_ylim(-5, 100)
    ax1.set_xlabel(r"$\xi$")
    ax1.set_ylabel("$s$")
    ax1.set_title("Entropy")

    # ax2.set_xlabel(r"$\xi$")
    # ax2.set_ylabel("$v$")

    return fig
