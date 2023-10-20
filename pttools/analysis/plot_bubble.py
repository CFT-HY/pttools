import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.utils import create_fig_ax
from pttools.bubble.bubble import Bubble


def plot_bubble(bubble: Bubble, fig: plt.Figure = None, ax: plt.Figure = None, path: str = None):
    if not bubble.solved:
        bubble.solve()
    fig, ax = create_fig_ax(fig, ax)

    ax.plot(bubble.xi, bubble.v)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel(r"$v$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid()

    if path is not None:
        fig.savefig(path)
    return fig, ax


def plot_bubble_w(bubble: Bubble, fig: plt.figure = None, ax: plt.Figure = None, path: str = None):
    if not bubble.solved:
        bubble.solve()
        fig, ax = create_fig_ax(fig, ax)

        ax.plot(bubble.xi, bubble.w)
        ax.set_xlabel(r"$\xi$")
        ax.set_ylabel(r"$w$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.1*np.max(bubble.w))
        ax.grid()

        if path is not None:
            fig.savefig(path)
        return fig, ax
