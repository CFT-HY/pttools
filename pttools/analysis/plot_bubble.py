import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.utils import A4_PAPER_SIZE, create_fig_ax
from pttools.bubble.bubble import Bubble


def plot_bubble(bubble: Bubble, fig: plt.Figure = None, path: str = None):
    if fig is None:
        fig = plt.figure(figsize=A4_PAPER_SIZE)
    axs = fig.subplots(1, 2)
    plot_bubble_v(bubble, fig, axs[0])
    plot_bubble_w(bubble, fig, axs[1])
    fig.suptitle(bubble.label_latex)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig


def plot_bubble_common(bubble: Bubble, fig: plt.Figure, ax: plt.Axes, path: str = None):
    ax.set_xlabel(r"$\xi$")
    ax.set_xlim(0, 1)
    ax.grid()

    if path is not None:
        fig.savefig(path)
    return fig, ax


def plot_bubble_v(bubble: Bubble, fig: plt.Figure = None, ax: plt.Axes = None, path: str = None):
    if not bubble.solved:
        bubble.solve()
    fig, ax = create_fig_ax(fig, ax)

    ax.plot(bubble.xi, bubble.v)
    ax.set_ylabel(r"$v$")
    ax.set_ylim(0, 1)
    return plot_bubble_common(bubble, fig, ax, path)


def plot_bubble_w(bubble: Bubble, fig: plt.figure = None, ax: plt.Axes = None, path: str = None):
    if not bubble.solved:
        bubble.solve()
    fig, ax = create_fig_ax(fig, ax)

    ax.plot(bubble.xi, bubble.w)
    ax.set_ylabel(r"$w$")
    ax.set_ylim(0, 1.1*np.max(bubble.w))

    return plot_bubble_common(bubble, fig, ax, path)
