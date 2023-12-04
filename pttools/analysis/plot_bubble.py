import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.utils import A4_PAPER_SIZE, create_fig_ax
from pttools.bubble.bubble import Bubble


def plot_bubble(bubble: Bubble, fig: plt.Figure = None, path: str = None, **kwargs):
    if fig is None:
        fig = plt.figure(figsize=A4_PAPER_SIZE)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    plot_bubble_v(bubble, fig, ax1, **kwargs)
    plot_bubble_w(bubble, fig, ax2, **kwargs)
    ax1.tick_params("x", labelbottom=False)
    fig.suptitle(bubble.label_latex)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig


def plot_bubble_common(bubble: Bubble, fig: plt.Figure, ax: plt.Axes, path: str = None):
    ax.set_xlabel(r"$\xi$")
    ax.set_xlim(
        max(bubble.xi[1] / 1.1, 0),
        min(bubble.xi[-2] * 1.1, 1.0)
    )
    ax.grid()

    if path is not None:
        fig.savefig(path)
    return fig, ax


def plot_bubble_v(bubble: Bubble, fig: plt.Figure = None, ax: plt.Axes = None, path: str = None, **kwargs):
    if not bubble.solved:
        bubble.solve()
    fig, ax = create_fig_ax(fig, ax)

    ax.plot(bubble.xi, bubble.v, **kwargs)
    ax.set_ylabel(r"$v$")
    ax.set_ylim(0, min(1, 1.1*np.max(bubble.v)))
    return plot_bubble_common(bubble, fig, ax, path)


def plot_bubble_w(bubble: Bubble, fig: plt.figure = None, ax: plt.Axes = None, path: str = None, **kwargs):
    if not bubble.solved:
        bubble.solve()
    fig, ax = create_fig_ax(fig, ax)

    ax.plot(bubble.xi, bubble.w, **kwargs)
    ax.set_ylabel(r"$w$")
    ax.set_ylim(np.min(bubble.w)/1.1, 1.1*np.max(bubble.w))

    return plot_bubble_common(bubble, fig, ax, path)
