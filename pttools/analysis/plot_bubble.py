import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.utils import A4_PAPER_SIZE, create_fig_ax
from pttools.bubble.bubble import Bubble


def plot_bubble(bubble: Bubble, fig: plt.Figure = None, path: str = None, **kwargs):
    fig, ax_v, ax_w = setup_bubble_plot(fig)
    plot_bubble_v(bubble, fig, ax_v, **kwargs)
    plot_bubble_w(bubble, fig, ax_w, **kwargs)
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

    if ax.get_legend_handles_labels() != ([], []):
        ax.legend()

    if path is not None:
        fig.savefig(path)
    return fig, ax


def plot_bubble_v(bubble: Bubble, fig: plt.Figure = None, ax: plt.Axes = None, path: str = None, **kwargs):
    if not bubble.solved:
        bubble.solve()
    fig, ax = create_fig_ax(fig, ax)

    ax.plot(bubble.xi, bubble.v, **kwargs)
    ax.set_ylabel(r"$v$")
    ax.set_ylim(
        0,
        min(1, 1.1 * max(line.get_ydata().max() for line in ax.lines))
    )
    return plot_bubble_common(bubble, fig, ax, path)


def plot_bubble_w(bubble: Bubble, fig: plt.figure = None, ax: plt.Axes = None, path: str = None, **kwargs):
    if not bubble.solved:
        bubble.solve()
    fig, ax = create_fig_ax(fig, ax)

    ax.plot(bubble.xi, bubble.w, **kwargs)
    ax.set_ylabel(r"$w$")
    ax.set_ylim(
        min(line.get_ydata().min() for line in ax.lines) / 1.1,
        max(line.get_ydata().max() for line in ax.lines) * 1.1
    )

    return plot_bubble_common(bubble, fig, ax, path)


def setup_bubble_plot(fig: plt.Figure = None) -> tp.Tuple[plt.Figure, plt.Axes, plt.Axes]:
    if fig is None:
        fig = plt.figure(figsize=A4_PAPER_SIZE)
    ax_v = fig.add_subplot(211)
    ax_w = fig.add_subplot(212, sharex=ax_v)
    ax_v.tick_params("x", labelbottom=False)
    fig.tight_layout()
    return fig, ax_v, ax_w
