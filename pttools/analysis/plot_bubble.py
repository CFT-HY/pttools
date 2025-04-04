"""Plot a single bubble"""

import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.utils import A4_PAPER_SIZE, create_fig_ax, legend
from pttools.bubble.bubble import Bubble


def plot_bubble(bubble: Bubble, fig: plt.Figure = None, path: str = None, **kwargs):
    """Plot the velocity and enthalpy profiles of a bubble"""
    fig, ax_v, ax_w = setup_bubble_plot_multifig(fig)
    plot_bubble_v(bubble, fig, ax_v, **kwargs)
    plot_bubble_w(bubble, fig, ax_w, **kwargs)
    fig.suptitle(bubble.label_latex)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig


def plot_bubble_common(bubble: Bubble, fig: plt.Figure, ax: plt.Axes, path: str = None):
    """Common steps for plotting a bubble"""
    ax.set_xlabel(r"$\xi$")
    ax.set_xlim(
        np.nanmax([bubble.xi[1] / 1.1, 0]),
        np.nanmin([bubble.xi[-2] * 1.1, 1.0])
    )
    ax.grid()
    legend(ax)

    if path is not None:
        fig.savefig(path)
    return fig, ax


def plot_bubble_v(bubble: Bubble, fig: plt.Figure = None, ax: plt.Axes = None, path: str = None, **kwargs):
    """Plot the velocity profile of a bubble"""
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
    """Plot the enthalpy profile of a bubble"""
    if not bubble.solved:
        bubble.solve()
    fig, ax = create_fig_ax(fig, ax)

    ax.plot(bubble.xi, bubble.w, **kwargs)
    ax.set_ylabel(r"$w$")
    w_max = max(line.get_ydata().max() for line in ax.lines) * 1.1
    if np.isnan(w_max):
        w_max = 1
    ax.set_ylim(
        np.nanmax([min(line.get_ydata().min() for line in ax.lines) / 1.1, 0]),
        w_max
    )

    return plot_bubble_common(bubble, fig, ax, path)


def setup_bubble_plot_multifig(fig: plt.Figure = None) -> tp.Tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Set up the figure and axes for a bubble plot"""
    if fig is None:
        fig = plt.figure(figsize=A4_PAPER_SIZE)
    ax_v = fig.add_subplot(211)
    ax_w = fig.add_subplot(212, sharex=ax_v)
    ax_v.tick_params("x", labelbottom=False)
    fig.tight_layout()
    return fig, ax_v, ax_w
