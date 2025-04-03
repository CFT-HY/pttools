import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.utils import FigAndAxes, create_fig_ax, legend
from pttools.bubble.bubble import Bubble


def setup_bubbles_plot(bubbles: tp.List[Bubble], fig: plt.Figure = None, ax: plt.Axes = None) -> FigAndAxes:
    for bubble in bubbles:
        if not bubble.solved:
            bubble.solve()
    fig, ax = create_fig_ax(fig, ax)
    return fig, ax


def plot_bubbles_common(bubbles: tp.List[Bubble], fig: plt.Figure = None, ax: plt.Axes = None, path: str = None) -> FigAndAxes:
    ax.set_xlabel(r"$\xi$")
    xi_min = np.nanmin([bubble.xi[1] for bubble in bubbles])
    xi_max = np.nanmax([bubble.xi[-2] for bubble in bubbles])
    ax.set_xlim(
        np.nanmax([xi_min / 1.1, 0]),
        np.nanmin([xi_max * 1.1, 1])
    )
    ax.grid()
    legend(ax)

    if path is not None:
        fig.savefig(path)
    return fig, ax


def plot_bubbles_v(bubbles: tp.List[Bubble], fig: plt.Figure = None, ax: plt.Axes = None, path: str = None, v_max: float = 1) -> FigAndAxes:
    fig, ax = setup_bubbles_plot(bubbles, fig, ax)
    for bubble in bubbles:
        ax.plot(bubble.xi, bubble.v, label=bubble.label_latex)
    ax.set_ylabel("$v$")
    ax.set_ylim(0, v_max)
    return plot_bubbles_common(bubbles, fig, ax, path)


def plot_bubbles_w(bubbles: tp.List[Bubble], fig: plt.Figure = None, ax: plt.Axes = None, path: str = None) -> FigAndAxes:
    fig, ax = setup_bubbles_plot(bubbles, fig, ax)
    for bubble in bubbles:
        ax.plot(bubble.xi, bubble.w, label=bubble.label_latex)
    ax.set_ylabel("$w$")
    return plot_bubbles_common(bubbles, fig, ax, path)
