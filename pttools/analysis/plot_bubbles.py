"""Utilities for plotting multiple bubbles"""

import typing as tp

import matplotlib.pyplot as plt
import numpy as np

from pttools.analysis.utils import FigAndAxes, create_fig_ax, legend
from pttools.bubble.bubble import BaseBubble

XI_LABEL = r"$\xi$"
V_LABEL = "$v$"
W_LABEL = "$w$"


# -----
# Common plotting functions
# -----

def setup_bubbles_plot(
        bubbles: tp.Collection[BaseBubble],
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None) -> FigAndAxes:
    """Set up a figure for plotting multiple bubbles"""
    for bubble in bubbles:
        if not bubble.solved:
            bubble.solve()
    fig, ax = create_fig_ax(fig, ax)
    return fig, ax


def setup_bubbles_plot_multifig(fig: plt.Figure | None = None) -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Set up the figure and axes for a bubble plot"""
    if fig is None:
        fig = plt.figure()
    ax_v = fig.add_subplot(211)
    ax_w = fig.add_subplot(212, sharex=ax_v)
    ax_v.tick_params("x", labelbottom=False)
    fig.tight_layout()
    return fig, ax_v, ax_w


def plot_bubbles_common(
        bubbles: tp.Collection[BaseBubble],
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        path: str | None = None) -> FigAndAxes:
    """Common steps for plotting multiple bubbles"""
    ax.set_xlabel(XI_LABEL)
    xi_min = np.nanmin([bubble.xi[1] for bubble in bubbles])
    xi_max = np.nanmax([bubble.xi[-2] for bubble in bubbles])
    ax.set_xlim(
        np.nanmax([xi_min * 0.9, 0]),
        np.nanmin([xi_max * 1.1, 1])
    )
    ax.grid()
    if len(bubbles) > 1:
        legend(ax)

    if path is not None:
        fig.savefig(path)
    return fig, ax


def plot_bubbles(
        bubbles: tp.Collection[BaseBubble],
        fig: plt.Figure | None = None,
        path: str | None = None,
        **kwargs) -> plt.Figure:
    """Plot the velocity and enthalpy profiles of bubbles"""
    fig, ax_v, ax_w = setup_bubbles_plot_multifig(fig)
    plot_bubbles_v(bubbles, fig, ax_v, **kwargs)
    plot_bubbles_w(bubbles, fig, ax_w, **kwargs)
    if len(bubbles) == 1:
        fig.suptitle(bubbles[0].label_latex)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path)
    return fig


# -----
# Individual plotting functions
# -----

def plot_bubbles_v(
        bubbles: tp.Collection[BaseBubble],
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        path: str | None = None,
        v_max: float = 1,
        **kwargs) -> FigAndAxes:
    """Plot the velocity profile of multiple bubbles"""
    fig, ax = setup_bubbles_plot(bubbles, fig, ax)
    for bubble in bubbles:
        if "label" in kwargs:
            ax.plot(bubble.xi, bubble.v, **kwargs)
        else:
            ax.plot(bubble.xi, bubble.v, label=bubble.label_latex, **kwargs)
    ax.set_ylabel(V_LABEL)
    ax.set_ylim(0, v_max)
    return plot_bubbles_common(bubbles, fig, ax, path)


def plot_bubbles_w(
        bubbles: tp.Collection[BaseBubble],
        fig: plt.Figure | None = None,
        ax: plt.Axes | None = None,
        path: str | None = None,
        **kwargs) -> FigAndAxes:
    """Plot the enthalpy profile of multiple bubbles"""
    fig, ax = setup_bubbles_plot(bubbles, fig, ax)
    for bubble in bubbles:
        if "label" in kwargs:
            ax.plot(bubble.xi, bubble.w, **kwargs)
        else:
            ax.plot(bubble.xi, bubble.w, label=bubble.label_latex, **kwargs)
    ax.set_ylabel(W_LABEL)
    return plot_bubbles_common(bubbles, fig, ax, path)
