"""Utilities for plotting and analysing data"""

import matplotlib as mpl
from matplotlib.legend import Legend
import matplotlib.pyplot as plt

from pttools.bubble.boundary import Phase
from pttools.models.base import BaseModel
from pttools import speedup

A4_PAPER_SIZE: tuple[float, float] = (11.7, 8.3)
A3_PAPER_SIZE: tuple[float, float] = (16.5, 11.7)
ENABLE_DRAWING: bool = not speedup.GITHUB_ACTIONS
FigAndAxes = tuple[plt.Figure, plt.Axes]


def create_fig_ax(
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        figsize: tuple[float, float] = None) -> FigAndAxes:
    """Create a figure and axes if necessary"""
    if fig is None:
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot()
        else:
            fig = ax.get_figure()
    if ax is None:
        ax = fig.add_subplot()
    return fig, ax


def legend(ax: plt.Axes, **kwargs) -> Legend | None:
    """Add a legend to the axes if there are any legend labels"""
    return None if ax.get_legend_handles_labels() == ([], []) else ax.legend(**kwargs)


def model_phase_label(model: BaseModel, phase: Phase) -> str:
    """Get the label text for the model and phase"""
    if phase == Phase.SYMMETRIC:
        phase_str = "s"
    elif phase == Phase.BROKEN:
        phase_str = "b"
    else:
        phase_str = f"{phase:.2f}"
    return rf"{model.label_latex}, $\phi$={phase_str}"


def setup_plotting(font: str = "serif", font_size: int = 20, usetex: bool = True) -> None:
    """Get decent-sized plots.

    LaTeX can cause problems if the system is not configured correctly.

    :param font: name of the default font
    :param font_size: font size for the labels
    :param usetex: whether to use LaTeX
    """
    plt.rc('text', usetex=usetex)
    plt.rc('font', family=font)

    mpl.rcParams.update({'font.size': font_size})
    mpl.rcParams.update({'lines.linewidth': 1.5})
    mpl.rcParams.update({'axes.linewidth': 2.0})
    mpl.rcParams.update({'axes.labelsize': font_size})
    mpl.rcParams.update({'xtick.labelsize': font_size})
    mpl.rcParams.update({'ytick.labelsize': font_size})
    # but make legend smaller
    mpl.rcParams.update({'legend.fontsize': 14})
