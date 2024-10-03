"""Utilities for plotting and analysing data"""

import typing as tp

import matplotlib as mpl
import matplotlib.pyplot as plt

from pttools.bubble.boundary import Phase
from pttools.models.base import BaseModel
from pttools import speedup

A4_PAPER_SIZE: tp.Tuple[float, float] = (11.7, 8.3)
A3_PAPER_SIZE: tp.Tuple[float, float] = (16.5, 11.7)
ENABLE_DRAWING: bool = not speedup.GITHUB_ACTIONS
FigAndAxes = tp.Tuple[plt.Figure, plt.Axes]


def create_fig_ax(fig: plt.Figure = None, ax: plt.Axes = None) -> FigAndAxes:
    if fig is None:
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
        else:
            fig = ax.get_figure()
    if ax is None:
        ax = fig.add_subplot()
    return fig, ax


def model_phase_label(model: BaseModel, phase: Phase) -> str:
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
