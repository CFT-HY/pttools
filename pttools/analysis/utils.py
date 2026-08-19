"""Utilities for plotting and analysing data"""

import os
import os.path
import typing as tp

from matplotlib import rcParams
from matplotlib.legend import Legend
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from pttools.bubble.phase import Phase
from pttools.models.base import BaseModel
from pttools.utils.system import IS_GITHUB_ACTIONS

A4_PAPER_SIZE: tuple[float, float] = (11.7, 8.3)
A3_PAPER_SIZE: tuple[float, float] = (16.5, 11.7)
ENABLE_DRAWING: bool = not IS_GITHUB_ACTIONS
FIG_FORMATS = ("eps", "pdf", "png", "svg")
type FigAndAxes = tuple[Figure, Axes]


def close_figs(*figs: Figure | None) -> None:
    for fig in figs:
        if fig is not None:
            plt.close(fig)


def create_fig_ax(
        fig: Figure | None = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None) -> FigAndAxes:
    """Create a figure and axes if necessary"""
    if fig is None:
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot()
        else:
            fig = ax.get_figure()
    elif ax is None:
        ax = fig.add_subplot()
    return fig, ax


def legend(ax: Axes, **kwargs) -> Legend | None:
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


def save_and_show_fig(
        fig: Figure,
        path: str,
        fig_dir: str | None = None,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        **kwargs) -> None:
    """Save and show a figure"""
    save_fig(fig=fig, path=path, fig_dir=fig_dir, formats=formats, makedirs=makedirs, **kwargs)
    if ENABLE_DRAWING:
        plt.show()


def save_and_show_figs(
        figs: dict[str, Figure],
        fig_dir: str | None = None,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        **kwargs) -> None:
    """Save and show figures"""
    save_figs(figs=figs, fig_dir=fig_dir, formats=formats, makedirs=makedirs, **kwargs)
    if ENABLE_DRAWING:
        plt.show()


def save_fig(
        fig: Figure,
        path: str,
        fig_dir: str | None = None,
        formats: tp.Iterable[str] = FIG_FORMATS,
        force_formats: bool = False,
        makedirs: bool = True,
        close: bool = False,
        **kwargs) -> None:
    """Save a figure"""
    is_abs = os.path.isabs(path)
    if makedirs and (is_abs or fig_dir is None):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    if not force_formats and "." in os.path.basename(path):
        fig.savefig(path if fig_dir is None else os.path.join(fig_dir, path), **kwargs)
    elif fig_dir is None or is_abs:
        for ext in formats:
            fig.savefig(f"{path}.{ext}", **kwargs)
    else:
        for ext in formats:
            format_dir = os.path.join(fig_dir, ext)
            if makedirs and not os.path.exists(format_dir):
                os.makedirs(format_dir, exist_ok=True)
            fig.savefig(f"{os.path.join(format_dir, path)}.{ext}", **kwargs)
    if close:
        plt.close(fig)


def save_figs(
        figs: dict[str, Figure],
        fig_dir: str | None = None,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        close: bool = False,
        **kwargs) -> None:
    """Save figures"""
    for path, fig in figs.items():
        save_fig(fig=fig, path=path, fig_dir=fig_dir, formats=formats, makedirs=makedirs, close=close, **kwargs)


def setup_plotting(
        axes_labelsize: int | None = None,
        axes_linewidth: float = 2.,
        font: str = "serif",
        font_size: int = 20,
        legend_fontsize: int  = 14,
        lines_linewidth: float = 1.5,
        usetex: bool = True) -> None:
    """Get decent-sized plots.

    LaTeX can cause problems if the system is not configured correctly.

    :param axes_labelsize: axes.labelsize
    :param axes_linewidth: axes.linewidth
    :param font: name of the default font
    :param font_size: font size for the labels
    :param legend_fontsize: legend.fontsize
    :param lines_linewidth: lines.linewidth
    :param usetex: whether to use LaTeX
    """
    plt.rc("text", usetex=usetex)
    plt.rc("font", family=font)
    rcParams.update({
        "axes.labelsize": font_size if axes_labelsize is None else axes_labelsize,
        "axes.linewidth": axes_linewidth,
        "font.size": font_size,
        "legend.fontsize": legend_fontsize,
        "lines.linewidth": lines_linewidth,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size
    })
