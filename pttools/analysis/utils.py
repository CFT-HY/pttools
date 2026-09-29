"""Utilities for plotting and analysing data."""

import os
from pathlib import Path
import typing as tp

from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.legend import Legend
import matplotlib.pyplot as plt

from pttools.bubble.phase import Phase
from pttools.models.base import BaseModel
from pttools.utils.system import IS_GITHUB_ACTIONS

#: A4 paper size in inches
A4_PAPER_SIZE: tuple[float, float] = (11.7, 8.3)
#: A3 paper size in inches
A3_PAPER_SIZE: tuple[float, float] = (16.5, 11.7)
#: Whether to enable drawing of the plots
ENABLE_DRAWING: bool = not IS_GITHUB_ACTIONS
#: File formats in which to save the figures by default
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
    """Create a figure and axes if necessary."""
    if fig is None:
        if ax is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot()
        else:
            fig = tp.cast(Figure, ax.get_figure())
    elif ax is None:
        ax = fig.add_subplot()
    return fig, ax


def legend(ax: Axes, **kwargs: tp.Any) -> Legend | None:
    """Add a legend to the axes if there are any legend labels."""
    return None if ax.get_legend_handles_labels() == ([], []) else ax.legend(**kwargs)


def model_phase_label(model: BaseModel, phase: Phase) -> str:
    """Get the label text for the model and phase."""
    if phase == Phase.SYMMETRIC:
        phase_str = "s"
    elif phase == Phase.BROKEN:
        phase_str = "b"
    else:
        phase_str = f"{phase:.2f}"
    return rf"{model.label_latex}, $\phi$={phase_str}"


def save_and_show_fig(
        fig: Figure,
        path: str | os.PathLike[str],
        fig_dir: str | os.PathLike[str] | None = None,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        **kwargs: tp.Any) -> None:
    """Save and show a figure."""
    save_fig(fig=fig, path=path, fig_dir=fig_dir, formats=formats, makedirs=makedirs, **kwargs)
    if ENABLE_DRAWING:
        plt.show()


def save_and_show_figs(
        figs: dict[str, Figure],
        fig_dir: str | os.PathLike[str] | None = None,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        **kwargs: tp.Any) -> None:
    """Save and show figures."""
    save_figs(figs=figs, fig_dir=fig_dir, formats=formats, makedirs=makedirs, **kwargs)
    if ENABLE_DRAWING:
        plt.show()


def save_fig(
        fig: Figure,
        path: str | os.PathLike[str],
        fig_dir: str | os.PathLike[str] | None = None,
        formats: tp.Iterable[str] = FIG_FORMATS,
        force_formats: bool = False,
        makedirs: bool = True,
        close: bool = False,
        **kwargs: tp.Any) -> None:
    """Save a figure.

    If the name of the path has an extension, the figure is saved once in that path.
    Otherwise, it is saved in each of the formats,
    as ``FIG_DIR/FORMAT/PATH.FORMAT`` if the path is relative and ``fig_dir`` is given,
    or as ``PATH.FORMAT`` if not.
    The subdirectories of a relative path are kept within the directory of each format.

    :param fig: figure to save
    :param path: path of the figure, with or without an extension
    :param fig_dir: directory for relative paths
    :param formats: file formats in which to save the figure, if the path has no extension
    :param force_formats: save in each of the formats, even if the name of the path has a dot
    :param makedirs: create the missing parent directories of the figure files
    :param close: close the figure after saving it
    :param kwargs: arguments for :meth:`matplotlib.figure.Figure.savefig`
    """
    path = Path(path)
    if not force_formats and "." in path.name:
        paths = [path if fig_dir is None else Path(fig_dir) / path]
    elif fig_dir is None or path.is_absolute():
        paths = [path.with_name(f"{path.name}.{ext}") for ext in formats]
    else:
        paths = [Path(fig_dir) / ext / path.with_name(f"{path.name}.{ext}") for ext in formats]

    for fig_path in paths:
        if makedirs:
            fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_path, **kwargs)
    if close:
        plt.close(fig)


def save_figs(
        figs: dict[str, Figure],
        fig_dir: str | os.PathLike[str] | None = None,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        close: bool = False,
        **kwargs: tp.Any) -> None:
    """Save figures."""
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
