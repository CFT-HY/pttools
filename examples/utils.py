"""Utilities for PTtools examples"""

import os.path
import typing as tp

from matplotlib.figure import Figure

import pttools.analysis.utils as plot_utils
from pttools.analysis.utils import FIG_FORMATS
from pttools.utils.docstrings import copy_docstrings

#: Figures directory for the examples
FIG_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig")
os.makedirs(FIG_DIR, exist_ok=True)


def save_and_show_fig(
        fig: Figure,
        path: str,
        fig_dir: str | None = FIG_DIR,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        **kwargs) -> None:
    plot_utils.save_and_show_fig(fig=fig, path=path, fig_dir=fig_dir, formats=formats, makedirs=makedirs, **kwargs)


def save_and_show_figs(
        figs: dict[str, Figure],
        fig_dir: str | None = FIG_DIR,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        **kwargs) -> None:
    plot_utils.save_and_show_figs(figs=figs, fig_dir=fig_dir, formats=formats, makedirs=makedirs, **kwargs)


def save_fig(
        fig: Figure,
        path: str,
        fig_dir: str | None = FIG_DIR,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        **kwargs) -> None:
    plot_utils.save_fig(fig=fig, path=path, fig_dir=fig_dir, formats=formats, makedirs=makedirs, **kwargs)


def save_figs(
        figs: dict[str, Figure],
        fig_dir: str | None = FIG_DIR,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        **kwargs) -> None:
    plot_utils.save_figs(figs=figs, fig_dir=fig_dir, formats=formats, makedirs=makedirs, **kwargs)


copy_docstrings({
    save_and_show_fig: plot_utils.save_and_show_fig,
    save_and_show_figs: plot_utils.save_and_show_figs,
    save_fig: plot_utils.save_fig,
    save_figs: plot_utils.save_figs,
})
