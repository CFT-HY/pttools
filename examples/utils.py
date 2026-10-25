"""Utilities for PTtools examples"""

from __future__ import annotations

import os
from pathlib import Path
import typing as tp

import pttools.plot.utils as plot_utils
from pttools.plot.utils import FIG_FORMATS
import pttools.type_hints as th
from pttools.utils.docstrings import copy_docstring_dec

if th.TYPE_CHECKING_OR_DOCS:
    from matplotlib.figure import Figure

#: Figures directory for the examples, which can be changed with the ``PTTOOLS_FIG_DIR`` environment variable,
#: e.g. by the nightly run ``pttools/docs/nightly.py``
FIG_DIR: Path = Path(os.environ.get("PTTOOLS_FIG_DIR") or Path(__file__).resolve().parent / "fig").resolve()
FIG_DIR.mkdir(parents=True, exist_ok=True)

#: Data directory for the files created by the examples, e.g. datasets
DATA_DIR: Path = Path(__file__).resolve().parent.parent / "data"


@copy_docstring_dec(plot_utils.save_and_show_fig)
def save_and_show_fig(
        fig: Figure,
        path: str | os.PathLike[str],
        fig_dir: str | os.PathLike[str] | None = FIG_DIR,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        **kwargs: tp.Any) -> None:
    plot_utils.save_and_show_fig(fig=fig, path=path, fig_dir=fig_dir, formats=formats, makedirs=makedirs, **kwargs)


@copy_docstring_dec(plot_utils.save_and_show_figs)
def save_and_show_figs(
        figs: dict[str, Figure],
        fig_dir: str | os.PathLike[str] | None = FIG_DIR,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        **kwargs: tp.Any) -> None:
    plot_utils.save_and_show_figs(figs=figs, fig_dir=fig_dir, formats=formats, makedirs=makedirs, **kwargs)


@copy_docstring_dec(plot_utils.save_fig)
def save_fig(
        fig: Figure,
        path: str | os.PathLike[str],
        fig_dir: str | os.PathLike[str] | None = FIG_DIR,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        **kwargs: tp.Any) -> None:
    plot_utils.save_fig(fig=fig, path=path, fig_dir=fig_dir, formats=formats, makedirs=makedirs, **kwargs)


@copy_docstring_dec(plot_utils.save_figs)
def save_figs(
        figs: dict[str, Figure],
        fig_dir: str | os.PathLike[str] | None = FIG_DIR,
        formats: tp.Iterable[str] = FIG_FORMATS,
        makedirs: bool = True,
        **kwargs: tp.Any) -> None:
    plot_utils.save_figs(figs=figs, fig_dir=fig_dir, formats=formats, makedirs=makedirs, **kwargs)
