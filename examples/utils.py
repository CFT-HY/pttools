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
    import pandas as pd

#: Figures directory for the examples, which can be changed with the ``PTTOOLS_FIG_DIR`` environment variable,
#: e.g. by the nightly run :mod:`pttools.docs.nightly`
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


def save_table(
        table: pd.DataFrame,
        name: str,
        fig_dir: str | os.PathLike[str] = FIG_DIR,
        index: bool = True,
        print_table: bool = True,
        **kwargs: tp.Any) -> str:
    """Save a table as Markdown in ``FIG_DIR/md/NAME.md`` and as CSV in ``FIG_DIR/csv/NAME.csv``.

    The subdirectories of the name are kept within the directory of each format.

    :param table: the table
    :param name: name of the table files without an extension
    :param fig_dir: directory of the subdirectories of the formats
    :param index: whether to include the index
    :param print_table: whether to also print the Markdown table
    :param kwargs: arguments for :py:meth:`pandas.DataFrame.to_markdown`, e.g. ``floatfmt``
    :return: the Markdown table
    """
    markdown = table.to_markdown(index=index, **kwargs)
    md_path = Path(fig_dir) / "md" / f"{name}.md"
    csv_path = Path(fig_dir) / "csv" / f"{name}.csv"
    for path in (md_path, csv_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(f"{markdown}\n", encoding="utf-8")
    table.to_csv(csv_path, index=index)
    if print_table:
        print(markdown)
    return markdown
