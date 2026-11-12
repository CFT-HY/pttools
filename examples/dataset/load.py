r"""
Loading the dataset of spectra
==============================

Load a dataset created by :ref:`sphx_glr_auto_examples_dataset_save.py`
with :py:class:`pttools.export.importer.Importer`, print a summary of its contents,
and plot a bubble and a spectrum from the stored data with their plotting methods.

The summary is created by :py:meth:`Importer.summary() <pttools.export.importer.Importer.summary>`.
It contains the number of models, bubbles and spectra, the size of the file,
and the ranges of the parameters of each table.
The values of the parameters that have at most :py:data:`~pttools.math.summary.MAX_LISTED_VALUES`
distinct values are listed.
The bubble and the spectrum are imported without recomputing them
with :py:meth:`Importer.import_spectrum() <pttools.export.importer.Importer.import_spectrum>`,
which gives an :py:class:`~pttools.export.spectrum.ImportedSpectrum`
and its :py:class:`~pttools.export.bubble.ImportedBubble`.
These have the same plotting methods as the computed :py:class:`~pttools.omgw0.spectrum.Spectrum`
and :py:class:`~pttools.bubble.bubble.Bubble`.

By default, the first HDF5 file in the ``data`` directory of the repository is loaded.
As the dataset is not available when building the documentation, the example is skipped by Sphinx-Gallery.
Run it from the root directory of the repository with e.g.

.. code-block:: bash

    python -m examples.dataset.load
    python -m examples.dataset.load --path /scratch/const_cs_large.h5 --bubble 10 --spectrum 100
"""

import argparse
import logging
from pathlib import Path

from matplotlib.figure import Figure
import numpy as np

from examples.utils import DATA_DIR, save_and_show_fig
from pttools.docs.examples import is_sphinx_gallery
from pttools.export import Importer, Table
from pttools.math.summary import MAX_LISTED_VALUES
from pttools.utils.time import log_duration

logger: logging.Logger = logging.getLogger(__name__)

#: File extensions of the HDF5 files
HDF5_SUFFIXES: tuple[str, ...] = (".h5", ".hdf5")


def default_path(data_dir: Path = DATA_DIR) -> Path:
    """The first HDF5 file in the data directory in alphabetical order.

    :param data_dir: the data directory
    :return: path of the file
    :raises FileNotFoundError: if there are no HDF5 files in the directory
    """
    paths = sorted(path for path in data_dir.glob("*") if path.suffix in HDF5_SUFFIXES and path.is_file())
    if not paths:
        raise FileNotFoundError(
            f"No HDF5 files found in {data_dir}. Create a dataset with: python -m examples.dataset.save")
    return paths[0]


def select_rows(importer: Importer, bubble: int | None, spectrum: int | None) -> tuple[int, int | None]:
    """Select the bubble and the spectrum to plot.

    If the spectrum is given, the bubble defaults to the bubble of the spectrum.
    Otherwise the bubble defaults to the first bubble that has not failed,
    and the spectrum to the first spectrum of the bubble.

    :param importer: the importer
    :param bubble: row index of the bubble, or None for the default
    :param spectrum: row index of the spectrum in :py:attr:`Table.SPECTRA_F <pttools.export.records.Table.SPECTRA_F>`,
        or None for the default
    :return: the row indices of the bubble and the spectrum, which is None if the bubble has no spectra
    """
    if importer.n_bubbles == 0:
        raise ValueError(f"The file has no bubbles: {importer.path}")
    if spectrum is not None:
        if bubble is None:
            bubble = int(importer.parent_indices(Table.SPECTRA_F, spectrum))
        return bubble, spectrum
    if bubble is None:
        ok = np.flatnonzero(~importer.read(Table.BUBBLES, "failed"))
        bubble = int(ok[0]) if ok.size else 0
    if importer.n_spectra_f == 0:
        return bubble, None
    spectra = np.flatnonzero(importer.parent_indices(Table.SPECTRA_F) == bubble)
    return bubble, int(spectra[0]) if spectra.size else None


@log_duration
def main(
        path: Path | None = None,
        bubble: int | None = None,
        spectrum: int | None = None,
        verify: bool = False,
        max_listed: int = MAX_LISTED_VALUES) -> list[Figure]:
    """Print the summary of a dataset and plot a bubble and a spectrum from it.

    :param path: path of the HDF5 file. Defaults to the first HDF5 file in the ``data`` directory.
    :param bubble: row index of the bubble to plot, see :py:func:`select_rows`
    :param spectrum: row index of the spectrum to plot, see :py:func:`select_rows`
    :param verify: whether to verify the file against its SHA-256 checksum file
    :param max_listed: maximum number of distinct values to list for each parameter
    :return: the figures of the bubble and the spectrum, or only that of the bubble if it has no spectra
    """
    path = default_path() if path is None else Path(path)
    with Importer(path, verify=verify) as importer:
        print(importer.summary(max_listed=max_listed))
        bubble, spectrum = select_rows(importer, bubble, spectrum)
        if spectrum is None:
            logger.warning("The bubble %d has no spectra with shared frequencies to plot.", bubble)
            return [importer.import_bubble(bubble).plot()]
        imported = importer.import_spectrum(spectrum, table=Table.SPECTRA_F)
    # The figures are created from the imported data, and therefore the file can be closed before plotting.
    fig_spectrum, _ax = imported.plot()
    return [imported.bubble.plot(), fig_spectrum]


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Print a summary of a dataset and plot a bubble and a spectrum")
    parser.add_argument("--path", type=Path, help="path of the HDF5 file (default: the first one in data/)")
    parser.add_argument("--bubble", type=int, help="row index of the bubble to plot")
    parser.add_argument("--spectrum", type=int, help="row index of the spectrum to plot")
    parser.add_argument("--verify", action="store_true", help="verify the SHA-256 checksum of the file")
    parser.add_argument(
        "--max-listed", type=int, default=MAX_LISTED_VALUES,
        help="maximum number of distinct values to list for each parameter")
    return parser.parse_args()


# The dataset is not available when building the documentation.
if __name__ == "__main__" and not is_sphinx_gallery():
    _args = parse_args()
    _figs = main(
        path=_args.path, bubble=_args.bubble, spectrum=_args.spectrum, verify=_args.verify,
        max_listed=_args.max_listed)
    save_and_show_fig(_figs[0], "const_cs_dataset_bubble")
    if len(_figs) > 1:
        save_and_show_fig(_figs[1], "const_cs_dataset_spectrum")
