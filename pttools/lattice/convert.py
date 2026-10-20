r"""Conversion of the output files of a hydrodynamic lattice simulation run to a single HDF5 file.

Reading the thousands of text files of the power spectra of a run takes seconds,
so they are converted to an HDF5 file in the run directory when the run is first loaded,
and the HDF5 file is read afterwards, see :py:func:`load_run`.
The file contains:

- ``k`` and ``counts``: the wavenumbers $k$ and the numbers of the lattice modes in the bins,
  which are the same in all the power spectrum files of a run, and are therefore stored only once.
- ``spectra/<quantity>/steps`` and ``spectra/<quantity>/power``: the time steps of the power spectra of each quantity,
  and the power spectra as a 2D array with the shape (time steps, bins).
  Each quantity has its own time steps, as some runs do not have all the quantities at all the time steps.
- ``globals/<column>``: the columns of the log of the global quantities (``*.out``), if it's available.
- The attributes of ``input``: the parameters of the input file of the simulation, if it's available.

For the formats of the original files, see :py:mod:`pttools.lattice.readers`.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import tempfile
import typing as tp

import h5py
import numpy as np

from pttools.lattice.quantities import Quantity
from pttools.lattice.readers import INPUT_FILE_NAME, parse_input_file, read_globals, read_power_spectrum
import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)

#: Version of the format of the HDF5 file. Files of other versions are converted again.
FORMAT_VERSION: int = 1
#: Name of the HDF5 file in the run directory
HDF5_FILE_NAME: str = "pttools_lattice_run.h5"
#: Pattern of the names of the power spectrum files
SPECTRUM_FILE_PATTERN: re.Pattern[str] = re.compile(r"(?P<quantity>.+)-ps-step(?P<step>\d+)\.txt")


@dataclass(frozen=True, slots=True)
class RunData:
    """The data of a run as read from its HDF5 file, except for the power spectra, which are read when needed."""

    #: Path of the HDF5 file
    path: Path
    #: $k$, the wavenumbers of the bins in lattice units
    k: th.FloatArr1D
    #: The numbers of the lattice modes in the bins
    counts: th.FloatArr1D
    #: The time steps of the power spectra of each quantity
    steps: dict[Quantity, th.IntArr1D]
    #: The columns of the log of the global quantities, or an empty dict if it was not available
    globals: dict[str, th.FloatArr1D]
    #: The parameters of the input file of the simulation, or an empty dict if it was not available
    params: dict[str, str]

    def power(self, quantity: Quantity) -> th.FloatArr2D:
        """The power spectra of the given quantity at all its time steps.

        :param quantity: the quantity
        :return: the power spectra with the shape (time steps, bins)
        :raises KeyError: if the run does not have the quantity
        """
        with h5py.File(self.path, "r") as file:
            return file[f"spectra/{Quantity(quantity).value}/power"][()]  # pyrefly: ignore[bad-index, bad-return]


def convert_run(
        path: str | os.PathLike[str],
        input_path: str | os.PathLike[str] | None = None,
        out_path: str | os.PathLike[str] | None = None,
        h5_path: str | os.PathLike[str] | None = None) -> Path:
    """Convert the output files of a simulation run to an HDF5 file.

    :param path: directory of the run
    :param input_path: path of the input file, defaults to :py:data:`pttools.lattice.readers.INPUT_FILE_NAME`
        in the run directory. It's optional.
    :param out_path: path of the log of the global quantities, defaults to the last ``*.out`` file
        in the run directory. It's optional.
    :param h5_path: path of the HDF5 file, defaults to :py:data:`HDF5_FILE_NAME` in the run directory
    :return: the path of the HDF5 file
    :raises FileNotFoundError: if the run directory has no power spectrum files
    :raises ValueError: if the wavenumbers or the numbers of modes of the power spectrum files differ
    """
    path = Path(path)
    h5_path = path / HDF5_FILE_NAME if h5_path is None else Path(h5_path)
    k, counts, spectra = _read_spectra(_spectrum_files(path))

    input_path = path / INPUT_FILE_NAME if input_path is None else Path(input_path)
    params = parse_input_file(input_path) if input_path.is_file() else {}
    if out_path is None:
        out_files = sorted(path.glob("*.out"))
        out_path = out_files[-1] if out_files else None
    global_quantities = read_globals(out_path) if out_path is not None and Path(out_path).is_file() else {}

    # The file is written under a temporary name and then renamed,
    # so that an interrupted conversion or a simultaneous conversion by another process
    # does not leave an incomplete file.
    fd, tmp_name = tempfile.mkstemp(suffix=".h5", prefix=f".{h5_path.stem}_", dir=h5_path.parent)
    os.close(fd)
    try:
        with h5py.File(tmp_name, "w") as file:
            file.attrs["format_version"] = FORMAT_VERSION
            file.create_dataset("k", data=k)
            file.create_dataset("counts", data=counts)
            for quantity, (steps, power) in spectra.items():
                group = file.create_group(f"spectra/{quantity.value}")
                group.create_dataset("steps", data=steps)
                group.create_dataset("power", data=power)
            globals_group = file.create_group("globals")
            for name, values in global_quantities.items():
                globals_group.create_dataset(name, data=values)
            input_group = file.create_group("input")
            for key, value in params.items():
                input_group.attrs[key] = value
        Path(tmp_name).replace(h5_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    logger.info("Converted the run %s to %s.", path, h5_path)
    return h5_path


def load_run(
        path: str | os.PathLike[str],
        input_path: str | os.PathLike[str] | None = None,
        out_path: str | os.PathLike[str] | None = None,
        reconvert: bool = False) -> RunData:
    """Load the data of a simulation run, converting it to an HDF5 file first if necessary.

    The run is converted with :py:func:`convert_run` if the HDF5 file does not exist yet,
    if its format version differs from :py:data:`FORMAT_VERSION`, or if ``reconvert`` is true.
    Otherwise, the original files are not read, and ``input_path`` and ``out_path`` are ignored.

    :param path: directory of the run
    :param input_path: path of the input file for the conversion, see :py:func:`convert_run`
    :param out_path: path of the log of the global quantities for the conversion, see :py:func:`convert_run`
    :param reconvert: whether to convert the run again even if the HDF5 file exists
    :return: the data of the run
    """
    h5_path = Path(path) / HDF5_FILE_NAME
    if not reconvert and h5_path.is_file():
        with h5py.File(h5_path, "r") as file:
            version = file.attrs.get("format_version")
        if version != FORMAT_VERSION:
            logger.info("The HDF5 file %s has the format version %s. Converting the run again.", h5_path, version)
            reconvert = True
    if reconvert or not h5_path.is_file():
        convert_run(path, input_path=input_path, out_path=out_path, h5_path=h5_path)
    with h5py.File(h5_path, "r") as file:
        spectra = tp.cast(h5py.Group, file["spectra"])
        return RunData(
            path=h5_path,
            k=file["k"][()],  # pyrefly: ignore[bad-index]
            counts=file["counts"][()],  # pyrefly: ignore[bad-index]
            steps={
                Quantity(name): spectra[f"{name}/steps"][()]  # pyrefly: ignore[bad-index]
                for name in spectra
            },
            globals={
                name: dataset[()] for name, dataset in tp.cast(h5py.Group, file["globals"]).items()
            },
            params={key: str(value) for key, value in file["input"].attrs.items()},
        )


def _read_spectra(
        files: dict[Quantity, dict[int, Path]]
        ) -> tuple[th.FloatArr1D, th.FloatArr1D, dict[Quantity, tuple[th.IntArr1D, th.FloatArr2D]]]:
    """Read the power spectrum files of a run.

    :param files: the files of each quantity by their time steps
    :return: the wavenumbers, the numbers of modes, and the time steps and the power spectra of each quantity
    :raises ValueError: if the wavenumbers or the numbers of modes of the files differ
    """
    first = next(iter(next(iter(files.values())).values()))
    k, _, counts = read_power_spectrum(first)
    spectra = {}
    for quantity in sorted(files):
        steps = np.array(sorted(files[quantity]), dtype=np.int_)
        power = np.empty((steps.size, k.size))
        for i, step in enumerate(steps):
            file = files[quantity][int(step)]
            k_file, p, counts_file = read_power_spectrum(file)
            if not (np.array_equal(k_file, k) and np.array_equal(counts_file, counts)):
                raise ValueError(
                    f"The wavenumbers or the numbers of modes of the power spectrum file {file} differ "
                    f"from those of {first}.")
            power[i] = p
        spectra[quantity] = (steps, power)
    return k, counts, spectra


def _spectrum_files(path: Path) -> dict[Quantity, dict[int, Path]]:
    """Find the power spectrum files of a run.

    :param path: directory of the run
    :return: the files of each quantity by their time steps
    :raises FileNotFoundError: if there are no power spectrum files
    """
    files: dict[Quantity, dict[int, Path]] = {}
    for file in path.iterdir():
        match = SPECTRUM_FILE_PATTERN.fullmatch(file.name)
        if match is None:
            continue
        try:
            quantity = Quantity(match["quantity"])
        except ValueError:
            logger.warning("Skipping the power spectrum file of an unknown quantity: %s", file)
            continue
        files.setdefault(quantity, {})[int(match["step"])] = file
    if not files:
        raise FileNotFoundError(f"No power spectrum files were found in the run directory: {path}")
    return files
