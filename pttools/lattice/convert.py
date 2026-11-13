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
- ``globals/<column>``: the columns of the log of the global quantities (``*.out`` or ``globals.txt``),
  if it's available.
- The attributes of ``input``: the parameters of the input file of the simulation, if it's available.
- ``bubble_locs``: the rows ``step x y z`` of the bubble location file, with the shape (bubbles, 4),
  or with the shape (0, 4) if it's not available, see :py:func:`pttools.lattice.readers.read_bubble_locs`.
- The attributes ``input_file``, ``globals_file`` and ``bubble_locs_file`` of the file:
  the names of the files that were converted, or empty strings if they were not available.

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
from pttools.lattice.readers import (
    BUBBLE_LOCS_COLUMNS,
    BUBBLE_LOCS_FILE_NAME,
    find_input_file,
    parse_input_file,
    read_bubble_locs,
    read_globals,
    read_power_spectrum,
)
import pttools.type_hints as th

__all__ = [
    "FORMAT_VERSION",
    "GLOBALS_TXT_FILE_NAME",
    "HDF5_FILE_NAME",
    "LEGACY_SPECTRUM_FILES",
    "LEGACY_SPECTRUM_FILE_PATTERN",
    "SPECTRUM_FILE_PATTERN",
    "RunData",
    "convert_run",
    "find_spectrum_files",
    "load_run",
]

logger: logging.Logger = logging.getLogger(__name__)

#: Version of the format of the HDF5 file. Files of other versions are converted again.
#: Version 2 added the bubble locations, the names of the converted files and the support for ``globals.txt``.
FORMAT_VERSION: int = 2
#: Name of the log of the global quantities of the older version of the simulation code of :cutting_2019:`\ `
GLOBALS_TXT_FILE_NAME: str = "globals.txt"
#: Name of the HDF5 file in the run directory
HDF5_FILE_NAME: str = "pttools_lattice_run.h5"
#: The prefixes of the names of the power spectrum files of the older version of the simulation code
#: of :cutting_2019:`\ `, ``<prefix>-step<N>.txt``, and their quantities.
#: Their format is the same as that of the later version, see :py:func:`pttools.lattice.readers.read_power_spectrum`.
#: The quantities were identified by comparing the runs that have the files of both naming conventions:
#: the velocity spectra are identical to those of the later naming convention,
#: and the integral of the GW power spectrum is within 10-30 % of the GW energy of the log of the global quantities,
#: see :py:mod:`pttools.lattice.chw19`.
#: The files ``fft-e-<N>.txt``, ``fft-phi-<N>.txt`` and ``fft-<N>.txt`` of this version have a different normalisation,
#: which is not known, and are therefore not imported.
LEGACY_SPECTRUM_FILES: dict[str, Quantity] = {
    "div-ps": Quantity.VEL_DIV,
    "power": Quantity.GW_TT,
    "rot-ps": Quantity.VEL_ROT,
    "tot-ps": Quantity.VEL_TOT,
}
#: Pattern of the names of the power spectrum files of the older version of the simulation code,
#: see :py:data:`LEGACY_SPECTRUM_FILES`
LEGACY_SPECTRUM_FILE_PATTERN: re.Pattern[str] = re.compile(
    rf"(?P<prefix>{'|'.join(re.escape(prefix) for prefix in LEGACY_SPECTRUM_FILES)})-step(?P<step>\d+)\.txt")
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
    #: The rows ``step x y z`` of the bubble location file with the shape (bubbles, 4),
    #: or with the shape (0, 4) if it was not available
    bubble_locs: th.IntArr2D

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
        globals_path: str | os.PathLike[str] | None = None,
        bubble_locs_path: str | os.PathLike[str] | None = None,
        h5_path: str | os.PathLike[str] | None = None) -> Path:
    """Convert the output files of a simulation run to an HDF5 file.

    :param path: directory of the run
    :param input_path: path of the input file, defaults to the file ``input-*`` in the run directory,
        see :py:func:`pttools.lattice.readers.find_input_file`. It's optional.
    :param globals_path: path of the log of the global quantities, defaults to the last ``*.out`` file
        in the run directory, or :py:data:`GLOBALS_TXT_FILE_NAME` if there are no ``*.out`` files.
        It's optional.
    :param bubble_locs_path: path of the bubble location file,
        defaults to :py:data:`pttools.lattice.readers.BUBBLE_LOCS_FILE_NAME` in the run directory. It's optional.
    :param h5_path: path of the HDF5 file, defaults to :py:data:`HDF5_FILE_NAME` in the run directory
    :return: the path of the HDF5 file
    :raises FileNotFoundError: if the run directory has no power spectrum files
    :raises ValueError: if the wavenumbers or the numbers of modes of the power spectrum files differ,
        if there are multiple power spectrum files of the same quantity and time step,
        or if ``input_path`` is not given and the run directory has multiple input files
    """
    path = Path(path)
    h5_path = path / HDF5_FILE_NAME if h5_path is None else Path(h5_path)
    k, counts, spectra = _read_spectra(find_spectrum_files(path))

    input_path = find_input_file(path) if input_path is None else Path(input_path)
    params = parse_input_file(input_path) if input_path is not None and input_path.is_file() else {}
    globals_path = _find_globals_file(path) if globals_path is None else Path(globals_path)
    global_quantities = read_globals(globals_path) if globals_path is not None and globals_path.is_file() else {}
    bubble_locs_path = path / BUBBLE_LOCS_FILE_NAME if bubble_locs_path is None else Path(bubble_locs_path)
    bubble_locs = read_bubble_locs(bubble_locs_path) if bubble_locs_path.is_file() \
        else np.empty((0, len(BUBBLE_LOCS_COLUMNS)), dtype=np.int_)

    # The file is written under a temporary name and then renamed,
    # so that an interrupted conversion or a simultaneous conversion by another process
    # does not leave an incomplete file.
    fd, tmp_name = tempfile.mkstemp(suffix=".h5", prefix=f".{h5_path.stem}_", dir=h5_path.parent)
    os.close(fd)
    try:
        with h5py.File(tmp_name, "w") as file:
            file.attrs["format_version"] = FORMAT_VERSION
            for attr, source in (
                    ("input_file", input_path if params else None),
                    ("globals_file", globals_path if global_quantities else None),
                    ("bubble_locs_file", bubble_locs_path if bubble_locs.size else None)):
                file.attrs[attr] = "" if source is None else source.name
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
            file.create_dataset("bubble_locs", data=bubble_locs)
        Path(tmp_name).replace(h5_path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    logger.info("Converted the run %s to %s.", path, h5_path)
    return h5_path


def _find_globals_file(path: Path) -> Path | None:
    """Find the log of the global quantities of a run, see :py:func:`convert_run`."""
    out_files = sorted(path.glob("*.out"))
    if out_files:
        return out_files[-1]
    globals_txt = path / GLOBALS_TXT_FILE_NAME
    return globals_txt if globals_txt.is_file() else None


def find_spectrum_files(path: Path) -> dict[Quantity, dict[int, Path]]:
    """Find the power spectrum files of a run.

    Both the naming convention :py:data:`SPECTRUM_FILE_PATTERN`
    and the older one :py:data:`LEGACY_SPECTRUM_FILE_PATTERN` are supported.
    Empty files, e.g. of an interrupted run, are skipped with a warning.

    :param path: directory of the run
    :return: the files of each quantity by their time steps
    :raises FileNotFoundError: if there are no power spectrum files
    :raises ValueError: if there are multiple files for the same quantity and time step
    """
    files: dict[Quantity, dict[int, Path]] = {}
    for file in sorted(path.iterdir()):
        legacy_match = LEGACY_SPECTRUM_FILE_PATTERN.fullmatch(file.name)
        if legacy_match is not None:
            quantity = LEGACY_SPECTRUM_FILES[legacy_match["prefix"]]
            step = int(legacy_match["step"])
        else:
            match = SPECTRUM_FILE_PATTERN.fullmatch(file.name)
            if match is None:
                continue
            try:
                quantity = Quantity(match["quantity"])
            except ValueError:
                logger.warning("Skipping the power spectrum file of an unknown quantity: %s", file)
                continue
            step = int(match["step"])
        if file.stat().st_size == 0:
            logger.warning("Skipping the empty power spectrum file: %s", file)
            continue
        quantity_files = files.setdefault(quantity, {})
        if step in quantity_files:
            raise ValueError(
                f"The run directory has multiple power spectrum files of {quantity.value} at the time step {step}: "
                f"{quantity_files[step].name} and {file.name}")
        quantity_files[step] = file
    if not files:
        raise FileNotFoundError(f"No power spectrum files were found in the run directory: {path}")
    return files


def load_run(
        path: str | os.PathLike[str],
        input_path: str | os.PathLike[str] | None = None,
        globals_path: str | os.PathLike[str] | None = None,
        bubble_locs_path: str | os.PathLike[str] | None = None,
        reconvert: bool = False) -> RunData:
    """Load the data of a simulation run, converting it to an HDF5 file first if necessary.

    The run is converted with :py:func:`convert_run` if the HDF5 file does not exist yet,
    if its format version differs from :py:data:`FORMAT_VERSION`, or if ``reconvert`` is true.
    Otherwise, the original files are not read, and the paths of the original files are ignored.

    :param path: directory of the run
    :param input_path: path of the input file for the conversion, see :py:func:`convert_run`
    :param globals_path: path of the log of the global quantities for the conversion, see :py:func:`convert_run`
    :param bubble_locs_path: path of the bubble location file for the conversion, see :py:func:`convert_run`
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
        convert_run(
            path, input_path=input_path, globals_path=globals_path, bubble_locs_path=bubble_locs_path,
            h5_path=h5_path)
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
            bubble_locs=file["bubble_locs"][()],  # pyrefly: ignore[bad-index]
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
