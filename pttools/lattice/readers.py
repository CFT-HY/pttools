r"""Readers of the output and input files of hydrodynamic lattice simulations.

- The input file of the simulation (``input-*``, e.g. :py:data:`INPUT_FILE_NAME`)
  has one ``key value`` pair per line, see :py:func:`find_input_file` and :py:func:`parse_input_file`.
- The log of the global quantities has one row per output step. It's either a CSV file with a header (``*.out``),
  or in the older version of the simulation code of :cutting_2019:`\ `,
  a whitespace-separated file without a header (``globals.txt``), see :py:func:`read_globals`.
- Each power spectrum file ``<quantity>-ps-step<N>.txt`` has three whitespace-separated columns:
  the wavenumber $k$ in lattice units, the power spectrum $\mathcal{P}(k)$
  and the number of lattice modes in the bin, see :py:func:`read_power_spectrum`.
  For the quantities and the normalisation of the power spectra, see :py:mod:`pttools.lattice.quantities`.
  The older version of the simulation code names some of the files differently,
  see :py:data:`pttools.lattice.convert.LEGACY_SPECTRUM_FILES`.
- The bubble location file (:py:data:`BUBBLE_LOCS_FILE_NAME`) has one row ``step x y z`` per bubble,
  see :py:func:`read_bubble_locs`.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

import pttools.type_hints as th

__all__ = [
    "BUBBLE_LOCS_COLUMNS",
    "BUBBLE_LOCS_FILE_NAME",
    "GLOBALS_TXT_COLUMNS",
    "INPUT_FILE_NAME",
    "INPUT_FILE_PATTERN",
    "find_input_file",
    "parse_input_file",
    "read_bubble_locs",
    "read_globals",
    "read_globals_csv",
    "read_globals_txt",
    "read_power_spectrum",
]

logger: logging.Logger = logging.getLogger(__name__)

#: The columns of the bubble location file, see :py:func:`read_bubble_locs`
BUBBLE_LOCS_COLUMNS: tuple[str, ...] = ("step", "x", "y", "z")
#: Name of the bubble location file of the simulation
BUBBLE_LOCS_FILE_NAME: str = "bubblelocs.txt"
#: Names of the columns of the whitespace-separated log of the global quantities (``globals.txt``)
#: of the older version of the simulation code of :cutting_2019:`\ `, by the number of columns.
#: The code wrote more columns in its later versions, and the shared columns have the same meanings,
#: except for the columns ``curl`` and ``unknown 15``, see below.
#: The quantities are integrals over the simulation volume, i.e. sums over the lattice sites times $\delta x^3$,
#: except for ``time``, ``GW Energy``, ``Nb``, ``s_max``, ``gamma_max`` and the numbers of sites and links.
#: Where the quantity is the same as in the ``.out`` files of the later version of the code,
#: the name is that of its column, but without the suffixes ``sym`` and ``broken``,
#: as the older version does not split the quantities between the phases.
#: The meanings were deduced from the data, as the source code of this version is not available:
#:
#: - ``total E``: total energy, whose mean is the "initial avg energy per site" of the logs of the simulations
#: - ``kin fluid``: $\int d^3x \, w \gamma^2 v^2$, so that $\bar{U}_f^2$ = ``kin fluid`` / (``rest E`` + ``pressure``)
#:   reproduces $\bar{U}_{f,\text{max}}$ of :cutting_2019:`\ ` table I
#: - ``pot phi``: $\int d^3x \, V(\phi, T)$, which is $\Delta V_0 \mathcal{V}$ in the symmetric phase
#:   and $\Delta V_0 (T/T_c)^4 \mathcal{V}$ in the broken phase
#: - ``grad phi`` and ``kin phi``: the gradient and the kinetic energies of the scalar field,
#:   whose ratio gives the wall speed $v_w^2$ = ``kin phi`` / ``grad phi``, as in the scripts of :correia_2025:`\ `
#: - ``unknown 6``: not identified. At late times, it's approximately $0.9 \bar{U}_f$,
#:   but it's slightly negative at some early times.
#: - ``GW Energy``: $\rho_\text{gw}$, mean GW energy density, as in the logs of the simulations
#:   ("Unnormalised GW energy [density]"). The integrals of the GW power spectra
#:   of this version of the code are 10-30 % larger, see :py:mod:`pttools.lattice.chw19`.
#: - ``Nb``: number of bubbles
#: - ``rest E``: $\int d^3x \, \epsilon$, rest-frame energy
#: - ``pressure``: $\int d^3x \, p$, which is zero in the first row, as it's not yet computed
#: - ``s_max``: $\eta \gamma_\text{max} \delta t / 2$, a numerical stability parameter
#:   (to a relative precision of $10^{-6}$ in the runs of :cutting_2019:`\ `)
#: - ``gamma_max``: $\gamma_\text{max}$, maximum Lorentz factor of the fluid
#: - ``curl``: $\int d^3x \, \lvert \nabla \times Y \rvert^2$, where $Y$ is $v$ or $J$ depending on the run,
#:   as estimated from the power spectra of the rotational component.
#:   In the runs with 18 columns, it's ``curl J``.
#: - ``div J``: $\int d^3x \, (\nabla \cdot J)^2$
#: - ``unknown 15``: not identified. It's small and fluctuates around zero,
#:   e.g. $\int d^3x \, \nabla \cdot Y$, which would vanish for periodic boundary conditions.
#: - ``N broken`` and ``N links``: the numbers of lattice sites in the broken phase
#:   and of the lattice links that cross the phase boundary
GLOBALS_TXT_COLUMNS: dict[int, tuple[str, ...]] = {
    13: (
        "step", "time", "total E", "kin fluid", "pot phi", "grad phi", "unknown 6", "GW Energy", "Nb", "rest E",
        "pressure", "s_max", "gamma_max"),
}
GLOBALS_TXT_COLUMNS[14] = GLOBALS_TXT_COLUMNS[13] + ("curl",)
GLOBALS_TXT_COLUMNS[15] = GLOBALS_TXT_COLUMNS[14] + ("kin phi",)
GLOBALS_TXT_COLUMNS[16] = GLOBALS_TXT_COLUMNS[15] + ("unknown 15",)
GLOBALS_TXT_COLUMNS[18] = GLOBALS_TXT_COLUMNS[13] + ("curl J", "kin phi", "div J", "N broken", "N links")
#: Name of the input file of the runs of :correia_2025:`\ `
INPUT_FILE_NAME: str = "input-test-largerun"
#: Glob pattern of the names of the input files of the simulation
INPUT_FILE_PATTERN: str = "input-*"


def find_input_file(path: str | os.PathLike[str]) -> Path | None:
    """Find the input file of a simulation run.

    The input file is the file in the run directory whose name matches :py:data:`INPUT_FILE_PATTERN`.
    The backup files of text editors, whose names end with ``~``, are ignored.

    :param path: directory of the run
    :return: the path of the input file, or None if there is none
    :raises ValueError: if there are multiple input files
    """
    files = sorted(
        file for file in Path(path).glob(INPUT_FILE_PATTERN) if file.is_file() and not file.name.endswith("~"))
    if len(files) > 1:
        raise ValueError(
            f"The run directory {path} has multiple input files: {[file.name for file in files]}. "
            "Please specify the input file.")
    return files[0] if files else None


def parse_input_file(path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse the input file of the simulation.

    The file has one ``key value`` pair per line, and comments start with ``#``.

    :param path: path of the input file
    :return: the parameters as strings
    """
    params = {}
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            content = line.split("#", 1)[0].strip()
            if not content:
                continue
            parts = content.split(maxsplit=1)
            params[parts[0]] = parts[1] if len(parts) > 1 else ""
    return params


def read_bubble_locs(path: str | os.PathLike[str]) -> th.IntArr2D:
    r"""Read the bubble location file of the simulation.

    Each row ``step x y z`` gives the time step of the nucleation of a bubble
    and the indices of the lattice site at its centre.
    The positions are therefore in units of the lattice spacing $\delta x$.
    This is also the format of the ``locfile`` of the nucleation of the simulation code.
    In the runs of :cutting_2019:`\ `, all the bubbles are nucleated at the time step 0,
    and the first bubble is always at the origin, as also stated by the logs of the simulations.

    :param path: path of the file
    :return: the rows of the file with the shape (bubbles, 4)
    :raises ValueError: if the file does not have four columns
    """
    data = np.loadtxt(path, dtype=np.int_, ndmin=2)
    if data.shape[1] != len(BUBBLE_LOCS_COLUMNS):
        raise ValueError(
            f"The bubble location file {path} should have the columns {BUBBLE_LOCS_COLUMNS}. "
            f"Got {data.shape[1]} columns.")
    return data


def read_globals(path: str | os.PathLike[str]) -> dict[str, th.FloatArr1D]:
    """Read the log of the global quantities.

    The format is detected from the first line.
    A CSV file with a header (``*.out``) is read with :py:func:`read_globals_csv`,
    and a whitespace-separated file without a header (``globals.txt``) with :py:func:`read_globals_txt`.

    :param path: path of the file
    :return: the columns of the file as arrays, with the column names as keys
    """
    with Path(path).open(encoding="utf-8") as file:
        first_line = file.readline()
    if "," in first_line:
        return read_globals_csv(path)
    return read_globals_txt(path)


def read_globals_csv(path: str | os.PathLike[str]) -> dict[str, th.FloatArr1D]:
    """Read the CSV log of the global quantities (``*.out``).

    The file is read with the C engine of :py:func:`pandas.read_csv`.
    If the run was interrupted, the last row may be incomplete, in which case it's ignored with a warning.

    :param path: path of the ``.out`` file
    :return: the columns of the file as arrays, with the column names as keys
    """
    data = pd.read_csv(path, engine="c", dtype=np.float64, skipinitialspace=True)
    data.columns = [str(name).strip() for name in data.columns]
    # The missing values of an incomplete row are read as nan.
    if len(data) > 0 and data.iloc[-1].isna().any():
        logger.warning("Ignoring the incomplete last row of %s, as the run was probably interrupted.", path)
        data = data.iloc[:-1]
    return {name: data[name].to_numpy(dtype=np.float64) for name in data.columns}


def read_globals_txt(path: str | os.PathLike[str]) -> dict[str, th.FloatArr1D]:
    """Read the whitespace-separated log of the global quantities without a header (``globals.txt``).

    The names of the columns are given by their number, see :py:data:`GLOBALS_TXT_COLUMNS`.
    If the run was interrupted, the last row may be incomplete, in which case it's ignored with a warning.

    :param path: path of the ``globals.txt`` file
    :return: the columns of the file as arrays, with the column names as keys
    :raises ValueError: if the number of columns is not in :py:data:`GLOBALS_TXT_COLUMNS`,
        or if a row other than the last one has a different number of columns
    """
    with Path(path).open(encoding="utf-8") as file:
        rows = [line.split() for line in file if line.strip()]
    if not rows:
        raise ValueError(f"The log of the global quantities is empty: {path}")
    n_cols = len(rows[0])
    if len(rows) > 1 and len(rows[-1]) != n_cols:
        logger.warning("Ignoring the incomplete last row of %s, as the run was probably interrupted.", path)
        rows = rows[:-1]
    if any(len(row) != n_cols for row in rows):
        raise ValueError(f"The rows of the log of the global quantities have different numbers of columns: {path}")
    if n_cols not in GLOBALS_TXT_COLUMNS:
        raise ValueError(
            f"Unknown format of the log of the global quantities with {n_cols} columns: {path}. "
            f"Supported numbers of columns: {sorted(GLOBALS_TXT_COLUMNS)}")
    data = np.array(rows, dtype=np.float64)
    return {name: data[:, i] for i, name in enumerate(GLOBALS_TXT_COLUMNS[n_cols])}


def read_power_spectrum(path: str | os.PathLike[str]) -> tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]:
    r"""Read a power spectrum file.

    The file has the whitespace-separated columns $k$, $\mathcal{P}(k)$ and the number of modes in each bin,
    and no header.
    It's read with :py:func:`numpy.loadtxt`, as it's faster than :py:func:`pandas.read_csv` for these files.
    When reading all the 13132 power spectrum files of the runs of :correia_2025:`\ `
    (about 4000 rows each) with NumPy 2.5 and pandas 3.0, the times were:

    - :py:func:`numpy.loadtxt`: 22.8 s
    - :py:func:`pandas.read_csv` with the C engine: 29.8 s,
      where 0.43 % of the values differed from those of :py:func:`numpy.loadtxt` by one unit of the last place
    - :py:func:`pandas.read_csv` with the C engine and ``float_precision="round_trip"``: 48.8 s,
      where the values were identical to those of :py:func:`numpy.loadtxt`

    :param path: path of the file
    :return: $k$, $\mathcal{P}(k)$ and the number of modes in each bin
    """
    data = np.loadtxt(path, ndmin=2)
    # The implementation with pandas, which is slower, as described above:
    # data = pd.read_csv(path, sep=r"\s+", header=None, engine="c", dtype=np.float64).to_numpy(dtype=np.float64)
    return data[:, 0], data[:, 1], data[:, 2]
