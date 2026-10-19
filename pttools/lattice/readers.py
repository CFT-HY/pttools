r"""Readers of the output and input files of hydrodynamic lattice simulations.

- The input file of the simulation (:py:data:`INPUT_FILE_NAME`)
  has one ``key value`` pair per line, see :py:func:`parse_input_file`.
- The log of the global quantities (``*.out``) is a CSV file with one row per output step,
  see :py:func:`read_globals`.
- Each power spectrum file ``<quantity>-ps-step<N>.txt`` has three whitespace-separated columns:
  the wavenumber $k$ in lattice units, the power spectrum $\mathcal{P}(k)$
  and the number of lattice modes in the bin, see :py:func:`read_power_spectrum`.
  For the quantities and the normalisation of the power spectra, see :py:mod:`pttools.lattice.quantities`.
"""

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

import pttools.type_hints as th

logger: logging.Logger = logging.getLogger(__name__)

#: Default name of the input file of the simulation
INPUT_FILE_NAME: str = "input-test-largerun"


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


def read_globals(path: str | os.PathLike[str]) -> dict[str, th.FloatArr1D]:
    """Read the CSV log of the global quantities.

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
