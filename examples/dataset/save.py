r"""
Dataset of spectra
==================

Compute a dataset of gravitational wave spectra for the constant sound speed model
with :py:func:`pttools.analysis.dataset.compute_dataset`,
and export it to an HDF5 file with :py:class:`pttools.export.exporter.Exporter`.

The parameters are varied on a grid of :py:class:`pttools.analysis.grid.Grid`,
which is created by :py:func:`~examples.dataset.save.create_grid`:

- $c_{s,s}^2$ and $c_{s,b}^2$, the sound speeds squared of the phases from $1/4$ to $1/3$,
- $\alpha_n$, the transition strength from 0.1 to 1 on a logarithmic axis,
- $v_\text{wall}$, the wall speed from 0.05 to 0.95,
- $\tilde{\beta}$, the nucleation rate parameter from 10 to $10^4$ on a logarithmic axis,
- $T_*$, the temperature at the time of GW production from 50 to 500 GeV,
- $g_*$, the degrees of freedom at the time of GW production from 80 to 130,
- the nucleation type: exponential and simultaneous.

The lower limit of $\alpha_n$ is 0.1 instead of 0.05,
since some of the models of the grid don't allow $\alpha_n$ below about 0.09,
and the grid would then have holes.
The models are created with the same initial parameters :py:data:`~examples.dataset.save.MODEL_KWARGS`,
from which $a_s$ and $V_s$ are adjusted to allow $\alpha_n$ down to 0.05 where possible.
This keeps the smallest $\alpha_n$ of the grid away from the limit of the model,
where the bubbles would be marked as failed due to negative entropy fluxes.

The spectra are computed for the fixed frequencies :py:data:`~examples.dataset.save.F`.
For the details of the computation, e.g. the handling of the bubbles that cannot be solved correctly,
see :py:mod:`pttools.analysis.dataset`.

The accuracy settings :py:data:`~examples.dataset.save.ACCURACY` have been chosen with
:ref:`sphx_glr_auto_examples_props_n_points.py`.

The computation of the large dataset takes several days even with dozens of CPU cores,
and therefore the computation is skipped when this example is run by Sphinx-Gallery
for building the documentation.
Run it from the root directory of the repository with e.g.

.. code-block:: bash

    python -m examples.dataset.save --size small --dry-run
    python -m examples.dataset.save --size small
    python -m examples.dataset.save --size large --path /scratch/const_cs_large.h5

The file can then be read with :py:class:`pttools.export.importer.Importer`.
The models, bubbles and spectra refer to their parents by the row indices in the columns
``/bubbles/model`` and ``/spectra_f/bubble``.
"""

import argparse
import functools
import logging
from pathlib import Path
import typing as tp

import numpy as np

from examples.utils import DATA_DIR
from pttools.analysis.dataset import DEFAULT_PROGRESS_INTERVAL, compute_dataset
from pttools.analysis.grid import Grid
from pttools.docs.examples import is_sphinx_gallery
from pttools.models import ConstCSModel
from pttools.speedup.options import MAX_WORKERS_DEFAULT
from pttools.ssm.accuracy import Accuracy
import pttools.type_hints as th
from pttools.utils.time import log_duration

logger: logging.Logger = logging.getLogger(__name__)

#: $f$, frequencies of the spectra in Hz
F: th.FloatArr1D = np.logspace(-6, 0, 100)

#: Minimum $\alpha_n$ of the grid
ALPHA_N_MIN: float = 0.1

#: Parameters of the models in addition to the sound speeds.
#: The models are adjusted to allow $\alpha_n$ below :py:data:`ALPHA_N_MIN`,
#: since the bubbles with $\alpha_n$ at the limit of the model would be marked as failed
#: due to the negative entropy fluxes.
MODEL_KWARGS: dict[str, tp.Any] = {"a_s": 1.5, "a_b": 1, "V_s": 1, "alpha_n_min": 0.05, "log_info": False}

#: Number of points per parameter range for the sizes of the dataset
SIZES: dict[str, int] = {"small": 3, "large": 10}

#: Minimum number of points per parameter range
MIN_POINTS: int = 2

#: Accuracy settings of the dataset, see :py:mod:`examples.props.n_points`
ACCURACY: Accuracy = Accuracy(n_xi=10000, n_z_lookup=5000, nx_P_tilde_gw=200, nT=1000)


def create_grid(n: int) -> Grid:
    """Create the parameter grid of the dataset with ``n`` points for each parameter range.

    :param n: number of points per parameter range, at least :py:data:`MIN_POINTS`
    :return: the grid
    """
    if n < MIN_POINTS:
        raise ValueError(f"The grid must have at least {MIN_POINTS} points per parameter range. Got: {n}")
    return Grid(
        model_axes={"css2": np.linspace(1/4, 1/3, n), "csb2": np.linspace(1/4, 1/3, n)},
        alpha_ns=np.geomspace(ALPHA_N_MIN, 1, n),
        v_walls=np.linspace(0.05, 0.95, n),
        beta_tildes=np.geomspace(10, 1e4, n),
        T_stars=np.linspace(50, 500, n),
        g_stars=np.linspace(80, 130, n),
    )


@log_duration
def main(
        n_points: int = SIZES["small"],
        path: Path | None = None,
        overwrite: bool = False,
        dry_run: bool = False,
        max_workers: int = MAX_WORKERS_DEFAULT,
        progress_interval: float = DEFAULT_PROGRESS_INTERVAL,
        accuracy: Accuracy = ACCURACY) -> Path:
    """Compute the dataset and export it to an HDF5 file.

    :param n_points: number of points per parameter range, see :py:data:`SIZES`
    :param path: path of the file. Defaults to ``data/const_cs_dataset_N.h5`` in the root of the repository.
    :param overwrite: whether to overwrite an existing file
    :param dry_run: only print the size estimate without computing the dataset
    :param max_workers: number of workers
    :param progress_interval: interval of the progress messages in seconds
    :param accuracy: accuracy settings
    :return: path of the file
    """
    return compute_dataset(
        grid=create_grid(n_points),
        path=DATA_DIR / f"const_cs_dataset_{n_points}.h5" if path is None else Path(path),
        model_factory=functools.partial(ConstCSModel, **MODEL_KWARGS),
        f=F,
        accuracy=accuracy,
        overwrite=overwrite,
        dry_run=dry_run,
        max_workers=max_workers,
        progress_interval=progress_interval,
    )


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments."""
    parser = argparse.ArgumentParser(description="Compute a dataset of spectra for the constant sound speed model")
    size = parser.add_mutually_exclusive_group()
    size.add_argument(
        "--size", choices=SIZES, default="small",
        help=f"size of the dataset, i.e. the number of points per parameter range: {SIZES}")
    size.add_argument("--n-points", type=int, help="custom number of points per parameter range")
    parser.add_argument("--path", type=Path, help="path of the HDF5 file")
    parser.add_argument("--overwrite", action="store_true", help="overwrite an existing file")
    parser.add_argument("--dry-run", action="store_true", help="only print the estimated size of the dataset")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS_DEFAULT, help="number of workers")
    parser.add_argument(
        "--progress-interval", type=float, default=DEFAULT_PROGRESS_INTERVAL,
        help="interval of the progress messages in seconds")
    return parser.parse_args()


# The computation is skipped in the documentation, as it would take too long.
if __name__ == "__main__" and not is_sphinx_gallery():
    _args = parse_args()
    main(
        n_points=SIZES[_args.size] if _args.n_points is None else _args.n_points,
        path=_args.path,
        overwrite=_args.overwrite,
        dry_run=_args.dry_run,
        max_workers=_args.workers,
        progress_interval=_args.progress_interval,
    )
