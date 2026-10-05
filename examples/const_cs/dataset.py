r"""
Dataset of spectra
==================

Compute a dataset of gravitational wave spectra for the constant sound speed model,
and export it to an HDF5 file with :py:class:`pttools.export.exporter.Exporter`.

The parameters are varied on a grid, which is created by :py:meth:`Grid.create`:

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
The models are created with the same initial parameters :py:data:`MODEL_KWARGS`,
from which $a_s$ and $V_s$ are adjusted to allow $\alpha_n$ down to 0.05 where possible.
This keeps the smallest $\alpha_n$ of the grid away from the limit of the model,
where the bubbles would be marked as failed due to negative entropy fluxes.

The spectra are computed for the fixed frequencies :py:data:`F`.
The model is shared by the bubbles of the same sound speeds,
and the bubble by the spectra of the same $\alpha_n$ and $v_\text{wall}$.
Therefore, each bubble and its spectra are computed in the same task of a worker process,
and the models are pickled to the worker processes, which preserves their identifiers.
Each model and bubble is therefore written to the file only once.

The bubbles of some parameters cannot be solved correctly,
e.g. deflagrations with a large $\alpha_n$ and a small $v_\text{wall}$, which don't exist,
or are marked as failed due to the validations of the solution.
These are kept in the file with their error flags, such as ``failed`` and ``solver_failed``,
so that they can be filtered out by the user of the dataset.
The spectra are computed for all the bubbles that have a fluid profile, including the failed ones.

The accuracy settings :py:data:`ACCURACY` have been chosen with :py:mod:`examples.props.n_points`.

The computation of the large dataset takes several days even with dozens of CPU cores,
and therefore this example is not run when building the documentation.
Run it from the root directory of the repository with e.g.

.. code-block:: bash

    python -m examples.const_cs.dataset --size small --dry-run
    python -m examples.const_cs.dataset --size small
    python -m examples.const_cs.dataset --size large --path /scratch/const_cs_large.h5

The file can then be read with :py:class:`pttools.export.importer.Importer`.
The models, bubbles and spectra refer to their parents by the row indices in the columns
``/bubbles/model`` and ``/spectra_f/bubble``.
"""

import argparse
from collections.abc import Iterator
import concurrent.futures as cf
import dataclasses
import datetime
import itertools
import logging
from pathlib import Path
import time
import typing as tp

import numpy as np

from pttools.bubble import Bubble
from pttools.export import Exporter, Extractor, FieldShape, Record, Table
from pttools.export.exporter import SizeEstimate, estimate_size
from pttools.models import ConstCSModel
from pttools.omgw0 import Spectrum
from pttools.speedup.options import MAX_WORKERS_DEFAULT
from pttools.speedup.parallel import create_process_pool
from pttools.ssm.nucleation import NucType
import pttools.type_hints as th

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

#: Default path of the file
DEFAULT_DIR: Path = Path(__file__).resolve().parents[2] / "data"

#: Minimum number of points per parameter range
MIN_POINTS: int = 2

#: Default interval of the progress messages in seconds
DEFAULT_PROGRESS_INTERVAL: float = 300


@dataclasses.dataclass(frozen=True)
class Accuracy:
    r"""Accuracy settings of the bubbles and spectra.

    :param n_xi: number of points of the fluid profiles of the bubbles
    :param n_z_lookup: number of points of the lookup arrays of the spectra
    :param nx_P_tilde_gw: number of points of the integration of $\tilde{P}_\text{gw}$,
        or None for the same as ``n_z_lookup``
    :param nT: number of points of the bubble lifetime distribution integration
    """

    n_xi: int
    n_z_lookup: int
    nx_P_tilde_gw: int | None  # noqa: N815
    nT: int  # noqa: N815

    def spectrum_kwargs(self) -> dict[str, tp.Any]:
        """The accuracy settings as arguments of :py:class:`pttools.omgw0.spectrum.Spectrum`."""
        return {"n_z_lookup": self.n_z_lookup, "nx_P_tilde_gw": self.nx_P_tilde_gw, "nT": self.nT}


#: Accuracy settings of the dataset, see :py:mod:`examples.props.n_points`
ACCURACY: Accuracy = Accuracy(n_xi=10000, n_z_lookup=5000, nx_P_tilde_gw=200, nT=1000)

#: Low accuracy settings for compiling the Numba functions with a fast computation
WARM_UP_ACCURACY: Accuracy = Accuracy(n_xi=500, n_z_lookup=200, nx_P_tilde_gw=50, nT=100)

#: The sound speeds of the models for which the Numba functions have been compiled in this process
_WARMED_UP: set[tuple[float, float]] = set()


@dataclasses.dataclass(frozen=True)
class SpectrumParams:
    """Parameters of a spectrum in addition to those of its bubble."""

    nuc_type: NucType
    beta_tilde: float
    T_star: float
    g_star: float


@dataclasses.dataclass(frozen=True)
class Grid:
    """The parameter grid of the dataset."""

    css2s: th.FloatArr1D
    csb2s: th.FloatArr1D
    alpha_ns: th.FloatArr1D
    v_walls: th.FloatArr1D
    beta_tildes: th.FloatArr1D
    T_stars: th.FloatArr1D
    g_stars: th.FloatArr1D
    nuc_types: tuple[NucType, ...] = tuple(NucType)

    @classmethod
    def create(cls, n: int) -> tp.Self:
        """Create a grid with ``n`` points for each parameter range.

        :param n: number of points per parameter range, at least :py:data:`MIN_POINTS`
        :return: the grid
        """
        if n < MIN_POINTS:
            raise ValueError(f"The grid must have at least {MIN_POINTS} points per parameter range. Got: {n}")
        return cls(
            css2s=np.linspace(1/4, 1/3, n),
            csb2s=np.linspace(1/4, 1/3, n),
            alpha_ns=np.geomspace(ALPHA_N_MIN, 1, n),
            v_walls=np.linspace(0.05, 0.95, n),
            beta_tildes=np.geomspace(10, 1e4, n),
            T_stars=np.linspace(50, 500, n),
            g_stars=np.linspace(80, 130, n),
        )

    @property
    def n_models(self) -> int:
        """Number of models."""
        return self.css2s.size * self.csb2s.size

    @property
    def n_bubbles(self) -> int:
        """Number of bubbles."""
        return self.n_models * self.alpha_ns.size * self.v_walls.size

    @property
    def n_spectra_per_bubble(self) -> int:
        """Number of spectra of each bubble."""
        return len(self.nuc_types) * self.beta_tildes.size * self.T_stars.size * self.g_stars.size

    @property
    def n_spectra(self) -> int:
        """Number of spectra."""
        return self.n_bubbles * self.n_spectra_per_bubble

    def model_params(self) -> Iterator[tuple[float, float]]:
        r"""The sound speeds $(c_{s,s}^2, c_{s,b}^2)$ of the models."""
        return itertools.product(self.css2s.tolist(), self.csb2s.tolist())

    def bubble_params(self) -> Iterator[tuple[float, float]]:
        r"""The parameters $(\alpha_n, v_\text{wall})$ of the bubbles of each model."""
        return itertools.product(self.alpha_ns.tolist(), self.v_walls.tolist())

    def spectrum_params(self) -> list[SpectrumParams]:
        """The parameters of the spectra of each bubble."""
        return [
            SpectrumParams(nuc_type=nuc_type, beta_tilde=beta_tilde, T_star=T_star, g_star=g_star)
            for nuc_type, beta_tilde, T_star, g_star in itertools.product(
                self.nuc_types, self.beta_tildes.tolist(), self.T_stars.tolist(), self.g_stars.tolist())
        ]

    def n_rows(self) -> dict[str, int]:
        """Number of rows of each table of the file."""
        return {Table.MODELS: self.n_models, Table.BUBBLES: self.n_bubbles, Table.SPECTRA_F: self.n_spectra}


@dataclasses.dataclass(frozen=True)
class TaskResult:
    """Result of the computation of a bubble and its spectra.

    :param bubble: the record of the bubble, or None if the bubble could not be created
    :param spectra: the records of the spectra
    :param n_skipped: number of spectra that could not be computed
    """

    bubble: Record | None
    spectra: list[Record]
    n_skipped: int


def create_bubble(model: ConstCSModel, alpha_n: float, v_wall: float, accuracy: Accuracy) -> Bubble:
    """Create and solve a bubble."""
    return Bubble(model, v_wall=v_wall, alpha_n=alpha_n, n_xi=accuracy.n_xi, log_success=False)


def create_spectrum(bubble: Bubble, params: SpectrumParams, accuracy: Accuracy) -> Spectrum:
    """Create and compute a spectrum on a single CPU core, as the parallelism is provided by the worker processes."""
    return Spectrum(
        bubble,
        beta_tilde=params.beta_tilde,
        T_star=params.T_star,
        g_star=params.g_star,
        nuc_type=params.nuc_type,
        f=F,
        parallel=False,
        **accuracy.spectrum_kwargs()
    )


def warm_up(model: ConstCSModel) -> None:
    """Compile the Numba functions for the model in this process with a fast low-accuracy computation.

    This way the computation times measured later do not include the compilation.
    A deflagration, a hybrid and a detonation are computed, as they use partially different functions.
    """
    key = (model.css2, model.csb2)
    if key in _WARMED_UP:
        return
    for v_wall in (0.3, 0.7, 0.9):
        bubble = create_bubble(model, alpha_n=0.3, v_wall=v_wall, accuracy=WARM_UP_ACCURACY)
        for nuc_type in NucType:
            create_spectrum(bubble, SpectrumParams(nuc_type, beta_tilde=100, T_star=100, g_star=100), WARM_UP_ACCURACY)
    _WARMED_UP.add(key)


def has_profile(bubble: Bubble) -> bool:
    """Whether the bubble has a fluid profile from which the spectra can be computed."""
    return bubble.solved and not bubble.solver_crashed and bool(np.all(np.isfinite(bubble.v)))


def compute_task(
        model: ConstCSModel,
        alpha_n: float,
        v_wall: float,
        spectrum_params: list[SpectrumParams],
        accuracy: Accuracy,
        extractor: Extractor) -> TaskResult:
    """Compute a bubble and its spectra, and extract their records.

    The records are sent to the main process instead of the objects,
    as they contain only the exported fields instead of all the arrays.
    """
    try:
        bubble = create_bubble(model, alpha_n=alpha_n, v_wall=v_wall, accuracy=accuracy)
    except (ValueError, RuntimeError):
        logger.exception(
            "Could not create the bubble with %s, alpha_n=%s, v_wall=%s", model.label_unicode, alpha_n, v_wall)
        return TaskResult(bubble=None, spectra=[], n_skipped=len(spectrum_params))
    bubble_record = extractor.extract(bubble)
    if not has_profile(bubble):
        logger.error("Skipping the spectra of the bubble without a fluid profile: %s", bubble.label_unicode)
        return TaskResult(bubble=bubble_record, spectra=[], n_skipped=len(spectrum_params))

    spectra: list[Record] = []
    n_skipped = 0
    for params in spectrum_params:
        try:
            spectrum = create_spectrum(bubble, params, accuracy)
        except (ValueError, RuntimeError, IndexError, ZeroDivisionError):
            logger.exception("Could not compute the spectrum of %s with %s", bubble.label_unicode, params)
            n_skipped += 1
            continue
        spectra.append(extractor.extract(spectrum))
    return TaskResult(bubble=bubble_record, spectra=spectra, n_skipped=n_skipped)


def format_size(n_bytes: float) -> str:
    """Format a size in bytes with binary prefixes."""
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n_bytes) < 1024:  # noqa: PLR2004
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TiB"


def format_duration(seconds: float) -> str:
    """Format a duration as days, hours, minutes and seconds."""
    return str(datetime.timedelta(seconds=round(seconds)))


def estimate_dataset(grid: Grid, accuracy: Accuracy, extractor: Extractor) -> tuple[SizeEstimate, float]:
    r"""Estimate the size of the dataset and the computation time of a spectrum.

    A bubble and a spectrum are computed at the center of the grid.
    The number of points of the fluid profiles varies by bubble,
    and therefore the profiles are presumed to have $n_\xi$ points each, which overestimates most of them.
    For large datasets, the size is dominated by the spectra, whose size is the same for all of them.
    The computation time varies by the parameters, and is therefore only a rough estimate.

    :return: the size estimate and the computation time of a spectrum on a single CPU core in seconds
    """
    model = ConstCSModel(css2=float(np.median(grid.css2s)), csb2=float(np.median(grid.csb2s)), **MODEL_KWARGS)
    warm_up(model)
    bubble = create_bubble(model, float(np.median(grid.alpha_ns)), float(np.median(grid.v_walls)), accuracy)
    start_time = time.perf_counter()
    spectrum = create_spectrum(bubble, grid.spectrum_params()[0], accuracy)
    spectrum_time = time.perf_counter() - start_time
    estimate = estimate_size(spectrum, extractor)
    n_profiles = sum(field.shape == FieldShape.RAGGED for field in extractor.fields(Bubble, Table.BUBBLES))
    estimate.row_bytes[Table.BUBBLES] += 8 * n_profiles * (accuracy.n_xi - bubble.xi.size)
    return estimate, spectrum_time


class Progress:
    """Progress monitoring with periodic log messages.

    :param n_spectra: total number of spectra
    :param n_bubbles: total number of bubbles
    :param interval: interval of the progress messages in seconds
    """

    def __init__(self, n_spectra: int, n_bubbles: int, interval: float = DEFAULT_PROGRESS_INTERVAL) -> None:
        """Start the timer."""
        self.n_spectra: int = n_spectra
        self.n_bubbles: int = n_bubbles
        self.interval: float = interval
        self.start_time: float = time.perf_counter()
        self.last_report: float = self.start_time
        #: Number of spectra that have been processed, including the skipped ones
        self.done_spectra: int = 0
        self.done_bubbles: int = 0
        self.skipped_spectra: int = 0
        self.failed_bubbles: int = 0

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds."""
        return time.perf_counter() - self.start_time

    def update(self, result: TaskResult) -> None:
        """Count the results of a task, and report the progress if the interval has passed."""
        self.done_bubbles += 1
        self.done_spectra += len(result.spectra) + result.n_skipped
        self.skipped_spectra += result.n_skipped
        if result.bubble is None or result.bubble.data.get("failed", False):
            self.failed_bubbles += 1
        self.report_if_due()

    def time_to_next_report(self) -> float:
        """Time until the next progress message in seconds."""
        return max(self.last_report + self.interval - time.perf_counter(), 0)

    def report_if_due(self) -> None:
        """Report the progress if the interval has passed since the previous message."""
        if self.time_to_next_report() <= 0:
            self.report()

    def report(self) -> None:
        """Log the progress and the estimated remaining time."""
        self.last_report = time.perf_counter()
        elapsed = self.elapsed
        fraction = self.done_spectra / self.n_spectra if self.n_spectra else 1
        remaining = elapsed * (1 - fraction) / fraction if fraction > 0 else float("nan")
        logger.info(
            "Computed %d/%d spectra (%.1f %%, %d skipped) and %d/%d bubbles (%d failed). "
            "Elapsed: %s, remaining: %s",
            self.done_spectra - self.skipped_spectra, self.n_spectra, 100 * fraction, self.skipped_spectra,
            self.done_bubbles, self.n_bubbles, self.failed_bubbles,
            format_duration(elapsed), "unknown" if np.isnan(remaining) else format_duration(remaining)
        )


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
    :param max_workers: number of worker processes
    :param progress_interval: interval of the progress messages in seconds
    :param accuracy: accuracy settings
    :return: path of the file
    """
    path = DEFAULT_DIR / f"const_cs_dataset_{n_points}.h5" if path is None else Path(path)
    grid = Grid.create(n_points)
    extractor = Extractor()
    estimate, spectrum_time = estimate_dataset(grid, accuracy, extractor)
    n_rows = grid.n_rows()
    logger.info(
        "The dataset will have %d models, %d bubbles and %d spectra (%d per bubble) with %d frequencies each. "
        "Estimated uncompressed size: %s (models: %s, bubbles: %s, spectra: %s). "
        "The compression may reduce the size of the file. "
        "Rough estimate of the computation time with %d worker processes: %s (%.2f s per spectrum)",
        grid.n_models, grid.n_bubbles, grid.n_spectra, grid.n_spectra_per_bubble, F.size,
        format_size(estimate.total(n_rows)),
        *(format_size(estimate.total({table: n_rows[table]})) for table in n_rows),
        max_workers, format_duration(spectrum_time * grid.n_spectra / max_workers), spectrum_time
    )
    if dry_run:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    models = [ConstCSModel(css2=css2, csb2=csb2, **MODEL_KWARGS) for css2, csb2 in grid.model_params()]
    spectrum_params = grid.spectrum_params()
    progress = Progress(n_spectra=grid.n_spectra, n_bubbles=grid.n_bubbles, interval=progress_interval)
    logger.info("Computing the dataset to %s with %d worker processes", path, max_workers)
    with Exporter(path, mode="w" if overwrite else "x") as exporter, \
            create_process_pool(max_workers=max_workers) as pool:
        # The tasks are ordered by model, so that each worker process needs the compiled functions of
        # only a few models at a time.
        pending: set[cf.Future[TaskResult]] = {
            pool.submit(compute_task, model, alpha_n, v_wall, spectrum_params, accuracy, extractor)
            for model in models
            for alpha_n, v_wall in grid.bubble_params()
        }
        while pending:
            done, pending = cf.wait(pending, timeout=progress.time_to_next_report(), return_when=cf.FIRST_COMPLETED)
            for fut in done:
                result = fut.result()
                if result.bubble is not None:
                    exporter.add(result.bubble)
                exporter.add_many(result.spectra)
                progress.update(result)
            progress.report_if_due()
    progress.report()
    logger.info(
        "Exported the dataset to %s in %s. Size of the file: %s",
        path, format_duration(progress.elapsed), format_size(path.stat().st_size)
    )
    return path


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
    parser.add_argument("--workers", type=int, default=MAX_WORKERS_DEFAULT, help="number of worker processes")
    parser.add_argument(
        "--progress-interval", type=float, default=DEFAULT_PROGRESS_INTERVAL,
        help="interval of the progress messages in seconds")
    return parser.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    main(
        n_points=SIZES[_args.size] if _args.n_points is None else _args.n_points,
        path=_args.path,
        overwrite=_args.overwrite,
        dry_run=_args.dry_run,
        max_workers=_args.workers,
        progress_interval=_args.progress_interval,
    )
