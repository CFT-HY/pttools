r"""Computing datasets of gravitational wave spectra on a parameter grid, and exporting them to HDF5 files.

The parameters are given as a :py:class:`~pttools.analysis.grid.Grid`,
and the dataset is computed with :py:func:`compute_dataset`
and exported with :py:class:`pttools.export.exporter.Exporter`.
For an example, see :ref:`sphx_glr_auto_examples_dataset_save.py`.

The spectra are computed for the fixed frequencies $f$.
The model is shared by the bubbles of the same model parameters,
and the bubble by the spectra of the same $\alpha_n$ and $v_\text{wall}$.
Therefore, each bubble and its spectra are computed in the same task of a worker.
When the workers are processes instead of threads (see :py:data:`pttools.speedup.options.USE_THREADS`),
the models are pickled to the worker processes, which preserves their identifiers.
Each model and bubble is therefore written to the file only once.

The bubbles of some parameters cannot be solved correctly,
e.g. deflagrations with a large $\alpha_n$ and a small $v_\text{wall}$, which don't exist,
or are marked as failed due to the validations of the solution.
These are kept in the file with their error flags, such as ``failed`` and ``solver_failed``,
so that they can be filtered out by the user of the dataset.
The spectra are computed for all the bubbles that have a fluid profile
(see :py:attr:`~pttools.bubble.bubble.base.BaseBubble.has_profile`), including the failed ones.

By default, the spectra of each bubble and nucleation type are created from a single spectrum,
which is computed for a $z$ array that covers the frequencies for all their parameter values,
see :py:func:`compute_spectra`.
"""

from collections.abc import Callable
import concurrent.futures as cf
import dataclasses
import logging
import os
from pathlib import Path
import time

import numpy as np

from pttools.analysis.grid import Grid, SpectrumParams
from pttools.bubble import Bubble
from pttools.export import Exporter, Extractor, FieldShape, Record, Table
from pttools.export.exporter import SizeEstimate, estimate_size
from pttools.models.model import Model
from pttools.omgw0 import Spectrum
from pttools.speedup.options import MAX_WORKERS_DEFAULT
from pttools.speedup.parallel import FakeExecutor, get_pool
from pttools.ssm.nucleation import NucType
from pttools.ssm.settings import SSMSettings
import pttools.type_hints as th
from pttools.utils.files import format_size
from pttools.utils.time import format_duration

__all__ = [
    "DEFAULT_PROGRESS_INTERVAL",
    "WARM_UP_ACCURACY",
    "Progress",
    "TaskResult",
    "compute_dataset",
    "compute_spectra",
    "compute_task",
    "create_bubble",
    "create_shared_spectrum",
    "create_spectrum",
    "estimate_dataset",
    "warm_up",
]

logger: logging.Logger = logging.getLogger(__name__)

#: Default interval of the progress messages in seconds
DEFAULT_PROGRESS_INTERVAL: float = 300

#: Low accuracy settings for compiling the Numba functions with a fast computation
WARM_UP_ACCURACY: SSMSettings = SSMSettings(n_xi=500, n_z_lookup=200, nx_P_tilde_gw=50, nT=100)

#: The identifiers of the models for which the Numba functions have been compiled in this process
_WARMED_UP: set[str] = set()


# This is defined before Progress, as Python 3.12 evaluates the type hints of Progress when defining it.
@dataclasses.dataclass(frozen=True, slots=True)
class TaskResult:
    """Result of the computation of a bubble and its spectra.

    :param bubble: the record of the bubble, or None if the bubble could not be created
    :param spectra: the records of the spectra
    :param n_skipped: number of spectra that could not be computed
    """

    bubble: Record | None
    spectra: list[Record]
    n_skipped: int


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


def compute_dataset(
        grid: Grid,
        path: str | os.PathLike[str],
        model_factory: Callable[..., Model],
        f: th.FloatArr1D,
        accuracy: SSMSettings,
        overwrite: bool = False,
        dry_run: bool = False,
        max_workers: int = MAX_WORKERS_DEFAULT,
        progress_interval: float = DEFAULT_PROGRESS_INTERVAL,
        share_spectra: bool = True) -> Path:
    """Compute a dataset of spectra on a parameter grid and export it to an HDF5 file.

    The size of the dataset and the computation time are first estimated with :py:func:`estimate_dataset`.

    :param grid: the parameter grid
    :param path: path of the file
    :param model_factory: function that creates a model from the parameters of
        :py:attr:`Grid.model_axes <pttools.analysis.grid.Grid.model_axes>` given as keyword arguments,
        e.g. ``functools.partial(ConstCSModel, a_s=1.5, a_b=1, V_s=1)``
    :param f: $f$, frequencies of the spectra in Hz
    :param accuracy: accuracy settings
    :param overwrite: whether to overwrite an existing file
    :param dry_run: only log the size estimate without computing the dataset
    :param max_workers: number of workers
    :param progress_interval: interval of the progress messages in seconds
    :param share_spectra: whether to create the spectra of each bubble from shared spectra,
        see :py:func:`compute_spectra`
    :return: path of the file
    :raises TypeError: if the pool of :py:func:`~pttools.speedup.parallel.get_pool` is a single-thread executor
    """
    path = Path(path)
    extractor = Extractor()
    estimate, bubble_time = estimate_dataset(grid, model_factory, f, accuracy, extractor, share_spectra)
    n_rows = grid.n_rows()
    logger.info(
        "The dataset will have %d models, %d bubbles and %d spectra (%d per bubble) with %d frequencies each. "
        "Estimated uncompressed size: %s (models: %s, bubbles: %s, spectra: %s). "
        "The compression may reduce the size of the file. "
        "Rough estimate of the computation time with %d workers: %s (%.2f s per bubble and its spectra)",
        grid.n_models, grid.n_bubbles, grid.n_spectra, grid.n_spectra_per_bubble, f.size,
        format_size(estimate.total(n_rows)),
        *(format_size(estimate.total({table: n_rows[table]})) for table in n_rows),
        max_workers, format_duration(bubble_time * grid.n_bubbles / max_workers), bubble_time
    )
    if dry_run:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    models = [model_factory(**params) for params in grid.model_params()]
    spectrum_params = grid.spectrum_params()
    progress = Progress(n_spectra=grid.n_spectra, n_bubbles=grid.n_bubbles, interval=progress_interval)
    logger.info("Computing the dataset to %s with %d workers", path, max_workers)
    with Exporter(path, mode="w" if overwrite else "x") as exporter, \
            get_pool(max_workers=max_workers) as pool:
        # concurrent.futures.wait() below requires real futures, which the single-thread executor does not give.
        if isinstance(pool, FakeExecutor):
            raise TypeError("The dataset has to be computed with a thread, interpreter or process pool.")
        # The tasks are ordered by model, so that with processes, each worker process needs the compiled functions
        # of only a few models at a time.
        pending: set[cf.Future[TaskResult]] = {
            pool.submit(compute_task, model, alpha_n, v_wall, spectrum_params, f, accuracy, extractor, share_spectra)
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
    logger.info("Exported the dataset to %s. Size of the file: %s", path, format_size(path.stat().st_size))
    return path


def compute_spectra(
        bubble: Bubble,
        spectrum_params: list[SpectrumParams],
        f: th.FloatArr1D,
        accuracy: SSMSettings,
        share_spectra: bool = True) -> tuple[list[Spectrum], int]:
    r"""Compute the spectra of a bubble on a single CPU core.

    With ``share_spectra``, a shared spectrum is computed for each nucleation type
    with :py:func:`create_shared_spectrum`, and the spectra are created from it with
    :py:meth:`pttools.omgw0.spectrum.Spectrum.with_params`, which reuses its computationally heavy part
    and interpolates it to the frequencies.
    This is much faster than computing the spectra separately,
    but the results differ from those of separately computed spectra by the interpolation error,
    see :py:data:`pttools.omgw0.spectrum.Z_COVERING_DENSITY`.
    The nucleation type cannot be changed with :py:meth:`~pttools.omgw0.spectrum.Spectrum.with_params`,
    as it affects the computationally heavy part.
    If the shared spectrum cannot be computed, the spectra of its nucleation type are computed separately.
    Without ``share_spectra``, all the spectra are computed separately with :py:func:`create_spectrum`.
    The spectra that cannot be computed are logged and skipped.

    :param bubble: the bubble
    :param spectrum_params: the parameters of the spectra
    :param f: $f$, frequencies of the spectra in Hz
    :param accuracy: accuracy settings
    :param share_spectra: whether to create the spectra from shared spectra
    :return: the spectra in the order of ``spectrum_params`` without the skipped ones,
        and the number of the skipped spectra
    """
    by_nuc_type: dict[NucType, list[SpectrumParams]] = {}
    for params in spectrum_params:
        by_nuc_type.setdefault(params.nuc_type, []).append(params)
    computed: dict[SpectrumParams, Spectrum] = {}
    n_skipped = 0
    for nuc_type, params_list in by_nuc_type.items():
        shared: Spectrum | None = None
        if share_spectra:
            try:
                shared = create_shared_spectrum(bubble, params_list, f, accuracy)
            except (ValueError, RuntimeError, IndexError, ZeroDivisionError):
                logger.exception(
                    "Could not compute the shared %s spectrum of %s. Computing the spectra separately.",
                    nuc_type, bubble.label_unicode)
        for params in params_list:
            try:
                computed[params] = create_spectrum(bubble, params, f, accuracy) if shared is None \
                    else shared.with_params(
                        beta_tilde=params.beta_tilde, T_star=params.T_star, g_star=params.g_star, f=f)
            except (ValueError, RuntimeError, IndexError, ZeroDivisionError):
                logger.exception("Could not compute the spectrum of %s with %s", bubble.label_unicode, params)
                n_skipped += 1
    return [computed[params] for params in spectrum_params if params in computed], n_skipped


def compute_task(
        model: Model,
        alpha_n: float,
        v_wall: float,
        spectrum_params: list[SpectrumParams],
        f: th.FloatArr1D,
        accuracy: SSMSettings,
        extractor: Extractor,
        share_spectra: bool = True) -> TaskResult:
    r"""Compute a bubble and its spectra, and extract their records.

    The records are sent to the main process instead of the objects,
    as they contain only the exported fields instead of all the arrays.

    :param model: the model of the bubble
    :param alpha_n: $\alpha_n$, transition strength
    :param v_wall: $v_\text{wall}$, wall speed
    :param spectrum_params: the parameters of the spectra
    :param f: $f$, frequencies of the spectra in Hz
    :param accuracy: accuracy settings
    :param extractor: the extractor of the records
    :param share_spectra: whether to create the spectra from shared spectra, see :py:func:`compute_spectra`
    :return: the records of the bubble and its spectra
    """
    try:
        bubble = create_bubble(model, alpha_n=alpha_n, v_wall=v_wall, accuracy=accuracy)
    except (ValueError, RuntimeError):
        logger.exception(
            "Could not create the bubble with %s, alpha_n=%s, v_wall=%s", model.label_unicode, alpha_n, v_wall)
        return TaskResult(bubble=None, spectra=[], n_skipped=len(spectrum_params))
    bubble_record = extractor.extract(bubble)
    if not bubble.has_profile:
        logger.error("Skipping the spectra of the bubble without a fluid profile: %s", bubble.label_unicode)
        return TaskResult(bubble=bubble_record, spectra=[], n_skipped=len(spectrum_params))

    spectra, n_skipped = compute_spectra(bubble, spectrum_params, f, accuracy, share_spectra)
    return TaskResult(
        bubble=bubble_record, spectra=[extractor.extract(spectrum) for spectrum in spectra], n_skipped=n_skipped)


def create_bubble(model: Model, alpha_n: float, v_wall: float, accuracy: SSMSettings) -> Bubble:
    r"""Create and solve a bubble.

    :param model: the model
    :param alpha_n: $\alpha_n$, transition strength
    :param v_wall: $v_\text{wall}$, wall speed
    :param accuracy: accuracy settings
    :return: the bubble
    """
    return Bubble(model, v_wall=v_wall, alpha_n=alpha_n, n_xi=accuracy.n_xi, log_success=False)


def create_shared_spectrum(
        bubble: Bubble,
        spectrum_params: list[SpectrumParams],
        f: th.FloatArr1D,
        accuracy: SSMSettings) -> Spectrum:
    """Create and compute a spectrum from which the spectra of the given parameters can be created.

    The spectrum is computed with :py:meth:`pttools.omgw0.spectrum.Spectrum.covering`
    on a single CPU core, as the parallelism is provided by the workers.

    :param bubble: the bubble
    :param spectrum_params: the parameters of the spectra, which must have the same nucleation type
    :param f: $f$, frequencies of the spectra in Hz
    :param accuracy: accuracy settings for the frequencies, see :py:meth:`pttools.omgw0.spectrum.Spectrum.covering`
    :return: the spectrum
    :raises ValueError: if the parameters have different nucleation types
    """
    nuc_types = {params.nuc_type for params in spectrum_params}
    if len(nuc_types) != 1:
        raise ValueError(f"The spectra must have the same nucleation type. Got: {nuc_types}")
    return Spectrum.covering(
        bubble,
        f,
        T_star=np.unique([params.T_star for params in spectrum_params]),
        g_star=np.unique([params.g_star for params in spectrum_params]),
        beta_tilde=np.unique([params.beta_tilde for params in spectrum_params]),
        settings=accuracy,
        nuc_type=nuc_types.pop(),
        # The datasets may cover both nucleation types with the same beta_tilde values.
        allow_beta_tilde_for_simultaneous=True,
        parallel=False
    )


def create_spectrum(bubble: Bubble, params: SpectrumParams, f: th.FloatArr1D, accuracy: SSMSettings) -> Spectrum:
    """Create and compute a spectrum on a single CPU core, as the parallelism is provided by the workers.

    :param bubble: the bubble
    :param params: the parameters of the spectrum
    :param f: $f$, frequencies of the spectrum in Hz
    :param accuracy: accuracy settings
    :return: the spectrum
    """
    return Spectrum(
        bubble,
        beta_tilde=params.beta_tilde,
        T_star=params.T_star,
        g_star=params.g_star,
        nuc_type=params.nuc_type,
        # The datasets may cover both nucleation types with the same beta_tilde values.
        allow_beta_tilde_for_simultaneous=True,
        f=f,
        parallel=False,
        **accuracy.spectrum_kwargs()
    )


def estimate_dataset(
        grid: Grid,
        model_factory: Callable[..., Model],
        f: th.FloatArr1D,
        accuracy: SSMSettings,
        extractor: Extractor,
        share_spectra: bool = True) -> tuple[SizeEstimate, float]:
    r"""Estimate the size of the dataset and the computation time of a bubble and its spectra.

    A bubble and a spectrum are computed at the center of the grid.
    With ``share_spectra``, a shared spectrum is also computed, see :py:func:`compute_spectra`.
    The number of points of the fluid profiles varies by bubble,
    and therefore the profiles are presumed to have $n_\xi$ points each, which overestimates most of them.
    For large datasets, the size is dominated by the spectra, whose size is the same for all of them.
    The computation time varies by the parameters, and is therefore only a rough estimate.

    :param grid: the parameter grid
    :param model_factory: function that creates a model, see :py:func:`compute_dataset`
    :param f: $f$, frequencies of the spectra in Hz
    :param accuracy: accuracy settings
    :param extractor: the extractor of the records
    :param share_spectra: whether the spectra are created from shared spectra
    :return: the size estimate and the computation time of a bubble and its spectra on a single CPU core in seconds
    """
    model = model_factory(**{name: float(np.median(values)) for name, values in grid.model_axes.items()})
    warm_up(model, f)
    bubble = create_bubble(model, float(np.median(grid.alpha_ns)), float(np.median(grid.v_walls)), accuracy)
    spectrum_params = grid.spectrum_params()
    params = spectrum_params[0]
    start_time = time.perf_counter()
    if share_spectra:
        shared = create_shared_spectrum(
            bubble, [p for p in spectrum_params if p.nuc_type == params.nuc_type], f, accuracy)
        shared_time = time.perf_counter() - start_time
        start_time = time.perf_counter()
        spectrum = shared.with_params(beta_tilde=params.beta_tilde, T_star=params.T_star, g_star=params.g_star, f=f)
        bubble_time = len(grid.nuc_types) * shared_time \
            + grid.n_spectra_per_bubble * (time.perf_counter() - start_time)
    else:
        spectrum = create_spectrum(bubble, params, f, accuracy)
        bubble_time = grid.n_spectra_per_bubble * (time.perf_counter() - start_time)
    estimate = estimate_size(spectrum, extractor)
    n_profiles = sum(field.shape == FieldShape.RAGGED for field in extractor.fields(Bubble, Table.BUBBLES))
    estimate.row_bytes[Table.BUBBLES] += 8 * n_profiles * (accuracy.n_xi - bubble.xi.size)
    return estimate, bubble_time


def warm_up(model: Model, f: th.FloatArr1D, accuracy: SSMSettings = WARM_UP_ACCURACY) -> None:
    """Compile the Numba functions for the model in this process with a fast low-accuracy computation.

    This way the computation times measured later do not include the compilation.
    A deflagration, a hybrid and a detonation are computed, as they use partially different functions.
    Their spectra are computed both separately and from a shared spectrum, see :py:func:`compute_spectra`.
    The models are identified by their :py:attr:`~pttools.models.base.BaseModel.id`,
    so that each model is warmed up only once per process.

    :param model: the model
    :param f: $f$, frequencies of the spectra in Hz
    :param accuracy: accuracy settings of the warm-up computation
    """
    if model.id in _WARMED_UP:
        return
    for v_wall in (0.3, 0.7, 0.9):
        bubble = create_bubble(model, alpha_n=0.3, v_wall=v_wall, accuracy=accuracy)
        spectrum_params = [
            SpectrumParams(nuc_type, beta_tilde=beta_tilde, T_star=100, g_star=100)
            for nuc_type in NucType for beta_tilde in (100, 1000)
        ]
        for share_spectra in (False, True):
            compute_spectra(bubble, spectrum_params, f, accuracy, share_spectra)
    _WARMED_UP.add(model.id)
