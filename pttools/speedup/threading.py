"""Utilities for the threading layers of Numba.

Numba selects its threading layer only when a function with ``parallel=True`` is first run.
The functions of this module determine the threading layer without launching its threads,
so that :py:mod:`pttools.speedup.options` can warn about the threading layer already when PTtools is imported.
"""

import logging

import numba
import numpy as np

# The TBB library has to be loaded before the Numba threading layer can be predicted.
from pttools.speedup.tbb import TBB_VERSION  # noqa: F401
from pttools.type_hints import FloatArr1D
from pttools.utils import system

logger: logging.Logger = logging.getLogger(__name__)


def check_numba_threading_layer(expected: str | None = None, concurrent: bool = True) -> str:
    """Check that the Numba threading layer works by running a function with ``parallel=True``.

    This initialises the threading layer, i.e. launches its threads.
    With GNU OpenMP, this should therefore not be called before forking worker processes.
    The threading layer can be forced with the environment variable ``NUMBA_THREADING_LAYER``,
    in which case Numba raises an error if it cannot be loaded, instead of falling back to another threading layer.
    This is used by the macOS workflow of the CI to check that OpenMP is available for Numba.

    :param expected: the name of the expected threading layer, e.g. ``"omp"``. If None, any threading layer is accepted.
    :param concurrent: whether to require that the threading layer supports concurrent parallelism,
        see :data:`pttools.speedup.options.CONCURRENT_PARALLELISM`
    :return: the name of the threading layer
    :raises RuntimeError: if Numba JIT compilation is disabled, if the result of the parallel function is wrong,
        if the threading layer is not the expected one, or if it does not support concurrent parallelism when required
    """
    from pttools.speedup import options  # noqa: PLC0415

    if options.NUMBA_DISABLE_JIT:
        raise RuntimeError("The Numba threading layer cannot be checked when Numba JIT compilation is disabled.")
    n = 10
    result = _parallel_sum(np.ones(n))
    layer = numba.threading_layer()
    vendor = numba_openmp_vendor() if layer == "omp" else None
    logger.info("Numba threading layer: %s, OpenMP vendor: %s", layer, vendor)
    if result != n:
        raise RuntimeError(f"The parallel function gave a wrong result with the threading layer {layer}: {result}")
    if expected is not None and layer != expected:
        raise RuntimeError(f"Expected the Numba threading layer {expected}, but got {layer}.")
    if concurrent and layer not in options.CONCURRENT_THREADING_LAYERS:
        raise RuntimeError(f"The Numba threading layer {layer} does not support concurrent parallelism.")
    return layer


def numba_openmp_vendor() -> str | None:
    """Vendor of the OpenMP library of the Numba OpenMP threading layer, e.g. ``"GNU"``, or None if not available."""
    try:
        from numba.np.ufunc import omppool  # noqa: PLC0415
    except ImportError:
        return None
    return str(omppool.openmp_vendor)


@numba.njit(parallel=True, cache=False)
def _parallel_sum(x: FloatArr1D) -> float:
    """Sum with a parallel loop, which initialises the Numba threading layer."""
    s = 0.
    for i in numba.prange(x.size):
        s += x[i]
    return s


def predict_numba_threading_layer() -> str | None:
    """Predict the threading layer that Numba will select, without launching its threads.

    Numba selects the threading layer only when a function with ``parallel=True`` is first run.
    This follows the selection logic of ``numba.np.ufunc.parallel._launch_threads()``,
    including the environment variables ``NUMBA_THREADING_LAYER`` and ``NUMBA_THREADING_LAYER_PRIORITY``,
    but only checks which threading layers can be loaded.
    The threads are not launched, as GNU OpenMP threads that are launched before forking
    would make the forked processes terminate when they use OpenMP,
    and this module is imported also by the forkserver process.

    :return: the name of the threading layer (``"tbb"``, ``"omp"`` or ``"workqueue"``),
        or None if it cannot be determined
    """
    try:
        from numba.core import config  # noqa: PLC0415
        from numba.np.ufunc import parallel  # noqa: PLC0415

        if parallel._is_initialized:  # noqa: SLF001
            return parallel._threading_layer  # noqa: SLF001

        requested = str(config.THREADING_LAYER).lower()
        priority = list(config.THREADING_LAYER_PRIORITY)
        candidates: list[str]
        if requested in priority:
            candidates = [requested]
        elif requested == "default":
            candidates = priority
        elif requested == "safe":
            candidates = ["tbb"]
        elif requested == "threadsafe":
            candidates = ["tbb", "omp"]
        elif requested == "forksafe":
            # GNU OpenMP on Linux is not fork-safe.
            candidates = ["tbb", "workqueue"] if system.IS_LINUX else ["tbb", "omp", "workqueue"]
        else:
            return None
        for layer in candidates:
            if _threading_layer_available(layer):
                return layer
    except Exception as e:
        # This relies on the internals of Numba, which may change.
        logger.debug("Could not predict the Numba threading layer.", exc_info=e)
    return None


def _threading_layer_available(layer: str) -> bool:
    """Whether the given Numba threading layer can be loaded, without launching its threads."""
    from numba.np.ufunc import parallel  # noqa: PLC0415
    try:
        if layer == "tbb":
            parallel._check_tbb_version_compatible()  # noqa: SLF001
            from numba.np.ufunc import tbbpool  # noqa: F401, PLC0415
        elif layer == "omp":
            from numba.np.ufunc import omppool  # noqa: F401, PLC0415
        elif layer == "workqueue":
            from numba.np.ufunc import workqueue  # noqa: F401, PLC0415
        else:
            return False
    except ImportError:
        return False
    return True
