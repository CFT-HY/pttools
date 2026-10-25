"""Options for JIT-compilation and other speedups."""

import logging
import os
import typing as tp

from pttools.speedup.threading import numba_openmp_vendor, predict_numba_threading_layer
from pttools.utils import system

logger: logging.Logger = logging.getLogger(__name__)

# On Windows, the maximum number of worker processes is limited to 61.
# https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor
MAX_WORKERS_WINDOWS: int = 61

#: Default maximum number of parallel worker processes
MAX_WORKERS_DEFAULT: int
if system.AVAILABLE_CPU_CORES is None:
    MAX_WORKERS_DEFAULT = 1
    logger.warning(
        "This platform does not provide info on the number of available CPU cores. Using 1 core. %s",
        system.platform_info()
    )
elif system.IS_WINDOWS and system.AVAILABLE_CPU_CORES > MAX_WORKERS_WINDOWS:
    MAX_WORKERS_DEFAULT = MAX_WORKERS_WINDOWS
else:
    MAX_WORKERS_DEFAULT = system.AVAILABLE_CPU_CORES

USE_THREADS: tp.Final[bool] = bool(int(os.getenv("PTTOOLS_USE_THREADS", "1")))
"""Whether :func:`pttools.speedup.parallel.run_parallel` and the functions that use it
run in threads instead of processes by default.

This is read from the environment variable ``PTTOOLS_USE_THREADS`` (``1`` or ``0``), and is enabled by default.
The default values of the arguments are set when PTtools is imported,
so the environment variable has to be set before importing PTtools.
"""

USE_INTERPRETERS: tp.Final[bool] = bool(int(os.getenv("PTTOOLS_USE_INTERPRETERS", "0")))
"""Whether :func:`pttools.speedup.parallel.run_parallel` and :func:`pttools.speedup.parallel.get_pool`
use subinterpreters (:class:`concurrent.futures.InterpreterPoolExecutor`) instead of processes by default,
when threads are not used (see :data:`USE_THREADS`).

This is read from the environment variable ``PTTOOLS_USE_INTERPRETERS`` (``1`` or ``0``), and is disabled by default,
as subinterpreters are experimental and require Python 3.14 or later.
Furthermore, as of NumPy 2.5, NumPy cannot be imported in subinterpreters
("module numpy._core._multiarray_umath does not support loading in subinterpreters"),
and therefore neither can Numba or PTtools.
Until this is fixed, subinterpreters can only run functions whose arguments and return values
do not require NumPy, which excludes the computations of PTtools.
The environment variable has to be set before importing PTtools.
"""


# The Numba threading layer is not selected until it's needed, and is therefore predicted below.
if not system.FORKING or not system.CPU_AFFINITY:
    msg: str = system.platform_info()
    if not system.CPU_AFFINITY:
        msg += (
            " This platform does not provide info on which CPU cores are available for this process. "
            "Using all cores."
        )
    logger.debug(msg)


#: Whether Numba JIT compilation has been disabled.
NUMBA_DISABLE_JIT: tp.Final[bool] = bool(int(os.getenv("NUMBA_DISABLE_JIT", "0")))

NUMBA_ENABLE_CACHE: bool = \
    bool(int(os.environ["NUMBA_ENABLE_CACHE"])) \
        if "NUMBA_ENABLE_CACHE" in os.environ \
        else (system.IS_GITHUB_ACTIONS or system.IS_PACKAGE or system.IS_READ_THE_DOCS)
"""Whether to enable caching for Numba-jitted functions

This is enabled by default when running on GitHub Actions or when installed as a pip package,
since in those cases the source code is not expected to change.
"""

#: Whether to use NumbaLSODA as the default ODE integrator.
NUMBA_INTEGRATE: tp.Final[bool] = bool(int(os.getenv("NUMBA_INTEGRATE", "0")))
#: Whether to use looser tolerances, which are necessary for the unit tests to pass with NumbaLSODA.
NUMBA_INTEGRATE_TOLERANCES: tp.Final[bool] = bool(
    int(os.getenv("NUMBA_INTEGRATE_TOLERANCES", str(int(NUMBA_INTEGRATE))))
)
#: Whether to use nested parallelism. This requires that either TBB or OpenMP is installed and working.
NUMBA_NESTED_PARALLELISM: tp.Final[bool] = bool(int(os.getenv("NUMBA_NESTED_PARALLELISM", "0")))
#: Default options for the custom JIT decorators
NUMBA_OPTS: dict[str, tp.Any] = {
    "cache": NUMBA_ENABLE_CACHE
}

if NUMBA_INTEGRATE:
    if NUMBA_DISABLE_JIT:
        raise RuntimeError("Numba integration cannot be enabled when Numba is disabled")
    logger.warning("Numba-jitted integration has been globally enabled. The results may not be as accurate.")


#: Threading layers of Numba that support calling functions with ``parallel=True`` from multiple threads concurrently,
#: and nested parallelism. The ``workqueue`` layer supports neither, and terminates the process when attempted.
CONCURRENT_THREADING_LAYERS: tp.Final[frozenset[str]] = frozenset({"tbb", "omp"})

#: Instructions for installing a threading layer that supports concurrent parallelism
THREADING_LAYER_INSTRUCTIONS: tp.Final[str] = (
    "To enable it, install the \"performance\" optional dependencies of PTtools, "
    "e.g. with pip install \"pttools-gw[performance]\" or uv sync --extra performance. "
    "If they don't provide TBB or OpenMP for this platform, install an OpenMP runtime with the system package manager, "
    "e.g. on ARM-based macOS with \"brew install libomp\", followed by adding \"$(brew --prefix libomp)/lib\" "
    "as an rpath to the OpenMP threading layer of Numba with install_name_tool so that Numba can find libomp.dylib, "
    "as explained in the installation instructions of PTtools, "
    "and on Linux the GNU OpenMP library, e.g. with \"apt install libgomp1\" or \"dnf install libgomp\". "
    "You can check the available threading layers in the section \"Threading Layer Information\" "
    "of the output of \"numba --sysinfo\"."
)


#: The Numba threading layer as predicted by :func:`pttools.speedup.threading.predict_numba_threading_layer`
#: when PTtools was imported.
NUMBA_THREADING_LAYER: tp.Final[str | None] = None if NUMBA_DISABLE_JIT else predict_numba_threading_layer()
#: Vendor of the OpenMP library if the Numba threading layer is OpenMP
NUMBA_OPENMP_VENDOR: tp.Final[str | None] = numba_openmp_vendor() if NUMBA_THREADING_LAYER == "omp" else None
CONCURRENT_PARALLELISM: tp.Final[bool] = NUMBA_DISABLE_JIT or NUMBA_THREADING_LAYER in CONCURRENT_THREADING_LAYERS
"""Whether Numba-compiled functions with ``parallel=True`` can be run concurrently from multiple threads.

This requires a threading layer that supports it, i.e. TBB or OpenMP (see :data:`CONCURRENT_THREADING_LAYERS`).
When Numba JIT compilation is disabled, there are no Numba threads, and this is therefore always true.
This is required by :func:`pttools.analysis.parallel.create_spectra` to enable the Numba parallelism of the spectra
when they are computed in threads, together with :data:`SHARED_THREAD_POOL`.
"""
SHARED_THREAD_POOL: tp.Final[bool] = NUMBA_THREADING_LAYER == "tbb"
"""Whether Numba-compiled functions with ``parallel=True`` that are run from multiple threads share a thread pool.

With TBB, the parallel regions started from all threads are run by the same worker threads,
and therefore the number of threads stays close to the number of CPU cores.
With OpenMP, each thread that starts a parallel region gets a team of worker threads of its own,
and therefore running parallel regions from multiple threads simultaneously can oversubscribe the CPU cores.
The ``workqueue`` layer does not support running parallel regions from multiple threads simultaneously at all.
When Numba JIT compilation is disabled, there are no Numba threads, and this is false.
This is required by :func:`pttools.analysis.parallel.create_spectra` to enable the Numba parallelism of the spectra
when they are computed in threads, together with :data:`CONCURRENT_PARALLELISM`.
"""

if not CONCURRENT_PARALLELISM:
    logger.warning(
        "The Numba threading layer is %s, which does not support concurrent parallelism, "
        "i.e. running Numba functions with parallel=True from multiple threads simultaneously. "
        "Therefore the Numba parallelism of the spectra is disabled when they are computed in threads. %s",
        "unknown" if NUMBA_THREADING_LAYER is None else NUMBA_THREADING_LAYER,
        THREADING_LAYER_INSTRUCTIONS
    )

# Numba considers only GNU OpenMP to be unsafe for forking.
# Other OpenMP libraries, such as LLVM and Intel OpenMP, are used on other operating systems than Linux.
if NUMBA_OPENMP_VENDOR == "GNU" and system.FORKING:
    logger.warning(
        "The Numba threading layer is GNU OpenMP, and the multiprocessing start method is \"fork\". "
        "If this process runs Numba functions with parallel=True before starting worker processes, "
        "the workers terminate when they run such functions, and the process pool fails with BrokenProcessPool. "
        "To avoid this, use the \"forkserver\" or \"spawn\" start method "
        "(see multiprocessing.set_start_method()), or install TBB with the \"performance\" optional dependencies "
        "of PTtools, e.g. with pip install \"pttools-gw[performance]\" or uv sync --extra performance."
    )
