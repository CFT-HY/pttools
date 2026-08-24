"""Options for JIT-compilation and other speedups"""

import logging
import os
import typing as tp

from pttools.utils import system

logger = logging.getLogger(__name__)

#: Default maximum number of parallel worker processes
MAX_WORKERS_DEFAULT: int
if system.AVAILABLE_CPU_CORES is None:
    MAX_WORKERS_DEFAULT = 1
    logger.warning(
        "This platform does not provide info on the number of available CPU cores. Using 1 core. %s",
        system.platform_info()
    )
elif system.IS_WINDOWS and system.AVAILABLE_CPU_CORES > 61:
    # On Windows, the maximum number of worker processes is limited to 61.
    # https://docs.python.org/3/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor
    MAX_WORKERS_DEFAULT = 61
else:
    MAX_WORKERS_DEFAULT = system.AVAILABLE_CPU_CORES


# The choice of the Numba threading layer cannot be printed here, since it's not selected until needed.
if not system.FORKING or not system.CPU_AFFINITY:
    msg = system.platform_info()
    if not system.CPU_AFFINITY:
        msg += (
            " This platform does not provide info on which CPU cores are available for this process. "
            "Using all cores."
        )
    logger.debug(msg)


#: Whether Numba JIT compilation has been disabled.
NUMBA_DISABLE_JIT: tp.Final[bool] = bool(int(os.getenv("NUMBA_DISABLE_JIT", "0")))

NUMBA_ENABLE_CACHE: bool = \
    bool(int(os.getenv("NUMBA_ENABLE_CACHE"))) \
        if "NUMBA_ENABLE_CACHE" in os.environ \
        else (system.IS_GITHUB_ACTIONS or system.IS_PIP_PACKAGE)
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
#: Default options for the custom njit decorator.
NUMBA_OPTS: dict[str, tp.Any] = {
    # Caching does not work properly with functions that have dependencies across files
    # "cache": True
}

if NUMBA_INTEGRATE:
    if NUMBA_DISABLE_JIT:
        raise RuntimeError("Numba integration cannot be enabled when Numba is disabled")
    logger.warning("Numba-jitted integration has been globally enabled. The results may not be as accurate.")
