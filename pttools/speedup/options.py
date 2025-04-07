"""Options for JIT-compilation and other speedups"""

import logging
import multiprocessing
import os
import platform
import typing as tp

logger = logging.getLogger(__name__)

GITHUB_ACTIONS: tp.Final[bool] = "GITHUB_ACTIONS" in os.environ
START_METHOD: str = multiprocessing.get_start_method()
FORKING: bool = START_METHOD == "fork"
UNAME = platform.uname()
CPU_AFFINITY: bool = False
IS_WINDOWS: bool = platform.system() == "Windows"

#: Maximum workers for ProcessPoolExecutor (determined dynamically based on the available CPUs)
try:
    # This is available only on some platforms
    MAX_WORKERS_DEFAULT: int = len(os.sched_getaffinity(0))
    CPU_AFFINITY = True
except AttributeError:
    # multiprocessing.cpu_count() is a wrapper around os.cpu_count()
    # https://stackoverflow.com/a/53537394
    MAX_WORKERS_DEFAULT: int = multiprocessing.cpu_count()

# The choice of the Numba threading layer cannot be printed here, since it's not selected until needed.
if not FORKING or not CPU_AFFINITY:
    msg = "Platform: %s (%s, %s) on %s (%s)."
    if not CPU_AFFINITY:
        msg += " This platform does not provide info on which CPU cores are available for this process. Using all cores."
    if START_METHOD != "fork":
        msg += \
            " This platform does not support forking." \
            " Starting parallel processes will be slower, since the cs2 functions have to be compiled in each sub-process."
    logger.debug(msg, UNAME.system, UNAME.release, START_METHOD, UNAME.processor, UNAME.machine)

#: Whether Numba JIT compilation has been disabled.
NUMBA_DISABLE_JIT: tp.Final[bool] = bool(int(os.getenv("NUMBA_DISABLE_JIT", "0")))
#: Whether to use NumbaLSODA as the default ODE integrator.
NUMBA_INTEGRATE: tp.Final[bool] = bool(int(os.getenv("NUMBA_INTEGRATE", "0")))
#: Whether to use looser tolerances, which are necessary for the unit tests to pass with NumbaLSODA.
NUMBA_INTEGRATE_TOLERANCES: tp.Final[bool] = bool(
    int(os.getenv("NUMBA_INTEGRATE_TOLERANCES", str(int(NUMBA_INTEGRATE))))
)
#: Whether to use nested parallelism. This requires that either TBB or OpenMP is installed and working.
NUMBA_NESTED_PARALLELISM: tp.Final[bool] = bool(int(os.getenv("NUMBA_NESTED_PARALLELISM", "0")))
#: Default options for the custom njit decorator.
NUMBA_OPTS: tp.Dict[str, any] = {
    # Caching does not work properly with functions that have dependencies across files
    # "cache": True
}

if NUMBA_INTEGRATE:
    if NUMBA_DISABLE_JIT:
        raise RuntimeError("Numba integration cannot be enabled when Numba is disabled")
    logger.warning("Numba-jitted integration has been globally enabled. The results may not be as accurate.")
