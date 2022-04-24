"""Options for JIT-compilation and other speedups"""

import logging
import os
import typing as tp

logger = logging.getLogger(__name__)

# TODO: Add typing.Final for these when Python 3.8 becomes the oldest supported version.
# https://www.python.org/dev/peps/pep-0591/

#: Whether Numba JIT compilation has been disabled.
NUMBA_DISABLE_JIT = bool(int(os.getenv("NUMBA_DISABLE_JIT", "0")))
#: Whether to use NumbaLSODA as the default ODE integrator.
NUMBA_INTEGRATE = bool(int(os.getenv("NUMBA_INTEGRATE", "0")))
#: Whether to use looser tolerances, which are necessary for the unit tests to pass with NumbaLSODA.
NUMBA_INTEGRATE_TOLERANCES = bool(int(os.getenv("NUMBA_INTEGRATE_TOLERANCES", str(int(NUMBA_INTEGRATE)))))
#: Whether to use nested parallelism. This requires that either TBB or OpenMP is installed and working.
NUMBA_NESTED_PARALLELISM = bool(int(os.getenv("NUMBA_NESTED_PARALLELISM", "0")))
#: Default options for the custom njit decorator.
NUMBA_OPTS: tp.Dict[str, any] = {
    # Caching does not work properly with functions that have dependencies across files
    # "cache": True
}

if NUMBA_INTEGRATE:
    if NUMBA_DISABLE_JIT:
        raise RuntimeError("Numba integration cannot be enabled when Numba is disabled")
    logger.warning("Numba-jitted integration has been globally enabled. The results may not be as accurate.")
