"""Options for JIT-compilation and other speedups"""

import logging
import os
import typing as tp

logger = logging.getLogger(__name__)

NUMBA_DISABLE_JIT = os.getenv("NUMBA_DISABLE_JIT", False)
NUMBA_INTEGRATE = os.getenv("NUMBA_INTEGRATE", False)
NUMBA_OPTS: tp.Dict[str, any] = {
    # Caching does not work properly with functions that have dependencies across files
    # "cache": True
}

if NUMBA_INTEGRATE:
    if NUMBA_DISABLE_JIT:
        raise RuntimeError("Numba integration cannot be enabled when Numba is disabled")
    logger.warning("Numba-jitted integration has been globally enabled. The results may not be as accurate.")
