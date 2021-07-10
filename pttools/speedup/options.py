"""Options for JIT-compilation and other speedups"""

import os
import typing as tp

NUMBA_DISABLE_JIT = os.getenv("NUMBA_DISABLE_JIT", False)
NUMBA_INTEGRATE = False
NUMBA_OPTS: tp.Dict[str, any] = {
    # Caching does not work properly with functions that have dependencies across files
    # "cache": True
}
