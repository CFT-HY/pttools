"""Script for testing the Numba caching of functions that call cached parallel functions.

This is run in separate processes by :class:`tests.test_numba_fixes.TestReloadInit`,
since the caching behaviour that is being tested can be observed only between processes.

Usage: ``python tests/numba_reload_init.py callee|caller``,
with ``NUMBA_CACHE_DIR`` set in the environment and with the repository directory in ``PYTHONPATH``.
"""

import sys

import numba
import numpy as np

# Importing the speedup module applies the Numba fixes.
import pttools.speedup  # noqa: F401


@numba.njit(parallel=True, cache=True)
def parallel_sum(x: np.ndarray) -> float:
    """A parallel function, whose machine code refers to the Numba threading layer."""
    s = 0.
    for i in numba.prange(x.size):
        s += x[i]
    return s


@numba.njit(cache=True)
def caller(x: np.ndarray) -> float:
    """A cached function, which gets the machine code of the parallel function linked into its own."""
    return parallel_sum(x) + 1.


def main(mode: str) -> None:
    x = np.ones(1000)
    if mode == "callee":
        result = parallel_sum(x)
        expected = x.size
    elif mode == "caller":
        result = caller(x)
        expected = x.size + 1
    else:
        raise ValueError(f"Unknown mode: {mode}")
    if result != expected:
        raise RuntimeError(f"Got {result}, expected {expected}")


if __name__ == "__main__":
    main(sys.argv[1])
