"""Utilities for speeding up the simulations with Numba"""

import typing as tp

import numba

NUMBA_OPTS: tp.Dict[str, any] = {
    "cache": True
}


def njit(func: callable):
    return numba.njit(func, **NUMBA_OPTS)


def generated_jit(func: callable):
    return numba.generated_jit(func, nopython=True, **NUMBA_OPTS)
