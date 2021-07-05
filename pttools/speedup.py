"""Utilities for speeding up the simulations with Numba

Apparently complex decorators calling Numba may cause segmentation faults when profiled.
https://github.com/numba/numba/issues/3625
"""

import collections
import functools
import inspect
import logging
import threading
import typing as tp

import numba
import numpy as np

logger = logging.getLogger(__name__)

ENABLED: bool = True

NUMBA_OPTS: tp.Dict[str, any] = {
    # Caching does not work properly with functions that have dependencies across files
    # "cache": True
}


@numba.njit(parallel=True)
def logspace(start: float, stop: float, num: int, base: float = 10.0) -> np.ndarray:
    """Numba version of numpy.logspace"""
    y = np.linspace(start, stop, num)
    return base**y


def njit_module(**kwargs):
    """Adapted from numba.jit_module.

    May cause segmentation faults with profilers.
    """
    # Get the module jit_module is being called from
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    # Replace functions in module with jit-wrapped versions
    for name, obj in module.__dict__.items():
        if inspect.isfunction(obj) and inspect.getmodule(obj) == module:
            logger.debug("Auto decorating function {} from module {} with jit "
                          "and options: {}".format(obj, module.__name__, kwargs))
            module.__dict__[name] = numba.njit(obj, **NUMBA_OPTS, **kwargs)


def njit(func: callable = None, **kwargs):
    """Wrapper for numba.njit.

    May cause segmentation faults with profilers.
    """
    # print(func, kwargs)

    def _njit(func2):
        # print("Jitting", func2)
        return numba.njit(func2, **NUMBA_OPTS, **kwargs)
    if func is None:
        return _njit
    return _njit(func)


def generated_jit(func: callable):
    if ENABLED:
        return numba.generated_jit(func, nopython=True, **NUMBA_OPTS)
    return func


def threadsafe_lru(func):
    """From
    https://noamkremen.github.io/a-simple-threadsafe-caching-decorator.html
    """
    func = functools.lru_cache()(func)
    lock_dict = collections.defaultdict(threading.Lock)

    def _thread_lru(*args, **kwargs):
        key = functools._make_key(args, kwargs, typed=True)
        with lock_dict[key]:
            return func(*args, **kwargs)

    return _thread_lru
