"""
Custom decorators for JIT-compilation with Numba

Warning: complex decorators calling Numba may cause segmentation faults when profiled.
https://github.com/numba/numba/issues/3625
"""

import functools
import inspect
import logging
import types
import typing as tp

import numba
import numpy as np

from pttools.speedup.options import NUMBA_DISABLE_JIT, NUMBA_OPTS

logger = logging.getLogger(__name__)


def njit(func: tp.Callable | None = None, **kwargs):
    """Wrapper for numba.njit, which applies the default options of :data:`NUMBA_OPTS`.

    Options given as keyword arguments override the defaults.
    For example, ``cache=True`` enables caching regardless of :data:`NUMBA_ENABLE_CACHE`.
    This should be used only for functions that do not call code from other files,
    and which can therefore be cached safely even when developing PTtools.

    May cause segmentation faults with profilers.
    """
    # Creating the dictionary before passing it to the function is necessary for handling duplicate values.
    opts = {**NUMBA_OPTS, **kwargs}
    def _njit(func2):
        return numba.njit(func2, **opts)
    if func is None:
        return _njit
    return _njit(func)


def _renamed_func(func: tp.Callable, suffix: str) -> tp.Callable:
    """Create a copy of the given function with the given suffix appended to its name"""
    renamed = types.FunctionType(
        func.__code__, func.__globals__, f"{func.__name__}{suffix}", func.__defaults__, func.__closure__
    )
    renamed.__qualname__ = f"{func.__qualname__}{suffix}"
    renamed.__module__ = func.__module__
    renamed.__doc__ = func.__doc__
    renamed.__annotations__ = func.__annotations__
    renamed.__kwdefaults__ = func.__kwdefaults__
    return renamed


def njit_parallel_pair(func: tp.Callable, **kwargs) -> tuple[tp.Callable, tp.Callable]:
    """Compile both a parallel and a serial version of the given function.

    The parallel version runs its ``numba.prange`` loops with multiple threads,
    whereas in the serial version ``numba.prange`` behaves like ``range``.
    Having both versions available makes it possible to disable the thread-based parallelism
    of an individual function, e.g. when the caller is already running in parallel.

    Compiling the same function twice with different options is not safe with the Numba cache,
    as the name of the cache file is based on the module, qualified name and first line number
    of the Python function, and the cache key on a hash of its bytecode.
    None of these depend on the compilation options, and therefore the two versions
    would share the same cache entry and silently overwrite each other's compiled code.
    This is avoided by compiling each version from a separately named copy of the function.

    :param func: the function to be compiled
    :param kwargs: additional options for :func:`njit`
    :return: the parallel version and the serial version of the function
    """
    return (
        njit(_renamed_func(func, "_parallel"), parallel=True, **kwargs),
        njit(_renamed_func(func, "_serial"), parallel=False, **kwargs)
    )


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
            logger.debug(
                "Auto decorating function %s from module %s with jit and options: %s",
                obj, module.__name__, kwargs
            )
            module.__dict__[name] = njit(obj, **kwargs)


def vectorize(**kwargs):
    """Extended version of numba.vectorize with support for NUMBA_DISABLE_JIT"""
    def vectorize_inner(func: tp.Callable):
        if NUMBA_DISABLE_JIT:
            # Using functools.wraps() ensures that docstrings etc. are preserved
            @functools.wraps(func)
            def wrapper(*func_args, **func_kwargs):
                # If called with scalars
                if not \
                        next((isinstance(arg, np.ndarray) for arg in func_args), False) or \
                        next((isinstance(arg, np.ndarray) for arg in func_kwargs.values()), False):
                    return func(*func_args, **func_kwargs)
                # If called with 0D arrays
                if not np.all([arg.ndim for arg in func_args] + [arg.ndim for arg in func_kwargs.values()]):
                    return func(
                        *[arg.item() for arg in func_args],
                        **{name: value.item() for name, value in func_kwargs.items()}
                    )
                # If called with arrays
                return np.array([
                    func(*i_args, **{name: value[i] for name, value in func_kwargs.items()})
                    for i, i_args in enumerate(zip(*func_args))
                ])
            return wrapper
        return functools.wraps(func)(numba.vectorize(**kwargs)(func))
    return vectorize_inner
