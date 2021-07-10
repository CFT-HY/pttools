"""
Custom decorators for JIT-compilation with Numba

Warning: complex decorators calling Numba may cause segmentation faults when profiled.
https://github.com/numba/numba/issues/3625
"""

import inspect
import logging

import numba
import numpy as np

from . import options
from . import utils

logger = logging.getLogger(__name__)


def generated_jit(func: callable):
    if options.NUMBA_DISABLE_JIT:
        return func
    return numba.generated_jit(func, nopython=True, **options.NUMBA_OPTS)


def njit(func: callable = None, **kwargs):
    """Wrapper for numba.njit.

    May cause segmentation faults with profilers.
    """
    # print(func, kwargs)

    def _njit(func2):
        # print("Jitting", func2)
        return numba.njit(func2, **options.NUMBA_OPTS, **kwargs)
    if func is None:
        return _njit
    return _njit(func)


def njit_if_numba_integrate(func: callable = None, **kwargs) -> callable:
    if func:
        return utils.conditional_decorator(numba.njit, options.NUMBA_INTEGRATE, **kwargs)(func)
    return utils.conditional_decorator(numba.njit, options.NUMBA_INTEGRATE, **kwargs)


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
            module.__dict__[name] = numba.njit(obj, **options.NUMBA_OPTS, **kwargs)


def vectorize(**kwargs):
    """Extended version of numba.vectorize with support for NUMBA_DISABLE_JIT"""
    def vectorize_inner(func):
        if options.NUMBA_DISABLE_JIT:
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
        return numba.vectorize(**kwargs)(func)
    return vectorize_inner
