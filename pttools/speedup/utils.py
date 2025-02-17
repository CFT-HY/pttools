"""Utilities for the speedups"""

import collections
import functools
import threading

import numpy as np


def conditional_decorator(dec: callable, condition: bool, **kwargs) -> callable:
    """Applies the given decorator if the given condition is True.

    :param dec: decorator
    :param condition: whether the decorator should be applied
    """
    def decorator(func: callable) -> callable:
        if condition:
            if kwargs:
                return functools.wraps(func)(dec(**kwargs)(func))
            return functools.wraps(func)(dec(func))
        return func
    return decorator


def copy_doc(copy_func: callable) -> callable:
    """Copies the doc string of the given function to another.
    This function is intended to be used as a decorator.
    From: https://stackoverflow.com/a/68901244

    .. code-block:: python3

        def foo():
            '''This is a foo doc string'''
            ...

        @copy_doc(foo)
        def bar():
            ...
    """

    def wrapped(func: callable) -> callable:
        func.__doc__ = copy_func.__doc__
        return func

    return wrapped


def is_nan_or_none(value: float = None) -> bool:
    return value is None or np.isnan(value)


def threadsafe_lru(func: callable) -> callable:
    """From
    https://noamkremen.github.io/a-simple-threadsafe-caching-decorator.html
    """
    func = functools.lru_cache()(func)
    lock_dict = collections.defaultdict(threading.Lock)

    def _thread_lru(*args, **kwargs):
        # pylint: disable=protected-access
        key = functools._make_key(args, kwargs, typed=True)
        with lock_dict[key]:
            return func(*args, **kwargs)

    return _thread_lru
