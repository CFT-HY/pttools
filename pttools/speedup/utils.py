"""Utilities for the speedups"""

import collections
import functools
import threading


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
