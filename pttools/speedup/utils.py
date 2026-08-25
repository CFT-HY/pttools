"""Utilities for the speedups"""

import collections
import functools
import threading
import typing as tp
from typing import Callable


def threadsafe_lru[T: Callable](func: T) -> T:
    """
    Thread-safe LRU cache

    From https://noamkremen.github.io/a-simple-threadsafe-caching-decorator.html
    """
    cached_func = functools.lru_cache()(func)
    lock_dict: collections.defaultdict[tp.Any, threading.Lock] = collections.defaultdict(threading.Lock)

    def _thread_lru(*args, **kwargs):
        # pylint: disable=protected-access
        key = functools._make_key(args, kwargs, typed=True)
        with lock_dict[key]:
            return cached_func(*args, **kwargs)

    return tp.cast(T, _thread_lru)
