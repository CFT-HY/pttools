"""Utilities for the speedups"""

import collections
import functools
import threading
from typing import Callable


def threadsafe_lru[T: Callable](func: T) -> T:
    """
    Thread-safe LRU cache

    From https://noamkremen.github.io/a-simple-threadsafe-caching-decorator.html
    """
    func = functools.lru_cache()(func)
    lock_dict = collections.defaultdict(threading.Lock)

    def _thread_lru(*args, **kwargs):
        # pylint: disable=protected-access
        key = functools._make_key(args, kwargs, typed=True)
        with lock_dict[key]:
            return func(*args, **kwargs)

    return _thread_lru
