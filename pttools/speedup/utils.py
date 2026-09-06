"""Utilities for the speedups."""

import collections
from collections.abc import Callable
import functools
import threading
import typing as tp


def threadsafe_lru[T: Callable](func: T) -> T:
    """
    Thread-safe LRU cache.

    From https://noamkremen.github.io/a-simple-threadsafe-caching-decorator.html
    """
    cached_func = functools.lru_cache()(func)
    lock_dict: collections.defaultdict[tp.Any, threading.Lock] = collections.defaultdict(threading.Lock)

    def _thread_lru(*args, **kwargs):
        key = functools._make_key(args, kwargs, typed=True)  # noqa: SLF001
        with lock_dict[key]:
            return cached_func(*args, **kwargs)

    return tp.cast(T, _thread_lru)
