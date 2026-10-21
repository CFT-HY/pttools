"""Caching of objects by their constructor arguments.

Creating models, bubbles and spectra can be slow, as it involves solving equations and compiling functions.
The classes that inherit from :py:class:`Cacheable`, such as
:py:class:`pttools.models.base.BaseModel`, :py:class:`pttools.bubble.bubble.BaseBubble`
and :py:class:`pttools.ssm.spectrum.SSMSpectrum`,
can therefore be created with :py:meth:`Cacheable.cached`,
which returns an existing object if one has already been created with the same arguments.

>>> from pttools.models import BagModel
>>> model = BagModel.cached(a_s=1.1, a_b=1, V_s=1)
>>> model is BagModel.cached(a_s=1.1, a_b=1, V_s=1)
True
>>> model is BagModel(a_s=1.1, a_b=1, V_s=1)
False

The arguments are compared by value, except for objects that do not define equality,
such as models and bubbles, which are compared by identity.
Therefore, the bubbles of a cached model are also found from the cache,
but not the bubbles of a model that has been created without the cache.
NumPy arrays are compared by their data type, shape and contents.

As the cached objects are shared, they should not be modified.
For example, a bubble created with ``solve=False`` may already have been solved by another user of the cache.
If you need an object of your own, create it directly with the constructor.

The objects are stored in :py:data:`CACHE`, which is separate for each process.
It holds at most :py:attr:`ObjectCache.max_size` objects,
and when it's full, the least recently used objects are removed from it.
It can be emptied with :py:meth:`ObjectCache.clear` to free memory.
"""

from collections import OrderedDict
from collections.abc import Callable, Hashable
import enum
from fractions import Fraction
import functools
import inspect
import logging
import threading
import typing as tp

import numpy as np

logger: logging.Logger = logging.getLogger(__name__)

__all__ = [
    "CACHE",
    "DEFAULT_CACHE_MAX_SIZE",
    "Cacheable",
    "ObjectCache",
    "cache_key",
]

#: Default maximum number of objects in the cache.
#: A spectrum with the default settings takes about 0.6 MB, and a bubble about 13 kB.
DEFAULT_CACHE_MAX_SIZE: int = 256


class _Nan:
    """Placeholder for NaN in the cache keys, as NaN is not equal to itself."""

    def __repr__(self) -> str:
        return "nan"


_NAN: _Nan = _Nan()


class Cacheable:
    """Mixin for classes whose objects can be cached with :py:meth:`cached`.

    See :py:mod:`pttools.utils.cache` for details.
    """

    @classmethod
    def cached[**P, T](cls: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
        """Get an object with the given constructor arguments from the cache, creating it if necessary.

        The arguments are the same as those of the constructor of the class.
        If the arguments cannot be used as a cache key, e.g. because they contain unhashable objects,
        a new object is created without caching it.

        :return: the cached object, or a new object if there was none
        """
        return CACHE.get(cls, args, kwargs)


class ObjectCache:
    """A thread-safe least recently used cache of objects by their class and constructor arguments.

    The objects of :py:class:`Cacheable` classes are stored in the global instance :py:data:`CACHE`.
    """

    def __init__(self, max_size: int = DEFAULT_CACHE_MAX_SIZE):
        """Create an empty cache.

        :param max_size: the maximum number of objects in the cache. Zero disables the caching.
        :raises ValueError: if the maximum size is negative
        """
        self._cache: OrderedDict[Hashable, tp.Any] = OrderedDict()
        self._lock: threading.Lock = threading.Lock()
        self._max_size: int = 0
        self.hits: int = 0
        self.misses: int = 0
        self.max_size = max_size

    @property
    def max_size(self) -> int:
        """The maximum number of objects in the cache. Zero disables the caching.

        When this is reduced, the least recently used objects are removed until the cache fits in it.
        """
        return self._max_size

    @max_size.setter
    def max_size(self, max_size: int) -> None:
        if max_size < 0:
            raise ValueError(f"The maximum size of the cache must not be negative. Got: {max_size}")
        with self._lock:
            self._max_size = max_size
            self._evict()

    def clear(self) -> None:
        """Remove all the objects from the cache and reset its statistics."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def get[T](self, cls: Callable[..., T], args: tuple[tp.Any, ...], kwargs: dict[str, tp.Any]) -> T:
        """Get an object with the given constructor arguments from the cache, creating it if necessary.

        :param cls: the class
        :param args: the positional arguments of the constructor
        :param kwargs: the keyword arguments of the constructor
        :return: the cached object, or a new object if there was none
        """
        try:
            key = cache_key(cls, args, kwargs)
        except TypeError as e:
            logger.debug("Cannot cache the object of %s, as the arguments cannot be used as a key: %s", cls, e)
            return cls(*args, **kwargs)

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self.hits += 1
                return self._cache[key]
            self.misses += 1

        # The object is created outside the lock so that slow constructors don't block other threads.
        obj = cls(*args, **kwargs)

        with self._lock:
            # If another thread has created the same object simultaneously, use its object.
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            if self._max_size > 0:
                self._cache[key] = obj
                self._evict()
        return obj

    def info(self) -> dict[str, int]:
        """Get the statistics of the cache.

        :return: a dictionary with the number of cache ``hits`` and ``misses``,
            the current ``size`` of the cache and its ``max_size``
        """
        with self._lock:
            return {"hits": self.hits, "misses": self.misses, "size": len(self._cache), "max_size": self._max_size}

    def _evict(self) -> None:
        """Remove the least recently used objects until the cache fits its maximum size. Requires the lock."""
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


#: The global cache of the objects of :py:class:`Cacheable` classes. This is separate for each process.
CACHE: ObjectCache = ObjectCache()


def cache_key(cls: Callable[..., tp.Any], args: tuple[tp.Any, ...], kwargs: dict[str, tp.Any]) -> Hashable:
    """Create the cache key for the given class and constructor arguments.

    The arguments are bound to the signature of the constructor, and the default values are filled in,
    so that the same object is found regardless of whether the arguments are given positionally,
    as keywords or left at their default values.

    :param cls: the class
    :param args: the positional arguments of the constructor
    :param kwargs: the keyword arguments of the constructor
    :return: the cache key
    :raises TypeError: if the arguments do not match the signature or cannot be converted to a hashable key
    """
    bound = _signature(cls).bind(*args, **kwargs)
    bound.apply_defaults()
    return cls, tuple((name, _freeze(value)) for name, value in bound.arguments.items())


def _freeze(value: object) -> Hashable:  # noqa: PLR0911
    """Convert a value to a hashable form that is equal for equal values.

    :raises TypeError: if the value cannot be converted
    """
    if value is None or isinstance(value, (bool, str, bytes, enum.Enum, Fraction)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _NAN if np.isnan(value) else float(value)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return "ndarray", value.shape, tuple(_freeze(item) for item in value.flat)
        return "ndarray", value.dtype.str, value.shape, value.tobytes()
    if isinstance(value, (tuple, list)):
        return type(value).__name__, tuple(_freeze(item) for item in value)
    if isinstance(value, dict):
        return "dict", frozenset((key, _freeze(item)) for key, item in value.items())
    if isinstance(value, (set, frozenset)):
        return "set", frozenset(_freeze(item) for item in value)
    # Objects that don't define equality, such as models, are compared by identity.
    hash(value)
    return value


@functools.cache
def _signature(cls: Callable[..., tp.Any]) -> inspect.Signature:
    """Get the signature of the constructor of the class."""
    return inspect.signature(cls)
