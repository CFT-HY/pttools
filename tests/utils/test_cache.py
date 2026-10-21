"""Unit tests for the caching of objects by their constructor arguments."""

import threading
import typing as tp
import unittest

import numpy as np
import pytest

from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.utils.cache import CACHE, Cacheable, ObjectCache, cache_key


class Obj(Cacheable):
    """A cacheable class for testing."""

    def __init__(self, x: float, y: tp.Any = None, *args: tp.Any, z: str = "z", **kwargs: tp.Any) -> None:
        """Store the arguments."""
        self.x: float = x
        self.y: tp.Any = y
        self.args: tuple[tp.Any, ...] = args
        self.z: str = z
        self.kwargs: dict[str, tp.Any] = kwargs


class SubObj(Obj):
    """A subclass of the cacheable test class."""


class UnhashableArg:
    """An object that cannot be used in a cache key."""

    __hash__ = None  # type: ignore[assignment]


class CacheTest(unittest.TestCase):
    """Unit tests for the caching of objects by their constructor arguments."""

    def test_args_as_keywords(self) -> None:
        """The same object should be found regardless of how the arguments are given."""
        obj = Obj.cached(1.5, 2)
        assert Obj.cached(x=1.5, y=2) is obj
        assert Obj.cached(1.5, y=2, z="z") is obj
        assert Obj.cached(np.float64(1.5), np.int64(2)) is obj

    def test_arrays(self) -> None:
        """Arrays should be compared by their contents."""
        obj = Obj.cached(1.6, np.array([1., np.nan, 3.]))
        assert Obj.cached(1.6, np.array([1., np.nan, 3.])) is obj
        assert Obj.cached(1.6, np.array([1., 2., 3.])) is not obj
        assert Obj.cached(1.6, np.array([1, 2, 3])) is not Obj.cached(1.6, np.array([1., 2., 3.]))

    def test_clear(self) -> None:
        """Clearing the cache should remove the objects and reset the statistics."""
        cache = ObjectCache()
        obj = cache.get(Obj, (2.6,), {})
        cache.get(Obj, (2.6,), {})
        cache.clear()
        assert cache.info() == {"hits": 0, "misses": 0, "size": 0, "max_size": cache.max_size}
        assert cache.get(Obj, (2.6,), {}) is not obj

    def test_different_args(self) -> None:
        """Different arguments should give different objects."""
        obj = Obj.cached(1.7, 2)
        assert Obj.cached(1.7, 3) is not obj
        assert Obj.cached(1.7, 2, 3) is not obj
        assert Obj.cached(1.7, 2, z="a") is not obj
        assert Obj.cached(1.7, 2, w=1) is not obj

    def test_identity(self) -> None:
        """Objects that don't define equality should be compared by identity."""
        model1 = BagModel.cached(a_s=1.1, a_b=1, V_s=1)
        model2 = BagModel(a_s=1.1, a_b=1, V_s=1)
        assert BagModel.cached(a_s=1.1, a_b=1, V_s=1) is model1
        assert model2 is not model1
        bubble1 = Bubble.cached(model1, v_wall=0.5, alpha_n=0.1)
        assert Bubble.cached(model1, 0.5, 0.1) is bubble1
        assert Bubble.cached(model2, v_wall=0.5, alpha_n=0.1) is not bubble1

    def test_info(self) -> None:
        """The statistics of the cache should be updated."""
        info1 = CACHE.info()
        Obj.cached(1.8)
        Obj.cached(1.8)
        info2 = CACHE.info()
        assert info2["misses"] == info1["misses"] + 1
        assert info2["hits"] == info1["hits"] + 1

    def test_key_errors(self) -> None:
        """Invalid arguments should raise a TypeError."""
        with pytest.raises(TypeError):
            cache_key(Obj, (), {})
        with pytest.raises(TypeError):
            cache_key(Obj, (1., UnhashableArg()), {})

    def test_max_size(self) -> None:
        """The least recently used objects should be removed when the cache is full."""
        # A separate cache is used to avoid removing the objects of the other tests from the global cache.
        cache = ObjectCache(max_size=2)
        obj1 = cache.get(Obj, (1.9,), {})
        obj2 = cache.get(Obj, (2.0,), {})
        assert cache.get(Obj, (1.9,), {}) is obj1
        cache.get(Obj, (2.1,), {})
        assert cache.info() == {"hits": 1, "misses": 3, "size": 2, "max_size": 2}
        assert cache.get(Obj, (1.9,), {}) is obj1
        assert cache.get(Obj, (2.0,), {}) is not obj2
        cache.max_size = 0
        assert cache.info()["size"] == 0
        assert cache.get(Obj, (1.9,), {}) is not cache.get(Obj, (1.9,), {})
        with pytest.raises(ValueError, match="must not be negative"):
            cache.max_size = -1
        with pytest.raises(ValueError, match="must not be negative"):
            ObjectCache(max_size=-1)

    def test_nan(self) -> None:
        """NaN arguments should be considered equal."""
        assert Obj.cached(np.nan) is Obj.cached(float("nan"))

    def test_nested(self) -> None:
        """Lists, tuples, dicts and sets should be compared by their contents."""
        obj = Obj.cached(2.2, [1, (2, 3)], data={"a": [1, 2], "b": {3}})
        assert Obj.cached(2.2, [1, (2, 3)], data={"b": {3}, "a": [1, 2]}) is obj
        assert Obj.cached(2.2, (1, (2, 3)), data={"a": [1, 2], "b": {3}}) is not obj

    def test_subclass(self) -> None:
        """Subclasses should have their own objects."""
        obj = Obj.cached(2.3)
        sub_obj = SubObj.cached(2.3)
        assert isinstance(sub_obj, SubObj)
        assert sub_obj is not obj
        assert SubObj.cached(2.3) is sub_obj

    def test_threads(self) -> None:
        """Simultaneous creation of the same object in multiple threads should give the same object."""
        results: list[Obj] = []
        lock = threading.Lock()

        def create() -> None:
            obj = Obj.cached(2.4)
            with lock:
                results.append(obj)

        threads = [threading.Thread(target=create) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(results) == len(threads)
        assert all(obj is results[0] for obj in results)

    def test_unhashable(self) -> None:
        """Objects with unhashable arguments should be created without caching."""
        arg = UnhashableArg()
        obj = Obj.cached(2.5, arg)
        assert obj.y is arg
        assert Obj.cached(2.5, arg) is not obj
