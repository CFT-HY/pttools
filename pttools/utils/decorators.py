import functools
import typing as tp
from typing import Callable


def conditional_decorator[T: Callable](dec: T, condition: bool, **kwargs) -> T:
    """Applies the given decorator if the given condition is True.

    :param dec: decorator
    :param condition: whether the decorator should be applied
    """
    def decorator[T2: Callable](func: T2) -> T2:
        if condition:
            if kwargs:
                return functools.wraps(func)(dec(**kwargs)(func))
            return functools.wraps(func)(dec(func))
        return func
    return tp.cast(T, decorator)


def for_all_methods(decorator):
    """Apply a decorator to all methods of a class
    https://stackoverflow.com/a/6307868
    """
    def decorate(cls):
        for attr in cls.__dict__:  # there's probably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate
