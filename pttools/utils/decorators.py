import functools
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
    return decorator
