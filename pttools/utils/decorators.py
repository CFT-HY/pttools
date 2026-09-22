"""Generic decorators."""

from collections.abc import Callable
import functools
import typing as tp


class PostFunc[**P, T](tp.Protocol):
    """A post-processing function that reports its return type and its value on failure.

    These are used by the parallel processing of :func:`pttools.analysis.parallel.create_bubbles`,
    which needs the return type for allocating the output arrays,
    and the failure value for the bubbles that could not be solved.
    """

    #: Type of the return value, or a tuple of types if the function returns multiple values
    return_type: type | tuple[type, ...]
    #: Value that is returned when the computation fails,
    #: or a tuple of values if the function returns multiple values
    fail_value: tp.Any

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...


def conditional_decorator[T: Callable](dec: T, condition: bool, **kwargs) -> T:
    """Applies the given decorator if the given condition is True.

    :param dec: decorator
    :param condition: whether the decorator should be applied
    """
    def decorator[T2: Callable](func: T2) -> T2:
        if condition:
            if kwargs:
                return tp.cast(T2, functools.wraps(func)(dec(**kwargs)(func)))
            return tp.cast(T2, functools.wraps(func)(dec(func)))
        return func
    return tp.cast(T, decorator)


def for_all_methods(decorator):
    """Apply a decorator to all methods of a class
    https://stackoverflow.com/a/6307868.
    """
    def decorate(cls):
        for attr in cls.__dict__:  # there's probably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


def post_func[**P, T](
        return_type: type | tuple[type, ...],
        fail_value: tp.Any) -> Callable[[Callable[P, T]], PostFunc[P, T]]:
    """Mark a function as a :class:`PostFunc` by attaching the given attributes to it.

    :param return_type: type of the return value, or a tuple of types for multiple return values
    :param fail_value: value to be used when the computation fails
    """
    def decorator(func: Callable[P, T]) -> PostFunc[P, T]:
        # Attributes cannot be attached to a plain function in the type system.
        # pyrefly: ignore[missing-attribute]
        func.return_type = return_type
        # pyrefly: ignore[missing-attribute]
        func.fail_value = fail_value
        return tp.cast(PostFunc[P, T], func)
    return decorator
