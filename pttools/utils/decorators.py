"""Generic decorators."""

from collections.abc import Callable
import functools
import inspect
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

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
        """Call the function."""


if tp.TYPE_CHECKING:
    # Type checkers such as Pyrefly recognise functools.cached_property as a descriptor by its name,
    # but not its subclasses. The subclass below differs from it only by the docstring inheritance.
    cached_property = functools.cached_property
else:
    class cached_property[T](functools.cached_property[T]):  # noqa: N801
        """A :py:class:`functools.cached_property` that inherits the docstring of the attribute it overrides.

        If the decorated function has no docstring, the docstring of the property or method of the same name
        in the base classes of the owner class is used, e.g. that of an abstract property of a mixin.
        This way the docstring has to be written only once, and it's available for :py:func:`inspect.getdoc`
        and Sphinx autodoc, which do not inherit the docstrings of :py:class:`functools.cached_property`.
        """

        def __set_name__(self, owner: type, name: str) -> None:
            """Set the name of the attribute, and inherit the docstring from the base classes if it's missing."""
            super().__set_name__(owner, name)
            if self.__doc__:
                return
            for base in owner.__mro__[1:]:
                attr = base.__dict__.get(name)
                is_documentable = isinstance(attr, (property, functools.cached_property)) or inspect.isroutine(attr)
                if is_documentable and attr.__doc__:
                    self.__doc__ = attr.__doc__
                    return


class TypePreservingDecorator(tp.Protocol):
    """A decorator that returns the decorated test function or class with its type unchanged."""

    def __call__[F: tp.Callable[..., tp.Any]](self, func: F, /) -> F:
        """Decorate the test function or class, returning it with its type unchanged."""


def conditional_decorator[T: Callable](dec: T, condition: bool, **kwargs: tp.Any) -> T:
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


def for_all_methods[C: type](decorator: Callable[[tp.Any], tp.Any]) -> Callable[[C], C]:
    """Apply a decorator to all methods of a class.

    From: https://stackoverflow.com/a/6307868
    """
    def decorate(cls: C) -> C:
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
