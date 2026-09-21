"""Utilities for deprecating old code."""

import functools
import typing as tp
import warnings


def _deprecated_alias[**P, R](
        new: tp.Callable[P, R],
        old_name: str,
        param_map: dict[str, str] | None = None) -> tp.Callable[P, R]:
    """Create a deprecated alias for a renamed function.

    :param new: the new function
    :param old_name: the old name of the function
    :param param_map: mapping from old parameter names to new parameter names
    """
    @functools.wraps(new)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        warnings.warn(
            f"{old_name} is deprecated and will be removed in a future version. "
            f"Use {new.__name__} instead.",
            DeprecationWarning,
            stacklevel=2
        )
        if param_map:
            for old_param, new_param in param_map.items():
                if old_param in kwargs:
                    kwargs[new_param] = kwargs.pop(old_param)
        return new(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__qualname__ = old_name
    wrapper.__doc__ = f"Deprecated alias of :py:func:`{new.__module__}.{new.__name__}`."
    return wrapper
