"""Utilities for PTtools models."""

import inspect
import logging
import typing as tp

import numpy as np

import pttools.type_hints as th

logger = logging.getLogger(__name__)


def log_nan(x: th.FloatOrArr | None, name: str, caller: str, context_str: str) -> None:
    """Log that the given value is None or contains nan values."""
    if x is None:
        logger.error("Got None for %s in %s%s.", name, caller, context_str)
    elif np.isscalar(x):
        logger.error("Got nan for %s in %s%s.", name, caller, context_str)
    else:
        logger.error(
            "Got nan for %s/%s values of %s in %s%s.",
            np.sum(np.isnan(x)), np.size(x), name, caller, context_str
        )


def out_of_range_info(
        x: th.FloatOrArr,
        x_min: float,
        x_max: float,
        name: str,
        context_str: str,
        x_format: str,
        is_scalar: bool,
        too_smalls: bool | np.bool_ | th.BoolArr,
        too_larges: bool | np.bool_ | th.BoolArr) -> str:
    """Create the error message of :func:`check_value_in_range` for values outside the range."""
    if is_scalar:
        if np.any(too_smalls):
            return f"Got {name}={x:{x_format}} < {name}_min={x_min:{x_format}}{context_str}."
        return f"Got {name}={x:{x_format}} > {name}_max={x_max:{x_format}}{context_str}."
    too_small = np.any(too_smalls)
    too_large = np.any(too_larges)
    if too_small and too_large:
        return \
                f"Got {np.sum(too_smalls)} point(s) with {name} < {name}_min={x_min:{x_format}} " \
                f"and {np.sum(too_larges)} point(s) with {name} > {name}_max{context_str}. " \
                f"Most problematic values: {name}={np.min(x):{x_format}}, {name}={np.max(x):{x_format}}"
    if too_small:
        return \
                f"Got {np.sum(too_smalls)} point(s) with {name} < {name}_min={x_min:{x_format}}{context_str}. " \
                f"Most problematic value: {name}={np.min(x):{x_format}}."
    return \
            f"Got {np.sum(too_larges)} point(s) with {name} > {name}_max={x_max:{x_format}}{context_str}. " \
            f"Most problematic value: {name}={np.max(x):{x_format}}."


def check_value_in_range[T: th.FloatOrArr](
    x: T,
    x_min: float,
    x_max: float,
    name: str,
    context: str | None = None,
    x_format: str = ".6e",
    error_on_invalid: bool = True,
    nan_on_invalid: bool = True,
    log_invalid: bool = True) -> T:
    r"""Check that $x \in ({x}_\text{min}, {x}_\text{max})$ for the given $x$.

    :return: $x$, where the invalid values have been replaced with nan if ``nan_on_invalid`` is set
    """
    if x_min > x_max:
        raise ValueError(
            f"Invalid limits for range check: {name}_min={x_min:{x_format}} > {name}_max={x_max:{x_format}}."
        )

    is_scalar = np.isscalar(x)
    context_str = "" if context is None else f" for {context}"

    # None and nan should be logged, but not raise an exception.
    if x is None or np.any(np.isnan(x)):
        if log_invalid:
            log_nan(x, name=name, caller=inspect.stack()[1][3], context_str=context_str)
        # Scalar None cannot be tested for negativity.
        if x is None or is_scalar:
            return tp.cast(T, np.nan)

    too_smalls = x < x_min
    too_larges = x > x_max
    too_small = np.any(too_smalls)
    too_large = np.any(too_larges)

    # Shortcut for speed
    if not (too_small or too_large):
        return x

    info = out_of_range_info(
        x, x_min, x_max, name, context_str, x_format,
        is_scalar=is_scalar, too_smalls=too_smalls, too_larges=too_larges
    )

    if log_invalid:
        logger.error(info)
    if error_on_invalid:
        raise ValueError(info)

    if nan_on_invalid:
        if is_scalar:
            return tp.cast(T, np.nan)
        # np.isscalar() does not narrow the type for the type checker.
        x_arr = tp.cast(th.FloatArr, x).copy()
        if too_small:
            x_arr[too_smalls] = np.nan
        if too_large:
            x_arr[too_larges] = np.nan
        return tp.cast(T, x_arr)
    return x


def ensure_float(value: tp.Any, name: str, allow_none: bool = False) -> float:
    """Ensure that the given value is a float, and convert if necessary."""
    ensure_scalar(value, name, allow_none)
    return ensure_type(value, float, allow_none)


def ensure_floats(values: dict[str, tp.Any], allow_none: bool = False) -> list[float]:
    """Ensure that the given values are floats, and convert if necessary."""
    return [ensure_float(value, name, allow_none=allow_none) for name, value in values.items()]


def ensure_scalar(value: tp.Any, name: str, allow_none: bool = False) -> None:
    """Ensure that the given value is a scalar.

    Some functions such as :py:func:`np.vectorize` tend to give 0D arrays, which may cause subtle errors later on.
    """
    if not ((value is None and allow_none) or np.isscalar(value)):
        raise ValueError(f"{name} should be a scalar. Did you give e.g. a 0D array instead? Got: {name}={value}")


def ensure_type[T](value: tp.Any, cls: type[T], allow_none: bool = False) -> T:
    """Ensure that the given value is of the given type, and convert if necessary."""
    # The unbounded type variable resolves to object, whose constructor takes no arguments.
    # pyrefly: ignore[bad-argument-count]
    return tp.cast(T, value if (value is None and allow_none) or isinstance(value, cls) else cls(value))
