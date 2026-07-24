"""Utilities for PTtools models"""

import inspect
import logging
import typing as tp

import numpy as np

import pttools.type_hints as th

logger = logging.getLogger(__name__)


def check_value_in_range(
    x: th.FloatOrArr,
    x_min: float,
    x_max: float,
    name: str,
    context: str | None = None,
    x_format: str = ".6e",
    error_on_invalid: bool = True,
    nan_on_invalid: bool = True,
    log_invalid: bool = True) -> th.FloatOrArr:
    r"""Check that $x \in ({x}_\text{min}, {x}_\text{max})$ for the given $x$."""
    if x_min > x_max:
        raise ValueError(
            f"Invalid limits for range check: {name}_min={x_min:{x_format}} > {name}_max={x_max:{x_format}}."
        )

    is_scalar = np.isscalar(x)

    # None and nan should be logged, but not raise an exception.
    if x is None or np.any(np.isnan(x)):
        if log_invalid:
            logger.error("Got nan for %s in %s", name, inspect.stack()[1][3])
        # Scalar None cannot be tested for negativity.
        if x is None or is_scalar:
            return np.nan

    too_smalls = x < x_min
    too_larges = x > x_max
    too_small = np.any(too_smalls)
    too_large = np.any(too_larges)

    # Shortcut for speed
    if not (too_small or too_large):
        return x

    info = None
    context_str = "" if context is None else f" for {context}"
    if is_scalar:
        if too_small:
            info = f"Got {name}={x:{x_format}} < {name}_min={x_min:{x_format}}{context_str}."
        elif too_large:
            info = f"Got {name}={x:{x_format}} > w_max={x_max:{x_format}}{context_str}."
    else:
        if too_small and too_large:
            info = \
                f"Got {np.sum(too_smalls)} point(s) with {name} < {name}_min={x_min:{x_format}} " \
                f"and {np.sum(too_larges)} point(s) with {name} > {name}_max{context_str}. " \
                f"Most problematic values: {name}={np.min(x):{x_format}}, {name}={np.max(x):{x_format}}"
        elif too_small:
            info = \
                f"Got {np.sum(too_smalls)} point(s) with {name} < {name}_min={x_min:{x_format}}{context_str}. " \
                f"Most problematic value: {name}={np.min(x):{x_format}}."
        elif too_large:
            info = \
                f"Got {np.sum(too_larges)} point(s) with {name} > {name}_max={x_max:{x_format}}{context_str}. " \
                f"Most problematic value: {name}={np.max(x):{x_format}}."

    if log_invalid:
        logger.error(info)
    if error_on_invalid:
        raise ValueError(info)

    if nan_on_invalid and info is not None:
        if is_scalar:
            return np.nan
        x = x.copy()
        if too_small:
            x[too_small] = np.nan
        if too_large:
            x[too_large] = np.nan
    return x


def ensure_float(value: tp.Any, name: str, allow_none: bool = False) -> float:
    """Ensure that the given value is a float, and convert if necessary"""
    ensure_scalar(value, name, allow_none)
    return ensure_type(value, float, allow_none)


def ensure_floats(values: dict[str, tp.Any], allow_none: bool = False) -> list[float]:
    """Ensure that the given values are floats, and convert if necessary"""
    return [ensure_float(value, name, allow_none=allow_none) for name, value in values.items()]


def ensure_scalar(value: tp.Any, name: str, allow_none: bool = False) -> None:
    """Ensure that the given value is a scalar

    Some functions such as :py:func:`np.vectorize` tend to give 0D arrays, which may cause subtle errors later on.
    """
    if not ((value is None and allow_none) or np.isscalar(value)):
        raise ValueError(f"{name} should be a scalar. Did you give e.g. a 0D array instead? Got: {name}={value}")


def ensure_type[T](value: tp.Any, cls: tp.Type[T], allow_none: bool = False) -> T:
    """Ensure that the given value is of the given type, and convert if necessary"""
    return value if (value is None and allow_none) or isinstance(value, cls) else cls(value)
