"""Utility functions and constants for unit testing."""

import inspect
import typing as tp

import numpy as np
import numpy.typing as npt

import pttools.type_hints as th
from pttools.utils.math import rel_diff_arr, rel_diff_scalar
from pttools.utils.printing import DEFAULT_FMT, print_1d, print_2d


def _assert_allclose_header(name: str | None, rtol: float, atol: float, caller: str) -> list[str]:
    """Create the first lines of the error message of :func:`assert_allclose`."""
    return [
        f"assert_allclose failed {f'for {name} ' if name is not None else ''}in {caller}",
        f"Not equal to tolerance rtol={rtol}, atol={atol}"
    ]


def assert_allclose(
        actual: th.FloatOrArr1D2D | list[float] | list[list[float]] | None,
        desired: th.FloatOrArr1D2D | list[float] | list[list[float]],
        rtol: float = 1e-7,
        atol: float = 0,
        equal_nan: bool = True,
        err_msg: str = "",
        verbose: bool = True,
        name: str | None = None,
        fmt: str = DEFAULT_FMT,
        dtype: npt.DTypeLike = np.float64) -> None:
    """Assert that all array elements correspond to the reference within the given tolerances.

    :param actual: actual data
    :param desired: reference data
    :param rtol: relative tolerance
    :param atol: absolute tolerance:
    :param equal_nan: whether NaN values should be considered as equal
    :param err_msg: the error message to be printed in case of failure for dim >= 3 arrays
    :param verbose: whether to print additional info
    :param name: name of the array (printed to help identifying the cause of the error in for loops)
    :param fmt: formatting for printing the values
    :param dtype: data type for conversion from list to ndarray
    """
    if actual is None:
        actual = np.nan
    if np.isscalar(actual):
        if not np.isscalar(desired):
            raise TypeError(
                "Cannot compare a scalar to an array reference. "
                f"Got actual: {type(actual)}, desired: {type(desired)}"
            )
        # np.isscalar() does not narrow the type for the type checker.
        actual_scalar = tp.cast(float, actual)
        desired_scalar = tp.cast(float, desired)
        if np.isclose(actual_scalar, desired_scalar, rtol=rtol, atol=atol, equal_nan=equal_nan):
            return

        lines = [
            *_assert_allclose_header(name, rtol, atol, caller=inspect.stack()[1].function),
            f"Absolute difference: {np.abs(actual_scalar - desired_scalar)}",
            f"Relative difference: {rel_diff_scalar(actual_scalar, desired_scalar)}",
            f"Actual: {actual_scalar}, desired: {desired_scalar}"
        ]
        print("\n".join(lines))
    else:
        actual_arr: th.FloatArr = actual if isinstance(actual, np.ndarray) else np.array(actual, dtype=dtype)

        desired_arr: th.FloatArr
        if np.isscalar(desired):
            desired_arr = np.ones_like(actual_arr) * tp.cast(float, desired)
        elif isinstance(desired, np.ndarray):
            desired_arr = desired
        else:
            desired_arr = np.array(desired, dtype=dtype)

        if actual_arr.shape != desired_arr.shape:
            raise TypeError(
                "The actual and desired arrays should of be the same shape. "
                f"Got: {actual_arr.shape}, {desired_arr.shape}"
            )

        if actual_arr.ndim >= 3:  # noqa: PLR2004
            with np.printoptions(edgeitems=30, linewidth=200):
                np.testing.assert_allclose(actual_arr, desired_arr, rtol, atol, equal_nan, err_msg, verbose)
            return

        close_arr = np.isclose(actual_arr, desired_arr, rtol=rtol, atol=atol, equal_nan=equal_nan)
        if np.all(close_arr):
            return

        mismatched = actual_arr.size - np.sum(close_arr)
        lines = [
            *_assert_allclose_header(name, rtol, atol, caller=inspect.stack()[1].function),
            f"Mismatched elements: {mismatched} / {actual_arr.size} "
            f"({mismatched / actual_arr.size * 100:.1f}%)",
            f"Max absolute difference: {np.nanmax(np.abs(actual_arr - desired_arr))}",
            f"Max relative difference: {np.nanmax(rel_diff_arr(actual_arr, desired_arr))}"
        ]
        print("\n".join(lines))

        # The dimensions are checked at runtime, which does not narrow the shape types for the type checker.
        if actual_arr.ndim == 1:
            print_1d(
                tp.cast(th.FloatArr1D, actual_arr),
                tp.cast(th.FloatArr1D, desired_arr),
                tp.cast(th.BoolArr1D, close_arr)
            )
        elif actual_arr.ndim == 2:  # noqa: PLR2004
            print("Actual:")
            print_2d(tp.cast(th.FloatArr2D, actual_arr), tp.cast(th.BoolArr2D, close_arr), fmt)
            print("Desired:")
            print_2d(tp.cast(th.FloatArr2D, desired_arr), tp.cast(th.BoolArr2D, close_arr), fmt)

    raise AssertionError(". ".join(lines) + ".")
