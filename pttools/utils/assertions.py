"""Utility functions and constants for unit testing"""

import inspect

import numpy as np

import pttools.type_hints as th
from pttools.utils.math import rel_diff_arr, rel_diff_scalar
from pttools.utils.printing import DEFAULT_FMT, print_1d, print_2d


def assert_allclose(
        actual: th.FloatOrArr1D2D | list[float] | list[list[float]],
        desired: th.FloatOrArr1D2D | list[float] | list[list[float]],
        rtol: float = 1e-7,
        atol: float = 0,
        equal_nan: bool = True,
        err_msg: str = "",
        verbose: bool = True,
        name: str | None = None,
        fmt: str = DEFAULT_FMT,
        dtype: np.dtype = np.float64) -> None:
    """Assert that all array elements correspond to the reference within the given tolerances

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
    is_scalar = np.isscalar(actual)
    if is_scalar:
        if not np.isscalar(desired):
            raise TypeError(
                "Cannot compare a scalar to an array reference. "
                f"Got actual: {type(actual)}, desired: {type(desired)}"
            )
    else:
        if not isinstance(actual, np.ndarray):
            actual = np.array(actual, dtype=dtype)

        if np.isscalar(desired):
            desired = np.ones_like(actual) * desired
        elif not isinstance(desired, np.ndarray):
            desired = np.array(desired, dtype=dtype)

        if actual.shape != desired.shape:
            raise TypeError(
                f"The actual and desired arrays should of be the same shape. Got: {actual.shape}, {desired.shape}"
            )

        if actual.ndim >= 3:
            with np.printoptions(edgeitems=30, linewidth=200):
                np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg, verbose)
            return

    close = np.isclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if np.all(close):
        return

    lines = [
        f"assert_allclose failed {f'for {name} ' if name is not None else ''}in {inspect.stack()[1].function}",
        f"Not equal to tolerance rtol={rtol}, atol={atol}"
    ]
    if is_scalar:
        lines += [
            f"Absolute difference: {np.abs(actual - desired)}",
            f"Relative difference: {rel_diff_scalar(actual, desired)}"
            f"Actual: {actual}, desired: {desired}"
        ]
    else:
        mismatched = actual.size - np.sum(close)
        lines += [
            f"Mismatched elements: {mismatched} / {actual.size} ({mismatched / actual.size * 100:.1f}%)",
            f"Max absolute difference: {np.nanmax(np.abs(actual - desired))}",
            f"Max relative difference: {np.nanmax(rel_diff_arr(actual, desired))}"
        ]
    print("\n".join(lines))

    if not is_scalar:
        if actual.ndim == 1:
            print_1d(actual, desired, close)
        elif actual.ndim == 2:
            print("Actual:")
            print_2d(actual, close, fmt)
            print("Desired:")
            print_2d(desired, close, fmt)

    raise AssertionError(". ".join(lines) + ".")
