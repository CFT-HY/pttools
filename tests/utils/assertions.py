"""Utility functions and constants for unit testing"""

import inspect
import typing as tp

import numpy as np

from . import math as test_math
from . import printing


def assert_allclose(
        actual: tp.Union[float, tp.Union[tp.List[float], tp.List[list]], np.ndarray],
        desired: tp.Union[float, tp.Union[tp.List[float], tp.List[list]], np.ndarray],
        rtol: float = 1e-7,
        atol: float = 0,
        equal_nan: bool = True,
        err_msg: str = "",
        verbose: bool = True,
        fmt: str = printing.DEFAULT_FMT,
        dtype: np.dtype = np.float_):
    """Assert that all array elements correspond to the reference within the given tolerances

    :param actual: actual data
    :param desired: reference data
    :param rtol: relative tolerance
    :param atol: absolute tolerance:
    :param equal_nan: whether NaN values should be considered as equal
    :param err_msg: the error message to be printed in case of failure for dim >= 3 arrays
    :param verbose: whether to print additional info
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

        if actual.ndim >= 3:
            with np.printoptions(edgeitems=30, linewidth=200):
                np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg, verbose)
            return

    close = np.isclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if np.all(close):
        return

    print(f"assert_allclose failed in {inspect.stack()[1].function}")
    print(f"Not equal to tolerance rtol={rtol}, atol={atol}")
    if is_scalar:
        print(f"Absolute difference: {np.abs(actual - desired)}")
        print(f"Relative difference: {test_math.rel_diff_scalar(actual, desired)}")
        print(f"actual: {actual}, desired: {desired}")
    else:
        mismatched = actual.size - np.sum(close)
        print(f"Mismatched elements: {mismatched} / {actual.size} ({mismatched / actual.size * 100:.1f}%)")
        print(f"Max absolute difference: {np.nanmax(np.abs(actual - desired))}")
        print(f"Max relative difference: {np.nanmax(test_math.rel_diff_arr(actual, desired))}")

        if actual.ndim == 1:
            printing.print_1d(actual, desired, close)
        elif actual.ndim == 2:
            print("actual:")
            printing.print_2d(actual, close, fmt)
            print("desired:")
            printing.print_2d(desired, close, fmt)

    raise AssertionError(f"assert_allclose failed: not equal to tolerance rtol={rtol}, atol={atol}")
