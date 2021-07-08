"""Utility functions and constants for unit testing"""

import inspect
import os.path

import numpy as np

from tests import printing
import tests.math

TEST_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = os.path.join(TEST_PATH, "test_data")
TEST_RESULT_PATH = os.path.join(os.path.dirname(TEST_PATH), "test-results")
TEST_FIGURE_PATH = os.path.join(TEST_RESULT_PATH, "figures")


def assert_allclose(
        actual: np.ndarray,
        desired: np.ndarray,
        rtol: float = 1e-7,
        atol: float = 0,
        equal_nan: bool = True,
        err_msg: str = "",
        verbose: bool = True,
        fmt: str = printing.DEFAULT_FMT):
    if actual.ndim >= 3:
        with np.printoptions(edgeitems=30, linewidth=200):
            np.testing.assert_allclose(actual, desired, rtol, atol, equal_nan, err_msg, verbose)
        return

    close = np.isclose(actual, desired, rtol=rtol, atol=atol, equal_nan=equal_nan)
    if np.all(close):
        return
    print(f"assert_allclose failed in {inspect.stack()[1].function}")
    print(f"Not equal to tolerance rtol={rtol}, atol={atol}")
    mismatched = actual.size - np.sum(close)
    print(f"Mismatched elements: {mismatched} / {actual.size} ({mismatched / actual.size * 100:.1f}%)")
    print(f"Max absolute difference: {np.max(np.abs(actual - desired))}")
    print(f"Max relative difference: {np.max(tests.math.rel_diff_arr(actual, desired))}")

    if actual.ndim == 1:
        printing.print_1d(actual, desired, close)
    if actual.ndim == 2:
        print("actual:")
        printing.print_2d(actual, close, fmt)
        print("desired:")
        printing.print_2d(desired, close, fmt)

    raise AssertionError(f"assert_allclose failed: arrays not equal to tolerance rtol={rtol}, atol={atol}")
