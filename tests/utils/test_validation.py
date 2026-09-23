"""Tests for the validation utilities."""

import logging
import unittest

import numpy as np

from pttools.utils.validation import check_value_in_range


class CheckValueInRangeTest(unittest.TestCase):
    def test_valid(self) -> None:
        self.assertEqual(check_value_in_range(1., 0., 2., name="x"), 1.)
        arr = np.array([0.5, 1., 1.5])
        np.testing.assert_array_equal(check_value_in_range(arr, 0., 2., name="x"), arr)

    def test_invalid_limits(self) -> None:
        with self.assertRaises(ValueError):
            check_value_in_range(1., 2., 0., name="x")

    def test_too_small(self) -> None:
        with self.assertRaises(ValueError):
            check_value_in_range(-1., 0., 2., name="x")
        with self.assertLogs("pttools.utils.validation", level=logging.ERROR) as logs:
            ret = check_value_in_range(-1., 0., 2., name="x", context="a test", error_on_invalid=False)
        self.assertTrue(np.isnan(ret))
        self.assertEqual(len(logs.records), 1)
        self.assertIn("x_min", logs.output[0])
        self.assertIn("for a test", logs.output[0])

    def test_too_large_arr(self) -> None:
        arr = np.array([0.5, 1., 3.])
        with self.assertLogs("pttools.utils.validation", level=logging.ERROR):
            ret = check_value_in_range(arr, 0., 2., name="x", error_on_invalid=False)
        np.testing.assert_array_equal(ret, [0.5, 1., np.nan])
        # The given array should not be modified.
        np.testing.assert_array_equal(arr, [0.5, 1., 3.])

    def test_nan_scalar(self) -> None:
        with self.assertLogs("pttools.utils.validation", level=logging.ERROR) as logs:
            ret = check_value_in_range(np.nan, 0., 2., name="x", context="a test")
        self.assertTrue(np.isnan(ret))
        self.assertEqual(len(logs.records), 1)
        self.assertEqual(logs.records[0].getMessage(), "Got nan for x in test_nan_scalar for a test.")

    def test_nan_arr(self) -> None:
        arr = np.array([0.5, np.nan, 3.])
        with self.assertLogs("pttools.utils.validation", level=logging.ERROR) as logs:
            ret = check_value_in_range(arr, 0., 2., name="x", error_on_invalid=False)
        # The nan values are logged, and the too large values are logged separately.
        self.assertEqual(len(logs.records), 2)
        self.assertEqual(logs.records[0].getMessage(), "Got nan for 1/3 values of x in test_nan_arr.")
        np.testing.assert_array_equal(ret, [0.5, np.nan, np.nan])

    def test_none(self) -> None:
        with self.assertLogs("pttools.utils.validation", level=logging.ERROR) as logs:
            # None is not in the type hints, but it's handled for robustness.
            # pyrefly: ignore[bad-specialization]
            ret = check_value_in_range(None, 0., 2., name="x")
        # pyrefly: ignore[no-matching-overload]
        self.assertTrue(np.isnan(ret))
        self.assertEqual(logs.records[0].getMessage(), "Got None for x in test_none.")

    def test_no_logging(self) -> None:
        with self.assertNoLogs("pttools.utils.validation", level=logging.ERROR):
            ret = check_value_in_range(np.nan, 0., 2., name="x", log_invalid=False)
            ret2 = check_value_in_range(-1., 0., 2., name="x", error_on_invalid=False, log_invalid=False)
        self.assertTrue(np.isnan(ret))
        self.assertTrue(np.isnan(ret2))
