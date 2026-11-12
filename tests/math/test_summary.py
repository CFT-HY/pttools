"""Tests for the compact formatting and the summaries of the values of parameters."""

import unittest

import numpy as np

from pttools.math.formatting import compact
from pttools.math.summary import summarize_values


class CompactTest(unittest.TestCase):
    """Tests for :py:func:`pttools.math.formatting.compact`."""

    @staticmethod
    def test_compact() -> None:
        """Floats are rounded, strings are quoted and other values are converted with str."""
        assert compact(1 / 3) == "0.333333"
        assert compact(np.float64(1e-12)) == "1e-12"
        assert compact(1 / 3, precision=2) == "0.33"
        assert compact(np.inf) == "inf"
        assert compact("a, b") == "'a, b'"
        assert compact(np.str_("a")) == "'a'"
        assert compact(3) == "3"
        assert compact(True) == "True"


class SummaryTest(unittest.TestCase):
    """Tests for :py:func:`pttools.math.summary.summarize_values`."""

    @staticmethod
    def test_bool() -> None:
        """Booleans are summarized by the number of true values."""
        assert summarize_values("failed", np.array([True, False, True])) == "failed: 2/3 true"

    @staticmethod
    def test_constant() -> None:
        """Constants are summarized by their value, with the description."""
        assert summarize_values("x", [0.5, 0.5], "$x$, a value") == "x: 0.5  # $x$, a value"
        assert summarize_values("nan", [np.nan, np.nan]) == "nan: nan"

    @staticmethod
    def test_empty() -> None:
        """Empty values are reported as such."""
        assert summarize_values("x", []) == "x: no values"

    @staticmethod
    def test_numbers() -> None:
        """Numbers are summarized by their range, and the few distinct values are listed."""
        assert summarize_values("v", [0.1, 0.5, 0.1, 0.9]) == "v: 0.1 ... 0.9, 3 distinct: [0.1, 0.5, 0.9]"
        assert summarize_values("n", np.arange(20)) == "n: 0 ... 19, 20 distinct"
        assert summarize_values("n", np.arange(3), max_listed=2) == "n: 0 ... 2, 3 distinct"

    @staticmethod
    def test_rounding_errors() -> None:
        """Floats that differ only by rounding errors are considered the same."""
        assert summarize_values("x", [0.1 + 0.2, 0.3]) == "x: 0.3"
        assert summarize_values("x", [1, 1 + 1e-9], float_digits=6) == "x: 1"
        assert summarize_values("x", [1, 1 + 1e-9]) == "x: 1 ... 1, 2 distinct: [1, 1]"

    @staticmethod
    def test_strings() -> None:
        """Strings are summarized by the number of distinct values, and listed quoted."""
        assert summarize_values("s", ["b", "a, c", "b"]) == "s: 2 distinct: ['a, c', 'b']"
        assert summarize_values("s", [str(i) for i in range(11)]) == "s: 11 distinct"


if __name__ == "__main__":
    unittest.main()
