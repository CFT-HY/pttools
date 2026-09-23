"""Tests for the significant figure utilities."""

import unittest

from tests.paper.sigfig import round_sig, round_sig_error


class SigFigTest(unittest.TestCase):
    """Tests for rounding values to significant figures."""

    def test_round_sig(self):
        self.assertEqual(round_sig(1.23456, 3), "1.23")
        self.assertEqual(round_sig(123456, 2), "120000")
        self.assertEqual(round_sig(-0.001234, 2), "-0.0012")

    def test_round_sig_error(self):
        cases = (
            ((1.23456, 0.0123, 2), ("1.235", "0.012")),
            ((123.456, 1.23, 1), ("123", "1")),
            ((123.4, 20, 1), ("120", "20")),
            ((12345.6, 123, 2), ("12350", "120")),
            ((0.001234, 0.000056, 1), ("0.00123", "0.00006")),
            ((-1.23456, 0.0123, 2), ("-1.235", "0.012")),
        )
        for args, ref in cases:
            with self.subTest(args=args):
                self.assertEqual(round_sig_error(*args), ref)

    def test_round_sig_error_paren(self):
        self.assertEqual(round_sig_error(123.4, 20, 1, paren=True), "120(20)")
        self.assertEqual(round_sig_error(1.23456, 0.0123, 2, paren=True), "1.235(012)")

    def test_round_sig_error_large_error(self):
        """The error has more integer digits than the value."""
        self.assertEqual(round_sig_error(5.0, 12345, 2), ("0", "12000"))
        self.assertEqual(round_sig_error(1500., 12345, 2), ("2000", "12000"))

    def test_round_sig_error_negative(self):
        self.assertEqual(round_sig_error(-123.4, 20, 1), ("-120", "20"))

    def test_round_sig_error_small_value(self):
        self.assertEqual(round_sig_error(0.4, 3, 1), ("0", "3"))
        self.assertEqual(round_sig_error(-0.6, 3, 1), ("-1", "3"))
