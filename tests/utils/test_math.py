"""Tests for the mathematical utilities."""

import unittest

import numpy as np

from pttools.utils.math import finite_edge


class FiniteEdgeTest(unittest.TestCase):
    @staticmethod
    def sqrt_1_minus_x(x: float) -> float:
        return np.sqrt(1 - x) if x <= 1 else np.nan

    def test_increasing(self) -> None:
        edge = finite_edge(self.sqrt_1_minus_x, 0., 2.)
        self.assertTrue(np.isfinite(self.sqrt_1_minus_x(edge)))
        self.assertAlmostEqual(edge, 1., places=12)

    def test_decreasing(self) -> None:
        edge = finite_edge(lambda x: self.sqrt_1_minus_x(-x), 0., -2.)
        self.assertAlmostEqual(edge, -1., places=12)
