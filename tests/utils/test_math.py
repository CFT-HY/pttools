"""Tests for the mathematical utilities."""

import unittest

import numpy as np
import pytest

from pttools.utils.math import finite_edge, running_mean


class FiniteEdgeTest(unittest.TestCase):
    """Test finding the edge of the region where a function is finite."""

    @staticmethod
    def sqrt_1_minus_x(x: float) -> float:
        """Return sqrt(1 - x) for x <= 1, and nan otherwise."""
        return np.sqrt(1 - x) if x <= 1 else np.nan

    def test_increasing(self) -> None:
        """Test that the edge is found when searching in the increasing direction."""
        edge = finite_edge(self.sqrt_1_minus_x, 0., 2.)
        assert np.isfinite(self.sqrt_1_minus_x(edge))
        assert edge == pytest.approx(1., abs=5e-13)

    def test_decreasing(self) -> None:
        """Test that the edge is found when searching in the decreasing direction."""
        edge = finite_edge(lambda x: self.sqrt_1_minus_x(-x), 0., -2.)
        assert edge == pytest.approx(-1., abs=5e-13)


class RunningMeanTest(unittest.TestCase):
    """Test the running mean."""

    @staticmethod
    def test_running_mean() -> None:
        """The running mean of a constant is the constant, and that of a line is the line except at the ends."""
        np.testing.assert_allclose(running_mean(np.full(30, 2.), window=23), 2.)
        line = np.arange(50.)
        np.testing.assert_allclose(running_mean(line, window=5)[2:-2], line[2:-2])
