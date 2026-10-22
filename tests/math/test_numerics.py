"""Tests for the mathematical utilities."""

import unittest

import numpy as np
import pytest

from pttools import math as pt_math
from pttools.math.numerics import finite_edge, is_nan_or_outside, isclose_scalar, running_mean
from pttools.speedup import njit
from pttools.speedup.threads import powers_of_2


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


class IsCloseScalarTest(unittest.TestCase):
    """Test the scalar version of numpy.isclose."""

    @staticmethod
    def test_matches_numpy() -> None:
        """The results should be the same as those of numpy.isclose, including the special values."""
        values = (
            0., -0., 1e-9, -1e-9, 1e-8, 2e-8, 1., 1 + 1e-6, 1 + 1e-4, -1., 0.5, 1e10, 1e10 + 1e5, 1e10 + 1e6,
            np.inf, -np.inf, np.nan, np.float64(0.3), np.nextafter(1., 2.)
        )
        for a in values:
            for b in values:
                for rtol, atol in ((1e-5, 1e-8), (0., 0.), (1e-2, 0.)):
                    assert isclose_scalar(a, b, rtol=rtol, atol=atol) == np.isclose(a, b, rtol=rtol, atol=atol), \
                        (a, b, rtol, atol)

    @staticmethod
    def test_jitted() -> None:
        """The function should be usable in jitted functions and give the same results there."""
        @njit
        def jitted(a: float, b: float) -> tuple[bool, bool]:
            return isclose_scalar(a, b), isclose_scalar(a, b, 0., 0.)

        for a, b in ((1., 1. + 1e-6), (0., 1e-9), (0., np.inf), (np.inf, np.inf), (np.nan, np.nan), (1., 2.)):
            assert jitted(a, b) == (isclose_scalar(a, b), isclose_scalar(a, b, 0., 0.)), (a, b)


class IsNanOrOutsideTest(unittest.TestCase):
    """Test the check for nan values and values outside their bounds."""

    @staticmethod
    def test_scalar() -> None:
        """A single value should be checked against its bounds, which are inclusive unless isclose=True."""
        assert not is_nan_or_outside(0.5, 0., 1.)
        assert not is_nan_or_outside(1., 0., 1.)
        assert is_nan_or_outside(1., 0., 1., isclose=True)
        assert is_nan_or_outside(1e-9, 0., 1., isclose=True)
        assert not is_nan_or_outside(1e-3, 0., 1., isclose=True)
        assert is_nan_or_outside(np.nan, 0., 1.)
        assert is_nan_or_outside(-0.1, 0., 1.)
        assert is_nan_or_outside(1.1, 0., 1.)

    @staticmethod
    def test_infinite_bounds() -> None:
        """Infinite bounds should not be checked with isclose, so that an infinite value is within them."""
        assert not is_nan_or_outside(np.inf, 0., np.inf, isclose=True)
        assert not is_nan_or_outside(1e10, 0., np.inf, isclose=True)
        assert is_nan_or_outside(-np.inf, 0., np.inf)

    @staticmethod
    def test_tuple() -> None:
        """Several values should be checked against their bounds, and any invalid value should make it True."""
        lower = (0., 0., 0.)
        upper = (1., np.inf, 1.)
        assert not is_nan_or_outside((0.4, 1.3, np.float64(0.6)), lower, upper, isclose=True)
        assert is_nan_or_outside((0.4, 1.3, 1 - 1e-7), lower, upper, isclose=True)
        assert not is_nan_or_outside((0.4, 1.3, 1.), lower, upper)
        assert is_nan_or_outside((0.4, np.nan, 0.6), lower, upper)
        assert is_nan_or_outside((0.4, -1., 0.6), lower, upper)
        with pytest.raises(ValueError, match="same lengths"):
            is_nan_or_outside((0.4, 1.3), lower, upper)

    @staticmethod
    def test_jitted() -> None:
        """The function should be usable in jitted functions with scalars and tuples, and give the same results."""
        @njit
        def jitted(value: float, values: tuple[float, float]) -> tuple[bool, bool, bool, bool]:
            lower = (0., 0.)
            upper = (1., np.inf)
            return (
                is_nan_or_outside(value, 0., 1.), is_nan_or_outside(value, 0., 1., True),
                is_nan_or_outside(values, lower, upper), is_nan_or_outside(values, lower, upper, isclose=True)
            )

        for value, values in ((0.5, (0.5, 2.)), (1., (1., 2.)), (np.nan, (0.5, np.nan)), (-1., (0.5, -1.))):
            expected = (
                is_nan_or_outside(value, 0., 1.), is_nan_or_outside(value, 0., 1., True),
                is_nan_or_outside(values, (0., 0.), (1., np.inf)),
                is_nan_or_outside(values, (0., 0.), (1., np.inf), isclose=True)
            )
            assert jitted(value, values) == expected, (value, values)


class PowersOf2Test(unittest.TestCase):
    """Test the powers of 2."""

    @staticmethod
    def test_powers_of_2() -> None:
        """The powers of 2 up to the given value, and the value itself if it's not a power of 2 and requested."""
        np.testing.assert_array_equal(powers_of_2(8), [1, 2, 4, 8])
        np.testing.assert_array_equal(powers_of_2(12, include_max=True), [1, 2, 4, 8, 12])
        np.testing.assert_array_equal(powers_of_2(12), [1, 2, 4, 8])

    @staticmethod
    def test_reexport() -> None:
        """The function can also be imported from :py:mod:`pttools.math`."""
        assert pt_math.powers_of_2 is powers_of_2


class RunningMeanTest(unittest.TestCase):
    """Test the running mean."""

    @staticmethod
    def test_running_mean() -> None:
        """The running mean of a constant is the constant, and that of a line is the line except at the ends."""
        np.testing.assert_allclose(running_mean(np.full(30, 2.), window=23), 2.)
        line = np.arange(50.)
        np.testing.assert_allclose(running_mean(line, window=5)[2:-2], line[2:-2])
