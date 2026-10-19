"""Unit tests for the Numba overloads of scipy.integrate."""

import typing as tp
import unittest

import numpy as np
import pytest
import scipy.integrate

from pttools.math import assert_allclose
from pttools.speedup import njit
from pttools.speedup.integrate import simpson_h
import pttools.type_hints as th

#: Numbers of points, which include the special cases of 1 and 2 points, and both odd and even numbers
N_POINTS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 101, 1000)


@njit
def simpson_x_jitted(y: th.FloatArr1D, x: th.FloatArr1D) -> float:
    """JIT-compiled call of scipy.integrate.simpson with sample points, which uses the Numba overload."""
    return scipy.integrate.simpson(y, x=x)


@njit
def simpson_dx_jitted(y: th.FloatArr1D, dx: float) -> float:
    """JIT-compiled call of scipy.integrate.simpson with uniform spacing, which uses the Numba overload."""
    return scipy.integrate.simpson(y, dx=dx)


@njit
def simpson_default_jitted(y: th.FloatArr1D) -> float:
    """JIT-compiled call of scipy.integrate.simpson with the default spacing, which uses the Numba overload."""
    return scipy.integrate.simpson(y)


@njit
def simpson_axis_jitted(y: th.FloatArr1D, x: th.FloatArr1D, axis: int) -> float:
    """JIT-compiled call of scipy.integrate.simpson with an axis, which uses the Numba overload."""
    return scipy.integrate.simpson(y, x=x, axis=axis)


class SimpsonTest(unittest.TestCase):
    """Test the Numba overload of scipy.integrate.simpson against SciPy."""

    rng: np.random.Generator

    @tp.override
    def setUp(self) -> None:
        self.rng = np.random.default_rng(1)

    def test_irregular(self) -> None:
        """Test irregularly spaced sample points."""
        for n in N_POINTS:
            with self.subTest(n=n):
                x = np.sort(self.rng.uniform(0, 3, n))
                y = np.sin(x) + x**2
                assert_allclose(simpson_x_jitted(y, x), scipy.integrate.simpson(y, x=x), rtol=1e-13, atol=1e-15)

    def test_decreasing(self) -> None:
        """Test decreasing sample points, for which the integral has the opposite sign."""
        x = np.sort(self.rng.uniform(0, 3, 50))[::-1].copy()
        y = np.exp(-x)
        assert_allclose(simpson_x_jitted(y, x), scipy.integrate.simpson(y, x=x), rtol=1e-13)

    def test_uniform(self) -> None:
        """Test uniform spacing given with dx and with the default dx = 1."""
        for n in N_POINTS:
            with self.subTest(n=n):
                y = np.cos(np.linspace(0, 2, n)) + 2
                assert_allclose(simpson_dx_jitted(y, 0.25), scipy.integrate.simpson(y, dx=0.25), rtol=1e-13)
                assert_allclose(simpson_default_jitted(y), scipy.integrate.simpson(y), rtol=1e-13)

    def test_axis(self) -> None:
        """Test that the axis -1 and 0 are accepted for 1D arrays, and others are rejected."""
        x = np.linspace(0, 1, 11)
        y = x**3
        for axis in (-1, 0):
            with self.subTest(axis=axis):
                assert_allclose(
                    simpson_axis_jitted(y, x, axis), scipy.integrate.simpson(y, x=x, axis=axis), rtol=1e-13)
        # The messages of SciPy and of the overload differ, but the error type is the same.
        with pytest.raises(IndexError):
            simpson_axis_jitted(y, x, 1)

    def test_exact(self) -> None:
        r"""Test the polynomials for which the rule is exact.

        For irregular spacing, the rule is exact for quadratic polynomials, $\int_0^2 x^2 dx = \frac{8}{3}$,
        and for uniform spacing with an odd number of points, also for cubic polynomials, $\int_0^2 x^3 dx = 4$.
        """
        for n in (3, 4, 9, 10):
            with self.subTest(n=n):
                x = np.sort(np.concatenate(([0., 2.], self.rng.uniform(0, 2, n - 2))))
                assert_allclose(simpson_x_jitted(x**2, x), 8 / 3, rtol=1e-12)
        for n in (3, 9, 101):
            with self.subTest(n=n):
                x = np.linspace(0, 2, n)
                assert_allclose(simpson_dx_jitted(x**3, x[1] - x[0]), 4., rtol=1e-12)

    @staticmethod
    def test_simpson_h() -> None:
        """Test the core function directly with the intervals."""
        x = np.linspace(0, np.pi, 21)
        assert_allclose(simpson_h(np.sin(x), np.diff(x)), scipy.integrate.simpson(np.sin(x), x=x), rtol=1e-13)
