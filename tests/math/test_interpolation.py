"""Tests for the interpolation utilities."""

import unittest

import numpy as np

from pttools.math.interpolation import interpolate_log_log


class InterpolateLogLogTest(unittest.TestCase):
    """Tests for the log-log interpolation."""

    @staticmethod
    def test_power_law() -> None:
        """Power laws should be interpolated exactly."""
        xp = np.logspace(-3, 3, 20)
        x = np.logspace(-2.9, 2.9, 333)
        for index in (-2.5, 0., 3.):
            np.testing.assert_allclose(interpolate_log_log(x, xp, 2 * xp**index), 2 * x**index, rtol=1e-12)

    @staticmethod
    def test_nodes() -> None:
        """The values at the points of the data should be retained."""
        xp = np.logspace(-1, 2, 50)
        fp = np.exp(np.sin(xp))
        np.testing.assert_allclose(interpolate_log_log(xp, xp, fp), fp, rtol=1e-14)

    @staticmethod
    def test_reference() -> None:
        """The result should match the interpolation of the logarithms with NumPy."""
        rng = np.random.default_rng(0)
        xp = np.sort(rng.uniform(0.1, 10, 30))
        fp = rng.uniform(0.5, 2, 30)
        x = np.linspace(0.2, 9, 100)
        ref = np.exp(np.interp(np.log(x), np.log(xp), np.log(fp)))
        np.testing.assert_allclose(interpolate_log_log(x, xp, fp), ref, rtol=1e-14)

    @staticmethod
    def test_non_positive() -> None:
        """Values that are not all positive should be interpolated linearly in log x."""
        xp = np.array([1., 10., 100.])
        fp = np.array([0., 1., -1.])
        x = np.array([np.sqrt(10), np.sqrt(1000)])
        np.testing.assert_allclose(interpolate_log_log(x, xp, fp), [0.5, 0.], atol=1e-14)

    @staticmethod
    def test_outside() -> None:
        """Outside the range of the data, the values at the endpoints should be used."""
        xp = np.array([1., 10.])
        fp = np.array([2., 20.])
        np.testing.assert_allclose(interpolate_log_log(np.array([0.1, 100.]), xp, fp), [2., 20.], rtol=1e-14)


if __name__ == "__main__":
    unittest.main()
