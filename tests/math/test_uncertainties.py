"""Tests for the utilities for floats with uncertainties."""

import math
import unittest

import numpy as np
import pytest
from uncertainties import UFloat as UFloatClass
from uncertainties import ufloat, unumpy

from pttools.math.uncertainties import sqrt_u
from pttools.type_hints import UFloat, UFloatArr1D


class SqrtUTest(unittest.TestCase):
    """Tests for the square root of floats with uncertainties and arrays of them."""

    @staticmethod
    def test_float() -> None:
        """Floats and arrays of floats should give the same results and types as NumPy."""
        assert sqrt_u(2.) == math.sqrt(2.)
        x = np.array([0.5, 2., 9.])
        result = sqrt_u(x)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, np.sqrt(x))

    @staticmethod
    def test_ufloat() -> None:
        r"""The relative uncertainty of $\sqrt{x}$ should be half of that of $x$."""
        result: UFloat = sqrt_u(ufloat(4., 0.1))
        assert isinstance(result, UFloatClass)
        assert result.nominal_value == 2.
        assert result.std_dev == pytest.approx(0.025, rel=1e-15)

    @staticmethod
    def test_array() -> None:
        """Arrays of floats with uncertainties, and object arrays with also floats, should be supported."""
        result: UFloatArr1D = sqrt_u(unumpy.uarray([4., 9.], [0.1, 0.1]))
        np.testing.assert_array_equal(unumpy.nominal_values(result), [2., 3.])
        np.testing.assert_allclose(unumpy.std_devs(result), [0.025, 0.1 / 6], rtol=1e-15)
        mixed: UFloatArr1D = sqrt_u(np.array([4., ufloat(9., 0.1)], dtype=object))
        assert mixed[0] == 2.
        assert mixed[1].std_dev == pytest.approx(0.1 / 6, rel=1e-15)


if __name__ == "__main__":
    unittest.main()
