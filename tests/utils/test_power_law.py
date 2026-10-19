"""Tests for the fits of power laws."""

import unittest

import numpy as np
import pytest

from pttools.utils.power_law import fit_power_law


class PowerLawTest(unittest.TestCase):
    """Tests for the fitting functions with exact synthetic data."""

    @staticmethod
    def test_fit_power_law() -> None:
        r"""The index of exact data is recovered for both $\zeta$ and $\lambda$."""
        t = np.linspace(100, 3000, 60)
        for factor, index in ((-0.5, 1.3), (1., 0.25)):
            values = 0.3 * (1 + (t - t[0]) / 150)**(factor * index)
            fit = fit_power_law(t, values, factor=factor)
            assert fit.index == pytest.approx(index, rel=1e-6)
            assert fit.t_scale == pytest.approx(150, rel=1e-6)
            np.testing.assert_allclose(0.3 * fit(t), values, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
