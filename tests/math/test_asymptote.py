"""Tests for the fits of power law approaches to asymptotes."""

import unittest

import numpy as np
import pytest

from pttools.math.asymptote import asymptotic_growth, fit_asymptote, fit_start_index


class AsymptoteTest(unittest.TestCase):
    """Tests for the fitting functions with exact synthetic data."""

    @staticmethod
    def test_fit_asymptote() -> None:
        """The parameters of exact data are recovered, independently of the magnitude of the values."""
        t = np.linspace(500, 5000, 100)
        for value_inf in (1e-6, 2.5, 1e4):
            fit = fit_asymptote(t, asymptotic_growth(t, value_inf=value_inf, t_decay=300, index=0.7), t_decay_guess=200)
            assert fit.value == pytest.approx(value_inf, rel=1e-8)
            assert fit.t_decay == pytest.approx(300, rel=1e-8)
            assert fit.index == pytest.approx(0.7, rel=1e-8)
            assert (fit.t_start, fit.t_end) == (500, 5000)
            np.testing.assert_allclose(fit(t), asymptotic_growth(t, value_inf=value_inf, t_decay=300, index=0.7))

    @staticmethod
    def test_fit_asymptote_too_few_points() -> None:
        """A time series with fewer points than there are parameters cannot be fitted."""
        t = np.array([500., 1000.])
        with pytest.raises(ValueError, match="at least 3 points"):
            fit_asymptote(t, asymptotic_growth(t, value_inf=1., t_decay=300, index=0.7))

    @staticmethod
    def test_fit_start_index() -> None:
        """The fit starts after the last value below the fraction of the last value."""
        values = np.array([0., 0.5, 0.9, 0.7, 0.85, 0.95, 1.])
        assert fit_start_index(values, 0.8) == 4
        assert fit_start_index(values, 0.) == 0


if __name__ == "__main__":
    unittest.main()
