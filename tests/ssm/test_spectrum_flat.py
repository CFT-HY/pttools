r"""Tests for the flat-space quantities of the SSM spectrum, which are used for comparisons with simulations."""

import typing as tp
import unittest

import numpy as np
import pytest

from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.ssm import NucType, SSMSpectrum, power_gw_bag
from pttools.ssm.suppression import NO_HYBRIDS_EXT, SuppressionMethod


class SSMSpectrumFlatTest(unittest.TestCase):
    """Tests for :py:attr:`SSMSpectrum.pow_gw_rate` and :py:attr:`SSMSpectrum.pow_U_par`."""

    v_wall: tp.ClassVar[float] = 0.6
    alpha_n: tp.ClassVar[float] = 0.05
    y: tp.ClassVar[np.ndarray] = np.logspace(0, 2, 100)
    bubble: tp.ClassVar[Bubble]
    spectra: tp.ClassVar[dict[NucType, SSMSpectrum]]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.bubble = Bubble(BagModel(alpha_n_min=0.01), v_wall=cls.v_wall, alpha_n=cls.alpha_n)
        cls.spectra = {
            nuc_type: SSMSpectrum(
                cls.bubble, y=np.logspace(-2, 4, 1200), nuc_type=nuc_type, suppression_method=SuppressionMethod.NONE)
            for nuc_type in (NucType.EXPONENTIAL, NucType.SIMULTANEOUS)
        }

    def test_pow_gw_rate_legacy(self) -> None:
        r"""The growth rate agrees with $3 \Gamma^2$ times the legacy :py:func:`pttools.ssm.power_gw_bag`.

        The legacy function is the one used for computing the suppression datasets of :gowling_2021:`\ `
        in :py:mod:`pttools.ssm.suppression.suppression_ssm_data.suppression_ssm_calculator`,
        where $\Gamma = 4/3$ instead of the value of the fluid profile.
        The comparison is limited to $z \leq 100$, as the legacy path deviates by up to 10–20 % near the upper end
        of the given $z$ range for all solution types, which is an edge effect of its internal grids.
        """
        for nuc_type, spectrum in self.spectra.items():
            with self.subTest(nuc_type=nuc_type):
                legacy = 3 * self.bubble.mean_adiabatic_index**2 \
                    * power_gw_bag(self.y, (self.v_wall, self.alpha_n, nuc_type, (1,)))
                rate = np.exp(np.interp(np.log(self.y), np.log(spectrum.y), np.log(spectrum.pow_gw_rate)))
                np.testing.assert_allclose(rate, legacy, rtol=0.01)

    def test_pow_gw_rate_suppression(self) -> None:
        """The growth rate includes the suppression factor."""
        # The dataset is given explicitly, as the suppression factor of the default dataset WITH_HYBRIDS_EXT
        # is above 1 at these parameters, as some of its data points near the hybrids are.
        suppressed = SSMSpectrum(
            self.bubble, y=self.y, nuc_type=NucType.SIMULTANEOUS,
            suppression=NO_HYBRIDS_EXT, suppression_method=SuppressionMethod.DEFAULT)
        unsuppressed = SSMSpectrum(
            self.bubble, y=self.y, nuc_type=NucType.SIMULTANEOUS, suppression_method=SuppressionMethod.NONE)
        assert suppressed.suppression_factor < 1
        np.testing.assert_allclose(
            suppressed.pow_gw_rate, suppressed.suppression_factor * unsuppressed.pow_gw_rate, rtol=1e-10)

    def test_pow_U_par_normalisation(self) -> None:
        r"""$\int \frac{dz}{z} \mathcal{P}_{U_{\parallel}} = \bar{U}_f^2$."""
        for nuc_type, spectrum in self.spectra.items():
            with self.subTest(nuc_type=nuc_type):
                assert spectrum.ubar_par == pytest.approx(spectrum.ubarf, rel=0.01)

    def test_omega_gw_rate(self) -> None:
        """The integrated growth rate is the integral of the spectrum over ln z."""
        spectrum = self.spectra[NucType.SIMULTANEOUS]
        assert spectrum.omega_gw_rate == pytest.approx(
            np.trapezoid(spectrum.pow_gw_rate, np.log(spectrum.y)), rel=1e-12)

    def test_integral_scale(self) -> None:
        """The integral scale is of the order of the bubble spacing."""
        for spectrum in self.spectra.values():
            assert 0.01 < spectrum.integral_scale_par < 1


if __name__ == "__main__":
    unittest.main()
