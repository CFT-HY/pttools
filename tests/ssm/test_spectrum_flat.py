r"""Tests for the flat-space quantities of the SSM spectrum, which are used for comparisons with simulations."""

import typing as tp
import unittest

import numpy as np
import pytest

from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.ssm import NucType, SSMSpectrum, barotropic, power_gw_bag
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
        cls.bubble = Bubble.cached(BagModel.cached(alpha_n_min=0.01), v_wall=cls.v_wall, alpha_n=cls.alpha_n)
        cls.spectra = {
            nuc_type: SSMSpectrum.cached(
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
        Near the lower end, $z \lesssim 1.3$, the paths deviate by up to 1.8 % for simultaneous nucleation,
        since the thin shell of hybrids was preserved in the sine transform of $\lambda$.
        """
        for nuc_type, spectrum in self.spectra.items():
            with self.subTest(nuc_type=nuc_type):
                legacy = 3 * self.bubble.mean_adiabatic_index**2 \
                    * power_gw_bag(self.y, (self.v_wall, self.alpha_n, nuc_type, (1,)))
                rate = np.exp(np.interp(np.log(self.y), np.log(spectrum.y), np.log(spectrum.pow_gw_rate)))
                np.testing.assert_allclose(rate, legacy, rtol=0.02)

    def test_pow_gw_rate_suppression(self) -> None:
        """The growth rate includes the suppression factor."""
        # The dataset is given explicitly, as the suppression factor of the default dataset WITH_HYBRIDS_EXT
        # is above 1 at these parameters, as some of its data points near the hybrids are.
        suppressed = SSMSpectrum.cached(
            self.bubble, y=self.y, nuc_type=NucType.SIMULTANEOUS,
            suppression=NO_HYBRIDS_EXT, suppression_method=SuppressionMethod.DEFAULT)
        unsuppressed = SSMSpectrum.cached(
            self.bubble, y=self.y, nuc_type=NucType.SIMULTANEOUS, suppression_method=SuppressionMethod.NONE)
        assert suppressed.suppression_factor < 1
        np.testing.assert_allclose(
            suppressed.pow_gw_rate, suppressed.suppression_factor * unsuppressed.pow_gw_rate, rtol=1e-10)

    def test_pow_gw_from_rate(self) -> None:
        r"""$\mathcal{P}_\text{gw,ssm}$ is the growth rate times $r_* \mathcal{H}_* \eta_\text{v}$ and the dilution.

        This is the relation by which :py:class:`pttools.lattice.spectrum.LatticeSpectrum` is scaled
        to the expanding universe.
        """
        for nuc_type, spectrum in self.spectra.items():
            with self.subTest(nuc_type=nuc_type):
                np.testing.assert_allclose(
                    spectrum.pow_gw_ssm,
                    spectrum.r_star * spectrum.H_star_eta_v * spectrum.dilution_of_e * spectrum.pow_gw_rate,
                    rtol=1e-12)
                # The expansion factor is the effective lifetime per the source duration.
                assert spectrum.expansion_factor == pytest.approx(
                    spectrum.H_star_eta_v / (spectrum.r_star * spectrum.delta_tau_v), rel=1e-12)

    def test_expansion_quantities(self) -> None:
        """The quantities of the expansion of the universe of the base class are consistent with the SSM inputs."""
        spectrum = SSMSpectrum(
            self.bubble, y=self.y, r_star=0.2, N_sh=2., a_star_a_r_ratio=0.5, omega_barotropic=0.2, compute=False)
        nu = spectrum.nu_gdh2024
        assert nu > 0
        assert spectrum.delta_tau_v == pytest.approx(2. / self.bubble.ubarf, rel=1e-15)
        assert spectrum.eta_ratio == pytest.approx(0.2 * spectrum.delta_tau_v / (1 + nu), rel=1e-14)
        assert spectrum.source_lifetime_factor == barotropic.source_lifetime_factor(
            ubarf=self.bubble.ubarf, r_star=0.2, N_sh=2., nu=nu)
        assert spectrum.dilution_of_e == barotropic.dilution_of_e(a_star_a_r_ratio=0.5, nu=nu)
        assert spectrum.tau_end == pytest.approx(spectrum.tau_star + spectrum.delta_tau_v, rel=1e-15)
        assert spectrum.tau_star == pytest.approx((1 + nu) / 0.2, rel=1e-15)

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
