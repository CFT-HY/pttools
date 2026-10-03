"""Tests for the SSMSpectrum class."""

import typing as tp
import unittest

import numpy as np

from pttools.bubble import Bubble
from pttools.models import ConstCSModel
from pttools.ssm import DEFAULT_NUC_TYPE, SSMSpectrum
from tests.utils import TEST_JSON_PATH


class SSMSpectrumTest(unittest.TestCase):
    """Tests for the SSMSpectrum class."""

    spectrum: SSMSpectrum

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        model = ConstCSModel(css2=1/3-0.01, csb2=1/3-0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
        cls.spectrum = SSMSpectrum(bubble, beta_tilde=1.23)

    def test_beta(self) -> None:
        """Test that the nucleation rate parameter beta is finite for a given Hubble rate."""
        self.assertTrue(np.isfinite(self.spectrum.beta(1.23)))

    def test_bubble_spacing_enlargement_factor(self) -> None:
        """Test that the bubble spacing enlargement factor is larger than 1."""
        self.assertGreater(self.spectrum.bubble_spacing_enlargement_factor, 1.)

    def test_cs2_Tn(self) -> None:
        r"""In the constant sound speed model $c_s^2$ is constant in each phase."""
        self.assertAlmostEqual(self.spectrum.css2_Tn, 1/3 - 0.01, places=12)
        self.assertAlmostEqual(self.spectrum.csb2_Tn, 1/3 - 0.011, places=12)
        self.assertEqual(self.spectrum.css2_Tn, self.spectrum.bubble.css2_Tn)

    def test_dilution_of_e(self) -> None:
        """Test that the dilution of the energy density is between 0 and 1."""
        self.assertGreater(self.spectrum.dilution_of_e, 0)
        self.assertLessEqual(self.spectrum.dilution_of_e, 1)

    def test_eta_ratio(self) -> None:
        """Test that the source duration in units of the conformal time eta_* is larger than 1."""
        self.assertGreater(self.spectrum.eta_ratio, 1)

    def test_export(self) -> None:
        """Test that the spectrum can be exported as JSON."""
        self.spectrum.export(TEST_JSON_PATH / "ssm-spectrum.json")

    def test_H_star_eta_star(self) -> None:
        """Test that the conformal Hubble rate times the conformal time at the start of the acoustics is positive."""
        self.assertGreater(self.spectrum.H_star_eta_star, 0)

    def test_H_star_eta_sh(self) -> None:
        """Test that the Hubble-scaled shock formation time is positive."""
        self.assertGreater(self.spectrum.H_star_eta_sh, 0)

    def test_H_star_eta_v(self) -> None:
        """Test that the Hubble-scaled effective lifetime of the source is positive."""
        self.assertGreater(self.spectrum.H_star_eta_v, 0)

    def test_H_star_eta_v_old(self) -> None:
        """Test that the old approximation of the effective lifetime of the source is positive."""
        self.assertGreater(self.spectrum.H_star_eta_v_old, 0)

    def test_hx(self) -> None:
        """Test that the fractional volume hx at which bubble nucleation stops is positive."""
        self.assertGreater(self.spectrum.hx, 0)

    def test_k_peak_eta_star(self) -> None:
        """Test that the peak wavenumber scaled by the conformal time at GW formation is positive."""
        self.assertGreater(self.spectrum.k_peak_eta_star, 0)

    def test_J(self) -> None:
        """Test that the combined lifetime factor J is positive."""
        self.assertGreater(self.spectrum.J, 0)

    def test_nucleation_f(self) -> None:
        """Test that the nucleation suppression function f is positive."""
        self.assertGreater(self.spectrum.nucleation_f, 0)

    def test_pow_gw(self) -> None:
        """Test that the GW power spectrum has no nan values."""
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_gw)))

    def test_pow_gw_expanded(self) -> None:
        """Test that the expanded GW power spectrum has no nan values."""
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_gw_expanded)))

    def test_pow_gw_int(self) -> None:
        """Test that the integrated GW power spectrum has no nan values."""
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_gw_int)))

    def test_pow_gw_low(self) -> None:
        """Test that the low-frequency GW power spectrum has no nan values."""
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_gw_low)))

    def test_pow_gw_ssm(self) -> None:
        """Test that the SSM GW power spectrum has no nan values."""
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_gw_ssm)))

    def test_pow_v(self) -> None:
        """Test that the velocity power spectrum has no nan values."""
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_v)))

    def test_pow_v_tilde(self) -> None:
        """Test that the power spectrum of the velocity field v_tilde has no nan values."""
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_v_tilde)))

    def test_source_lifetime_factor(self) -> None:
        """Test that the source lifetime factor is positive."""
        self.assertGreater(self.spectrum.source_lifetime_factor, 0)

    def test_spec_den_gw_scaling(self) -> None:
        """Test that the scaling factor of the GW spectral density is positive."""
        self.assertGreater(self.spectrum.spec_den_gw_scaling, 0)

    def test_spec_den_v_tilde(self) -> None:
        """Test that the spectral density of the velocity field has no nan values."""
        self.assertFalse(np.any(np.isnan(self.spectrum.spec_den_v_tilde)))

    def test_suppression_factor(self) -> None:
        """Test that the suppression factor is positive."""
        self.assertGreater(self.spectrum.suppression_factor, 0)

    def test_tau_end(self) -> None:
        """Test that the conformal time when the anisotropic stress turns off is positive."""
        self.assertGreater(self.spectrum.tau_end, 0)

    def test_tau_star(self) -> None:
        """Test that the conformal time when the anisotropic stress turns on is positive."""
        self.assertGreater(self.spectrum.tau_star, 0)

    def test_ubarf(self) -> None:
        """Test that the RMS fluid velocity is positive also for a custom nucleation type."""
        self.assertGreater(self.spectrum.ubarf, 0)
        self.assertGreater(self.spectrum.ubarf_custom_nucleation(nuc_type=DEFAULT_NUC_TYPE), 0)

    def test_tau_order(self) -> None:
        """Test that the anisotropic stress turns off after it turns on."""
        self.assertGreater(self.spectrum.tau_end, self.spectrum.tau_star)

    def test_z_cross_approx(self) -> None:
        """Test that the approximate crossover point of the low and intermediate frequency forms is positive."""
        self.assertGreater(self.spectrum.z_cross_approx, 0)


if __name__ == "__main__":
    unittest.main()
