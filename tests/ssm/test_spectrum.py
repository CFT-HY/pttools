"""Tests for the SSMSpectrum class"""

import os.path
import unittest

import numpy as np

from pttools.bubble import Bubble
from pttools.models import ConstCSModel
from pttools.ssm import DEFAULT_NUC_TYPE, SSMSpectrum
from tests.utils import TEST_JSON_PATH


class SSMSpectrumTest(unittest.TestCase):
    """Tests for the SSMSpectrum class"""
    @classmethod
    def setUpClass(cls):
        model = ConstCSModel(css2=1/3-0.01, csb2=1/3-0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
        cls.spectrum = SSMSpectrum(bubble, beta_tilde=1.23)

    def test_beta(self):
        self.assertTrue(np.isfinite(self.spectrum.beta(1.23)))

    def test_bubble_spacing_enlargement_factor(self):
        self.assertGreater(self.spectrum.bubble_spacing_enlargement_factor, 1.)

    def test_dilution_of_e(self):
        self.assertGreater(self.spectrum.dilution_of_e, 0)
        self.assertLessEqual(self.spectrum.dilution_of_e, 1)

    def test_eta_ratio(self):
        self.assertGreater(self.spectrum.eta_ratio, 1)

    def test_export(self):
        self.spectrum.export(os.path.join(TEST_JSON_PATH, "ssm-spectrum.json"))

    def test_H_star_eta_star(self):
        self.assertGreater(self.spectrum.H_star_eta_star, 0)

    def test_H_star_tau_nl(self):
        self.assertGreater(self.spectrum.H_star_tau_nl, 0)

    def test_H_star_tau_sh(self):
        self.assertGreater(self.spectrum.H_star_tau_sh, 0)

    def test_H_star_tau_v(self):
        self.assertGreater(self.spectrum.H_star_tau_v, 0)

    def test_H_star_tau_v_old(self):
        self.assertGreater(self.spectrum.H_star_tau_v_old, 0)

    def test_hx(self):
        self.assertGreater(self.spectrum.hx, 0)

    def test_k_peak_eta_star(self):
        self.assertGreater(self.spectrum.k_peak_eta_star, 0)

    def test_J(self):
        self.assertGreater(self.spectrum.J, 0)

    def test_nucleation_f(self):
        self.assertGreater(self.spectrum.nucleation_f, 0)

    def test_pow_gw(self):
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_gw)))

    def test_pow_gw_expanded(self):
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_gw_expanded)))

    def test_pow_gw_int(self):
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_gw_int)))

    def test_pow_gw_low(self):
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_gw_low)))

    def test_pow_gw_ssm(self):
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_gw_ssm)))

    def test_pow_v(self):
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_v)))

    def test_pow_v_tilde(self):
        self.assertFalse(np.any(np.isnan(self.spectrum.pow_v_tilde)))

    def test_source_lifetime_factor(self):
        self.assertGreater(self.spectrum.source_lifetime_factor, 0)

    def test_spec_den_gw_scaling(self):
        self.assertGreater(self.spectrum.spec_den_gw_scaling, 0)

    def test_spec_den_v_tilde(self):
        self.assertFalse(np.any(np.isnan(self.spectrum.spec_den_v_tilde)))

    def test_suppression_factor(self):
        self.assertGreater(self.spectrum.suppression_factor, 0)

    def test_tau_end(self):
        self.assertGreater(self.spectrum.tau_end, 0)

    def test_tau_star(self):
        self.assertGreater(self.spectrum.tau_star, 0)

    def test_ubarf(self):
        self.assertGreater(self.spectrum.ubarf, 0)
        self.assertGreater(self.spectrum.ubarf_custom_nucleation(nuc_type=DEFAULT_NUC_TYPE), 0)

    def test_tau_order(self):
        self.assertGreater(self.spectrum.tau_end, self.spectrum.tau_star)

    def test_z_cross_approx(self):
        self.assertGreater(self.spectrum.z_cross_approx, 0)


if __name__ == "__main__":
    unittest.main()
