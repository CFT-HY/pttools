"""Unit tests for the properties of a bubble"""

import os.path
import unittest

import numpy as np

from pttools.bubble import DEFAULT_N_XI, Bubble
from pttools.models.bag import BagModel
from tests.utils import TEST_JSON_PATH


class BubbleTest(unittest.TestCase):
    """Unit tests for the properties of a bubble"""
    model: BagModel
    bubble: Bubble

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = BagModel(a_s=1.1, a_b=1, V_s=1)
        cls.bubble = Bubble(cls.model, v_wall=0.5, alpha_n=0.1)

    def test_export(self):
        self.bubble.export(os.path.join(TEST_JSON_PATH, "bubble.json"))

    def test_vp_vm_tilde_ratio_giese(self):
        self.assertGreater(self.bubble.vp_vm_tilde_ratio_giese, 0)

    @unittest.expectedFailure
    def test_v_mu(self):
        # Todo: fix this
        self.assertGreater(self.bubble.v_mu, 0)

    @unittest.expectedFailure
    def test_v_wall_1(self):
        Bubble(self.model, v_wall=1, alpha_n=0.1)

    def test_v_wall_high(self):
        Bubble(self.model, v_wall=0.95, alpha_n=0.1)

    def test_v_wall_low(self):
        Bubble(self.model, v_wall=0.01, alpha_n=0.1)

    def test_v_wall_low_custom_low_n_xi(self):
        """Test that the warning message for low v_wall and low n_xi is generated properly"""
        Bubble(self.model, v_wall=0.01, alpha_n=0.1, n_xi=DEFAULT_N_XI // 2)

    # -----
    # Averaged
    # -----
    def test_e_bar(self):
        self.assertGreater(self.bubble.e_bar, 0)

    def test_kappa(self):
        kappa = self.bubble.kappa
        self.assertGreater(kappa, 0)
        self.assertLess(kappa, 1)

    def test_kappa_giese(self):
        self.assertGreater(self.bubble.kappa_giese, 0)

    def test_g_star(self):
        self.assertGreater(self.bubble.g_star, 0)

    def test_gs_star(self):
        self.assertGreater(self.bubble.gs_star, 0)

    def test_mean_adiabatic_index(self):
        self.assertGreater(self.bubble.mean_adiabatic_index, 0)

    def test_nu_gdh2024(self):
        self.assertAlmostEqual(self.bubble.nu_gdh2024, 0)

    def test_omega(self):
        omega = self.bubble.omega
        self.assertGreater(omega, 0)
        self.assertLess(omega, 1)

    def test_omega_barotropic(self):
        omega = self.bubble.omega_barotropic
        self.assertGreater(omega, 0)
        self.assertLess(omega, 1)

    def test_T_star(self):
        self.assertGreater(self.bubble.T_star, 0)

    def test_ubarf(self):
        self.assertGreater(self.bubble.ubarf, 0)
        self.assertGreater(self.bubble.ubarf2, 0)

    def test_w_bar(self):
        self.assertGreater(self.bubble.w_bar, 0)

    # -----
    # bva = bubble volume averaged
    # -----
    def test_entropy_density_diff(self):
        self.assertGreater(self.bubble.entropy_density_diff, 0)

    def test_entropy_density_diff_relative(self):
        self.assertGreater(self.bubble.entropy_density_diff_relative, 0)

    def test_kinetic_energy_density(self):
        self.assertGreater(self.bubble.kinetic_energy_density, 0)

    def test_kinetic_energy_fraction(self):
        kef = self.bubble.kinetic_energy_fraction
        self.assertGreater(kef, 0)
        self.assertLess(kef, 1)

    def test_thermal_energy_density(self):
        self.assertGreater(self.bubble.thermal_energy_density, 0)

    def test_thermal_energy_fraction(self):
        tef = self.bubble.thermal_energy_fraction
        self.assertGreater(tef, 0)
        # self.assertLess(tef, 1)

    def test_trace_anomaly(self):
        self.assertTrue(np.isfinite(self.bubble.trace_anomaly))

    # -----
    # va = volume averaged
    # -----
    def test_va_enthalpy_density(self):
        self.assertGreater(self.bubble.va_enthalpy_density, 0)

    def test_va_entropy_density_diff(self):
        self.assertGreater(self.bubble.va_entropy_density_diff, 0)

    def test_va_entropy_density_diff_relative(self):
        self.assertGreater(self.bubble.va_entropy_density_diff_relative, 0)

    def test_va_kinetic_energy_fraction(self):
        kef = self.bubble.va_kinetic_energy_fraction
        self.assertGreater(kef, 0)
        self.assertLess(kef, 1)

    def test_va_kinetic_energy_density(self):
        ked = self.bubble.va_kinetic_energy_density
        self.assertGreater(ked, 0)

    def test_va_thermal_energy_density_diff(self):
        ted = self.bubble.va_thermal_energy_density_diff
        self.assertGreater(ted, 0)

    def test_va_thermal_energy_fraction(self):
        tef = self.bubble.va_thermal_energy_fraction
        self.assertGreater(tef, 0)
        self.assertLess(tef, 1)

    def test_va_trace_anomaly_diff(self):
        trace_anomaly = self.bubble.va_trace_anomaly_diff
        self.assertTrue(np.isfinite(trace_anomaly))
