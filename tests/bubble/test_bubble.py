"""Unit tests for the properties of a bubble"""

import os.path
import unittest

import numpy as np

from pttools.bubble import DEFAULT_N_XI, Bubble
from pttools.models.bag import BagModel
from tests.utils import TEST_JSON_PATH


class BubbleTest(unittest.TestCase):
    """Unit tests for the properties of a bubble"""
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = BagModel(a_s=1.1, a_b=1, V_s=1)
        cls.bubble = Bubble(cls.model, v_wall=0.5, alpha_n=0.1)

    def test_ebar(self):
        ebar = self.bubble.ebar
        self.assertGreater(ebar, 0)

    def test_export(self):
        self.bubble.export(os.path.join(TEST_JSON_PATH, "bubble.json"))

    def test_kappa(self):
        kappa = self.bubble.kappa
        self.assertGreater(kappa, 0)
        self.assertLess(kappa, 1)

    def test_bva_kinetic_energy_fraction(self):
        kef = self.bubble.kinetic_energy_fraction
        self.assertGreater(kef, 0)
        self.assertLess(kef, 1)

    def test_va_kinetic_energy_fraction(self):
        kef = self.bubble.va_kinetic_energy_fraction
        self.assertGreater(kef, 0)
        self.assertLess(kef, 1)

    def test_va_kinetic_energy_density(self):
        ked = self.bubble.va_kinetic_energy_density
        self.assertGreater(ked, 0)

    def test_mean_adiabatic_index(self):
        mabi = self.bubble.mean_adiabatic_index
        self.assertGreater(mabi, 0)

    def test_omega(self):
        omega = self.bubble.omega
        self.assertGreater(omega, 0)
        self.assertLess(omega, 1)

    def test_va_thermal_energy_density(self):
        ted = self.bubble.va_thermal_energy_density_diff
        self.assertGreater(ted, 0)

    def test_bva_thermal_energy_fraction(self):
        tef = self.bubble.thermal_energy_fraction
        self.assertGreater(tef, 0)

    def test_va_thermal_energy_fraction(self):
        tef = self.bubble.va_thermal_energy_fraction
        self.assertGreater(tef, 0)

    def test_va_trace_anomaly(self):
        trace_anomaly = self.bubble.va_trace_anomaly_diff
        self.assertTrue(np.isfinite(trace_anomaly))

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
