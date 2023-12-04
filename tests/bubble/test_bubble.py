import numpy as np
import unittest

from pttools.models.bag import BagModel
from pttools.bubble.bubble import Bubble


class BubbleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        model = BagModel(a_s=1.1, a_b=1, V_s=1)
        bubble = Bubble(model, v_wall=0.5, alpha_n=0.1)
        bubble.solve()
        cls.bubble = bubble

    def test_ebar(self):
        ebar = self.bubble.ebar
        self.assertGreater(ebar, 0)

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
