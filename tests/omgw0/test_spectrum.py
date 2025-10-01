"""Tests for the Spectrum class"""

import unittest

import numpy as np

from pttools.bubble import Bubble
from pttools.models import ConstCSModel
from pttools.omgw0 import Spectrum


class SpectrumTest(unittest.TestCase):
    """Tests for the Spectrum class"""
    @classmethod
    def setUpClass(cls):
        model = ConstCSModel(css2=1/3-0.01, csb2=1/3-0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
        cls.spectrum = Spectrum(bubble, r_star=0.1)

    def test_noise(self):
        self.spectrum.signal_to_noise_ratio()

    def test_noise_instrument(self):
        self.spectrum.signal_to_noise_ratio_instrument()

    def test_peak(self):
        self.spectrum.omgw0_peak()

    def test_R_star(self):
        self.spectrum.R_star()

    def test_total(self):
        val = self.spectrum.omgw0_total()
        ref = np.trapezoid(y=self.spectrum.omgw0(), x=self.spectrum.f())
        self.assertAlmostEqual(val, ref)


if __name__ == "__main__":
    unittest.main()
