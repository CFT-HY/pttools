"""Tests for the Spectrum class"""

import os.path
import unittest

import numpy as np

from pttools.bubble import Bubble
from pttools.models import ConstCSModel
from pttools.omgw0 import Spectrum
from tests.utils import TEST_JSON_PATH


class SpectrumTest(unittest.TestCase):
    """Tests for the Spectrum class"""
    @classmethod
    def setUpClass(cls):
        model = ConstCSModel(css2=1/3-0.01, csb2=1/3-0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
        cls.spectrum = Spectrum(bubble, r_star=0.1)

    def test_export(self):
        self.spectrum.export(os.path.join(TEST_JSON_PATH, "spectrum.json"))

    def test_noise(self):
        self.assertGreater(self.spectrum.signal_to_noise_ratio(), 0)

    def test_noise_instrument(self):
        self.assertGreater(self.spectrum.signal_to_noise_ratio_instrument(), 0)

    def test_peak(self):
        peak = self.spectrum.omgw0_peak()
        self.assertGreater(peak[0], 0)
        self.assertGreater(peak[1], 0)
        self.assertLess(peak[1], 1)

    def test_R_star(self):
        """Test that $0 < R_* < 1 \text{mm}$"""
        self.assertGreater(self.spectrum.R_star, 0)
        self.assertLess(self.spectrum.R_star, 1e-3)

    def test_spectrum(self):
        self.assertEqual(np.isnan(self.spectrum.omgw0()).sum(), 0)

    def test_total(self):
        val = self.spectrum.omgw0_total()
        ref = np.trapezoid(y=self.spectrum.omgw0(), x=self.spectrum.f())
        self.assertAlmostEqual(val, ref)


if __name__ == "__main__":
    unittest.main()
