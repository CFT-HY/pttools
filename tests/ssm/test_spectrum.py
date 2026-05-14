"""Tests for the SSMSpectrum class"""

import os.path
import unittest

from pttools.bubble import Bubble
from pttools.models import ConstCSModel
from pttools.ssm import SSMSpectrum
from tests.utils import TEST_JSON_PATH


class SSMSpectrumTest(unittest.TestCase):
    """Tests for the SSMSpectrum class"""
    @classmethod
    def setUpClass(cls):
        model = ConstCSModel(css2=1/3-0.01, csb2=1/3-0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
        cls.spectrum = SSMSpectrum(bubble)

    def test_export(self):
        self.spectrum.export(os.path.join(TEST_JSON_PATH, "ssm-spectrum.json"))

    def test_tau_order(self):
        self.assertGreater(self.spectrum.tau_end, self.spectrum.tau_star)


if __name__ == "__main__":
    unittest.main()
