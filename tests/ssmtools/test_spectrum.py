import unittest

from pttools.bubble import Bubble
from pttools.models import ConstCSModel
from pttools.ssmtools import Spectrum


class SpectrumTest(unittest.TestCase):
    @staticmethod
    def test_spectrum():
        model = ConstCSModel(css2=1/3-0.01, csb2=1/3-0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
        Spectrum(bubble)


if __name__ == "__main__":
    unittest.main()
