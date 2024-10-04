import unittest

import numpy as np

from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.ssmtools import Spectrum
from pttools import ssmtools
from tests.utils.assertions import assert_allclose


class SpectrumTest(unittest.TestCase):
    V_WALLS: np.ndarray = np.array([0.5, 0.7, 0.77])
    ALPHA_NS: np.ndarray = np.array([0.578, 0.151, 0.091])

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = BagModel(a_s=1.1, a_b=1, V_s=2)
        cls.bubbles = [
            Bubble(cls.model, v_wall=cls.V_WALLS[i], alpha_n=cls.ALPHA_NS[i])
            for i in range(cls.V_WALLS.size)
        ]
        cls.spectra = [Spectrum(bubble) for bubble in cls.bubbles]
        cls.z = cls.spectra[0].z

    def test_a2(self):
        a2_old = np.array([
            ssmtools.a2_e_conserving_bag(self.z, v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i])[0]
            for i in range(self.V_WALLS.size)
        ])
        a2_new = np.array([spectrum.a2 for spectrum in self.spectra])

    def test_spec_den_v(self):
        old = np.array([
            ssmtools.spec_den_v_bag(self.z, (self.V_WALLS[i], self.ALPHA_NS[i]))
            for i in range(self.V_WALLS.size)
        ])
        new = np.array([spectrum.spec_den_v for spectrum in self.spectra])
        assert_allclose(new, old, rtol=0.1)

    def test_gw(self):
        old = np.array([
            ssmtools.power_gw_scaled_bag(self.z, (self.V_WALLS[i], self.ALPHA_NS[i]))
            for i in range(self.V_WALLS.size)
        ])
        new = np.array([spectrum.pow_gw for spectrum in self.spectra])
        assert_allclose(new, old, rtol=0.1)


if __name__ == "__main__":
    unittest.main()
