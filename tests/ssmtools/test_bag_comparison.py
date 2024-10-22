import unittest

import numpy as np

from pttools.bubble import Bubble
from pttools.bubble.quantities import de_from_w_bag
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
        cls.z = cls.spectra[0].y

    def test_de(self):
        # The arrays have different sizes and cannot therefore be combined to a 2D array
        de_bag = [
            de_from_w_bag(w=bubble.w, xi=bubble.xi, v_wall=bubble.v_wall, alpha_n=bubble.alpha_n)
            for bubble in self.bubbles
        ]
        e = [
            bubble.model.e(bubble.w, bubble.phase)
            for bubble in self.bubbles
        ]
        de = [ei - ei[-1] for ei in e]
        for de_i, de_bag_i in zip(de, de_bag):
            assert_allclose(de_i, de_bag_i)

    def test_a2(self):
        a2_old = np.array([
            ssmtools.a2_e_conserving_bag(
                self.z, v_wall=self.V_WALLS[i], alpha_n=self.ALPHA_NS[i],
                v_ip=bubble.v, w_ip=bubble.w, xi=bubble.xi,
                v_sh=bubble.v_sh
            )[0]
            for i, bubble in enumerate(self.bubbles)
        ])
        a2_new = np.array([
            ssmtools.a2_e_conserving(bubble, z=self.z, cs=ssmtools.CS0)[0]
            for bubble in self.bubbles
        ])
        # a2_new2 = np.array([spectrum.a2 for spectrum in self.spectra])
        assert_allclose(a2_new, a2_old)
        # assert_allclose(a2_new2, a2_old)

    def test_spec_den_v(self):
        old = np.array([
            ssmtools.spec_den_v_bag(self.z, (self.V_WALLS[i], self.ALPHA_NS[i]))
            for i in range(self.V_WALLS.size)
        ])
        new = np.array([spectrum.spec_den_v for spectrum in self.spectra])
        assert_allclose(new, old, rtol=0.283)

    def test_gw(self):
        old = np.array([
            ssmtools.power_gw_scaled_bag(self.z, (self.V_WALLS[i], self.ALPHA_NS[i]))
            for i in range(self.V_WALLS.size)
        ])
        new = np.array([spectrum.pow_gw for spectrum in self.spectra])
        assert_allclose(new, old, rtol=0.519)


if __name__ == "__main__":
    unittest.main()
