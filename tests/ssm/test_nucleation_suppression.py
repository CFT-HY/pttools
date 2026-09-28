r"""Tests for the thermal suppression of bubble nucleation in the Sound Shell Model.

The suppression of :ajmi_2022:`\ ` changes the number density of bubbles, and therefore the mean bubble spacing
$R_{\ast} = \Lambda R_{\ast,0}$, but not the self-similar fluid profiles.
In the Sound Shell Model of :gw_pt_ssm:`\ ` the bubbles fill space, $n_b \langle \frac{4\pi}{3} v_w^3 T^3 \rangle = 1$,
so that $\bar{U}_f^2$ and the spectral densities as functions of $z = kR_{\ast}$ do not depend on $\Lambda$
(for a fixed lifetime distribution $\nu$).
Therefore, a spectrum computed from $\tilde{\beta}$ must be the same as one computed directly
from the resulting $r_{\ast} = \Lambda r_{\ast,0}$.
"""

import unittest

import numpy as np

from pttools.bubble import Bubble
from pttools.models.bag import BagModel
from pttools.ssm import SSMSpectrum
from pttools.ssm.nucleation import r_star0
from pttools.utils import assert_allclose


class NucleationSuppressionTest(unittest.TestCase):
    bubble: Bubble
    spectrum_beta: SSMSpectrum
    spectrum_r_star: SSMSpectrum

    @classmethod
    def setUpClass(cls) -> None:
        model = BagModel(alpha_n_min=0.01)
        cls.bubble = Bubble(model, v_wall=0.44, alpha_n=0.1)
        y = np.logspace(-1, 3, 300)
        cls.spectrum_beta = SSMSpectrum(cls.bubble, beta_tilde=100, y=y, low_k=False)
        cls.spectrum_r_star = SSMSpectrum(cls.bubble, r_star=cls.spectrum_beta.r_star, y=y, low_k=False)

    def test_lambda(self) -> None:
        lam = self.spectrum_beta.bubble_spacing_enlargement_factor
        self.assertGreater(lam, 1.)
        assert_allclose(self.spectrum_beta.r_star, lam * r_star0(100, self.bubble.v_wall))
        self.assertEqual(self.spectrum_r_star.bubble_spacing_enlargement_factor, 1.)

    def test_ubarf2(self) -> None:
        assert_allclose(self.spectrum_beta.ubarf2, self.spectrum_r_star.ubarf2)
        assert_allclose(self.spectrum_beta.ubarf_custom_nucleation(), self.spectrum_r_star.ubarf)

    def test_spec_den_v(self) -> None:
        assert_allclose(self.spectrum_beta.spec_den_v, self.spectrum_r_star.spec_den_v)

    def test_pow_gw(self) -> None:
        assert_allclose(self.spectrum_beta.pow_gw, self.spectrum_r_star.pow_gw)


if __name__ == "__main__":
    unittest.main()
