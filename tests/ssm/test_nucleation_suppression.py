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
from pttools.ssm.nucleation import nucleation_f, r_star0
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


class NucleationFTest(unittest.TestCase):
    r"""Tests for :func:`pttools.ssm.nucleation.nucleation_f`, :ajmi_2022:`\ ` eq. 50."""

    @staticmethod
    def top_hat(v_wall: float, v_sh: float, dT: float, hybrid: bool) -> tuple[np.ndarray, np.ndarray]:
        r"""A coarse profile with constant $\Delta T / T_n$ between the wall and the shock.

        The point at $\xi = v_\text{wall}$ on the outside of the wall carries $T_+$, as in the PTtools solutions.
        Hybrids also have a point at $\xi = v_\text{wall}$ on the inside, with a different temperature.
        """
        T_n = 1.
        xi = [0., 0.5 * v_wall]
        T = [0.9, 0.9]
        if hybrid:
            xi.append(v_wall)
            T.append(0.8)
        xi += [v_wall, 0.5 * (v_wall + v_sh), v_sh, v_sh * (1. + 1e-9), 1.]
        T += [T_n * (1. + dT)] * 3 + [T_n, T_n]
        return np.array(xi), np.array(T)

    def test_top_hat(self) -> None:
        r"""$f = ((v_\text{sh}/v_\text{wall})^3 - 1)(1 - e^{-\tilde\beta\Delta T/T_n})$ for a constant $\Delta T$."""
        v_wall, v_sh, dT, beta_tilde = 0.4, 0.55, 0.01, 100.
        expected = ((v_sh / v_wall) ** 3 - 1.) * (1. - np.exp(-beta_tilde * dT))
        for hybrid in (False, True):
            xi, T = self.top_hat(v_wall, v_sh, dT, hybrid)
            assert_allclose(nucleation_f(xi=xi, T=T, beta_tilde=beta_tilde, v_wall=v_wall), expected, rtol=1e-6)

    def test_bag_profile_small_vw(self) -> None:
        r"""Small-$v_\text{wall}$ limit :ajmi_2022:`\ ` eq. 71 (eq. 72 with $+c_s^2$ in the bracket)."""
        v_wall, alpha_n, beta_tilde = 0.05, 0.005, 10.
        bubble = Bubble(BagModel(alpha_n_min=0.001), v_wall=v_wall, alpha_n=alpha_n)
        bubble.solve()
        cs = 1 / np.sqrt(3)
        expected = 3 * alpha_n * beta_tilde * (1 + cs**2) / (4 * cs**2 * (1 - 3 * v_wall**2) ** 2) \
            * (cs**2 + v_wall**2 * (2 * v_wall - 3 * cs) / cs)
        assert_allclose(nucleation_f(xi=bubble.xi, T=bubble.T, beta_tilde=beta_tilde, v_wall=v_wall), expected,
                        rtol=0.005)
