r"""Tests for the $\Omega_{\text{gw},0}$ factors and the constants they are computed from"""

import unittest

import numpy as np

from pttools.omgw0 import const
from pttools.omgw0.factors import F_gw0

#: $a$, the radiation constant in $\frac{\text{J}}{\text{m}^3 \text{K}^4}$
A_RAD_REF: float = 7.565733e-16

F_GW0_REF: float = 3.57e-5
r"""$F_{\text{gw},0}$ for ${g}_\ast = g_{s\ast} = 100$
$$F_{\text{gw},0} = (3.57 \pm 0.05) \cdot 10^{-5} \left( \frac{100}{{g}_\ast} \right)^\frac{1}{3}$$
:caprini_2020:`\ `, eq. 20.
This assumes $h = 0.678$ of :planck_2015:`\ `.
"""

#: Uncertainty of :py:data:`F_GW0_REF`, :caprini_2020:`\ ` eq. 20
F_GW0_REF_ERR: float = 0.05e-5

#: $h$, dimensionless reduced Hubble constant :planck_2015:`\ `
H_REF = 0.678

#: $\Omega_{\gamma,0} h^2$, Planck 2018 / PDG value
OMEGA_PHOTON_H2_REF: float = 2.473e-5


class ConstTest(unittest.TestCase):
    r"""Tests for the constants from which $F_{\text{gw},0}$ is computed"""

    def test_a_rad(self):
        """The radiation constant should match the CODATA value"""
        self.assertAlmostEqual(const.A_RADIATION / A_RAD_REF, 1, places=6)

    def test_gs0(self):
        r"""The computed $g_{s0}$ should match the $3.91$ of :caprini_2020:`\ ` p. 12"""
        self.assertAlmostEqual(const.gs0(g0=2, n_nu=3), 3.91, places=2)
        self.assertAlmostEqual(const.GS0, 3.91, delta=0.0199)

    def test_omega_photon_h2(self):
        r"""$\Omega_{\gamma,0} h^2$ should match the literature value"""
        self.assertAlmostEqual(const.OMEGA_PHOTON_H2 / OMEGA_PHOTON_H2_REF, 1, places=4)

    def test_omega_photon_h_scaling(self):
        r"""$\Omega_{\gamma,0}$ should be $\Omega_{\gamma,0} h^2$ divided by $h^2$"""
        self.assertAlmostEqual(const.OMEGA_PHOTON * const.H2 / const.OMEGA_PHOTON_H2, 1, places=12)

    def test_h0_hz(self):
        r"""$H_0$ should be $h {H}_{100}$"""
        self.assertAlmostEqual(const.H0_HZ / (const.H * const.H0_100_HZ), 1, places=12)


class FGw0Test(unittest.TestCase):
    r"""Tests for $F_{\text{gw},0}$"""

    def test_reference_value(self):
        r"""The computed $F_{\text{gw},0}$ should be consistent with the value of :caprini_2020:`\ ` eq. 20

        This is the test that ties the computed constants to the literature.
        """
        computed = F_gw0(g_star=100., om_gamma0=const.OMEGA_PHOTON_H2 / (H_REF**2))
        self.assertAlmostEqual(computed, F_GW0_REF, delta=F_GW0_REF_ERR)
        # The agreement should in fact be better than the uncertainty of the reference value.
        self.assertAlmostEqual(computed / F_GW0_REF, 1, delta=0.0067)

    def test_gs_star_default(self):
        r"""Omitting $g_{s\ast}$ should be equivalent to setting $g_{s\ast} = {g}_\ast$"""
        for g_star in (10., 100., 106.75):
            with self.subTest(g_star=g_star):
                self.assertEqual(F_gw0(g_star=g_star), F_gw0(g_star=g_star, gs_star=g_star))

    def test_g_star_scaling(self):
        r"""For $g_{s\ast} = {g}_\ast$ the scaling should be $\left( \frac{100}{{g}_\ast} \right)^\frac{1}{3}$"""
        g_star = np.array([1., 10., 100., 106.75, 1000.])
        expected = F_gw0(g_star=100.) * (100 / g_star)**(1/3)
        np.testing.assert_allclose(F_gw0(g_star=g_star), expected, rtol=1e-12)

    def test_h_independence(self):
        r"""$h^2 F_{\text{gw},0}$ should not depend on the value of $h$

        $\Omega_{\gamma,0} = \frac{\Omega_{\gamma,0} h^2}{h^2}$,
        so the $h$ of $\Omega_{\gamma,0}$ cancels out in the observable $h^2 \Omega_{\text{gw},0}$.
        """
        values = [
            h**2 * F_gw0(g_star=100., om_gamma0=const.OMEGA_PHOTON_H2 / h**2)
            for h in (0.674, 0.678, 0.73)
        ]
        for value in values[1:]:
            self.assertAlmostEqual(value / values[0], 1, places=12)

    def test_gs_star_dependence(self):
        r"""$F_{\text{gw},0}$ should scale as $g_{s\ast}^{-\frac{4}{3}}$"""
        ratio = F_gw0(g_star=100., gs_star=50.) / F_gw0(g_star=100., gs_star=100.)
        self.assertAlmostEqual(ratio, 2**(4/3), places=12)


if __name__ == "__main__":
    unittest.main()
