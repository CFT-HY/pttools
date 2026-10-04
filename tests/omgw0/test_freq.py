r"""Tests for the frequency conversion functions of $\Omega_{\text{gw},0}$."""

import math
import unittest

import numpy as np
from scipy import constants

from pttools.omgw0 import const, freq


def f_star0_exact(T_star: float, ge_star: float, gs_star: float) -> float:
    r"""$f_{\ast,0} = \frac{H_{\ast}}{2\pi} \frac{a_{\ast}}{a_0}$ in Hz computed from the physical constants.

    $$H_{\ast}^2 = \frac{8 \pi G}{3 c^2} e_{\ast}, \quad
    e_{\ast} = \frac{\pi^2}{30} g_{e\ast} \frac{(k_{\text{B}} T_{\ast})^4}{(\hbar c)^3}, \quad
    \frac{a_{\ast}}{a_{0}} = \frac{T_{0}}{T_{\ast}} \left( \frac{g_{s0}}{g_{s\ast}} \right)^\frac{1}{3}$$

    :param T_star: $T_\ast$, temperature at the time of GW production in GeV
    :param ge_star: $g_{e\ast}$, degrees of freedom for energy density at the time of GW production
    :param gs_star: $g_{s\ast}$, degrees of freedom for entropy at the time of GW production
    :return: $f_{\ast,0}$ in Hz
    """
    kT_star = T_star * 1e9 * constants.e
    e_star = math.pi**2 / 30 * ge_star * kT_star**4 / (constants.hbar * constants.c)**3
    H_star = math.sqrt(8 * math.pi * constants.G * e_star / (3 * constants.c**2))
    a_ratio = (const.GS0 / gs_star)**(1/3) * const.T_CMB * constants.k / kT_star
    return H_star * a_ratio / (2 * math.pi)


F_STAR0_REF: float = 2.6e-6
r"""$f_{\ast,0,\text{ref}}$, $f_{\ast,0}$ in Hz for $T_\ast = 100 \text{ GeV}$ and $g_{e\ast} = g_{s\ast} = 100$

:caprini_2020:`\ ` eq. 31,
:gowling_2021:`\ ` eq. 2.13,
:gowling_2023:`\ ` eq. 2.9,
:hindmarsh_2017:`\ ` p. 10 (as $26 \mu\text{Hz}$ for $z = 10$).
"""

#: Uncertainty of :py:data:`F_STAR0_REF` due to its rounding to two significant digits
F_STAR0_REF_ERR: float = 0.05e-6

#: Relative tolerance for the differences between the physical constants of PTtools and SciPy
RTOL_CONSTANTS: float = 1e-8


class FStar0Test(unittest.TestCase):
    r"""Tests for $f_{\ast,0}$."""

    PARAMS: tuple[tuple[float, float, float], ...] = (
        (100., 100., 100.),
        (0.15, 50., 46.5),
        (200., 106.75, 106.75),
        (2.2, 80.2, 79.5),
        (1000., 300., 200.),
    )

    def test_literature_value(self) -> None:
        r"""Test that $f_{\ast,0}$ at $T_\ast = 100 \text{ GeV}$ and $g_\ast = 100$ is the value of the articles."""
        self.assertAlmostEqual(freq.f_star0(T_star=100., ge_star=100.), F_STAR0_REF, delta=F_STAR0_REF_ERR)
        # The older value of g_s0 gives a value that is equally close to that of the articles.
        self.assertAlmostEqual(freq.f_star0(T_star=100., ge_star=100., gs0=3.91), F_STAR0_REF, delta=F_STAR0_REF_ERR)

    def test_exact(self) -> None:
        r"""Test $f_{\ast,0}$ against a computation using the physical constants of SciPy.

        The tolerance is set by the rounding of :py:data:`pttools.omgw0.const.H_BAR`,
        as $f_{\ast,0} \propto \hbar^{-\frac{3}{2}}$.
        """
        for params in self.PARAMS:
            with self.subTest(params=params):
                np.testing.assert_allclose(freq.f_star0(*params), f_star0_exact(*params), rtol=RTOL_CONSTANTS)

    def test_array(self) -> None:
        r"""Test that $f_{\ast,0}$ can be computed for arrays."""
        T_star, ge_star, gs_star = (np.array(param) for param in zip(*self.PARAMS, strict=True))
        np.testing.assert_allclose(
            freq.f_star0(T_star, ge_star, gs_star),
            [f_star0_exact(*params) for params in self.PARAMS],
            rtol=RTOL_CONSTANTS
        )

    def test_literature_form(self) -> None:
        r"""Test that $f_{\ast,0}$ reduces to the form with $\left( \frac{g_{\ast}}{100} \right)^\frac{1}{6}$."""
        T_star = np.array([0.1, 100., 1e4])
        f_star0_ref = freq.f_star0(100., 100.)
        for g_star in (10., 100., 106.75):
            with self.subTest(g_star=g_star):
                expected = f_star0_ref * T_star / 100 * (g_star / 100)**(1/6)
                np.testing.assert_allclose(freq.f_star0(T_star, g_star, g_star), expected, rtol=1e-12)
                np.testing.assert_allclose(freq.f_star0(T_star, g_star), expected, rtol=1e-12)

    def test_z_inverse(self) -> None:
        """Test that the conversion from frequencies to wavenumbers is the inverse of the conversion to frequencies."""
        z = np.logspace(-1, 3, 5)
        T_star, ge_star, gs_star, r_star = 200., 106.75, 100., 0.1
        f = freq.f(z, r_star=r_star, f_star0=freq.f_star0(T_star, ge_star, gs_star))
        np.testing.assert_allclose(f, z * freq.f0(r_star, T_star, ge_star, gs_star), rtol=1e-14)
        np.testing.assert_allclose(
            freq.z(f, T_star=T_star, r_star=r_star, ge_star=ge_star, gs_star=gs_star), z, rtol=1e-14)
