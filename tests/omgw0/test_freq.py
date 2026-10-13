r"""Tests for the frequency conversion functions of $\Omega_{\text{gw},0}$."""

import math
import typing as tp
import unittest

import numpy as np
import pytest
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

    PARAMS: tp.ClassVar[tuple[tuple[float, float, float], ...]] = (
        (100., 100., 100.),
        (0.15, 50., 46.5),
        (200., 106.75, 106.75),
        (2.2, 80.2, 79.5),
        (1000., 300., 200.),
    )

    def test_literature_value(self) -> None:
        r"""Test that $f_{\ast,0}$ at $T_\ast = 100 \text{ GeV}$ and $g_\ast = 100$ is the value of the articles."""
        assert freq.f_star0(T_star=100., ge_star=100.) == pytest.approx(F_STAR0_REF, abs=F_STAR0_REF_ERR)
        # The older value of g_s0 gives a value that is equally close to that of the articles.
        assert freq.f_star0(T_star=100., ge_star=100., gs0=3.91) == pytest.approx(F_STAR0_REF, abs=F_STAR0_REF_ERR)

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
            freq.z_full(f, T_star=T_star, r_star=r_star, ge_star=ge_star, gs_star=gs_star), z, rtol=1e-14)
        np.testing.assert_allclose(
            freq.z(f, r_star=r_star, f_star0=freq.f_star0(T_star, ge_star, gs_star)), z, rtol=1e-14)


class ZRangeTest(unittest.TestCase):
    r"""Tests for the range of $z$ that covers a frequency range for multiple parameter combinations."""

    F_MIN: tp.ClassVar[float] = 1e-5
    F_MAX: tp.ClassVar[float] = 1e-1

    def test_scalar(self) -> None:
        """For a single combination, the range should be the conversion of the frequency limits."""
        params = {"T_star": 200., "r_star": 0.1, "ge_star": 106.75, "gs_star": 100.}
        z_min, z_max = freq.z_range(self.F_MIN, self.F_MAX, **params)
        assert z_min == freq.z_full(self.F_MIN, **params)
        assert z_max == freq.z_full(self.F_MAX, **params)

    def test_gs_star_default(self) -> None:
        r"""If $g_{s\ast}$ is not given, $g_{s\ast} = g_{e\ast}$ should be assumed."""
        assert freq.z_range(self.F_MIN, self.F_MAX, ge_star=80.) \
            == freq.z_range(self.F_MIN, self.F_MAX, ge_star=80., gs_star=80.)

    def test_arrays(self) -> None:
        """The range should cover the frequency range for all combinations, and be the tightest such range."""
        T_star = np.array([0.15, 100., 1000., 50.])
        r_star = np.array([0.01, 0.3, 0.1, 0.5])
        gs_star = np.array([10., 100., 106.75, 90.])
        ge_star = 100.
        z_min, z_max = freq.z_range(
            self.F_MIN, self.F_MAX, T_star=T_star, r_star=r_star, ge_star=ge_star, gs_star=gs_star)
        f_mins = []
        f_maxs = []
        for T_star_i, r_star_i, gs_star_i in zip(T_star, r_star, gs_star, strict=True):
            f_star0 = freq.f_star0(T_star_i, ge_star, gs_star_i)
            f_mins.append(freq.f(z_min, r_star=r_star_i, f_star0=f_star0))
            f_maxs.append(freq.f(z_max, r_star=r_star_i, f_star0=f_star0))
        # The tolerance accounts for the rounding of the conversion back to frequencies.
        assert max(f_mins) <= self.F_MIN * (1 + 1e-14)
        assert min(f_maxs) >= self.F_MAX * (1 - 1e-14)
        assert max(f_mins) == pytest.approx(self.F_MIN, rel=1e-14)
        assert min(f_maxs) == pytest.approx(self.F_MAX, rel=1e-14)

    def test_arrays_and_scalars(self) -> None:
        """An array of length one should give the same range as the corresponding scalar."""
        assert freq.z_range(self.F_MIN, self.F_MAX, T_star=np.array([200.]), r_star=0.2) \
            == freq.z_range(self.F_MIN, self.F_MAX, T_star=200., r_star=0.2)

    def test_invalid(self) -> None:
        """Invalid frequency ranges and arrays of different lengths or dimensions should raise an error."""
        with pytest.raises(ValueError, match="f_min <= f_max"):
            freq.z_range(self.F_MAX, self.F_MIN)
        with pytest.raises(ValueError, match="f_min <= f_max"):
            freq.z_range(0., self.F_MAX)
        with pytest.raises(ValueError, match="equal length"):
            freq.z_range(self.F_MIN, self.F_MAX, T_star=np.array([100., 200.]), r_star=np.array([0.1, 0.2, 0.3]))
        with pytest.raises(ValueError, match="1D array"):
            freq.z_range(self.F_MIN, self.F_MAX, T_star=np.array([[100., 200.]]))
        with pytest.raises(ValueError, match="finite and positive"):
            freq.z_range(self.F_MIN, self.F_MAX, T_star=np.array([100., np.nan]))
