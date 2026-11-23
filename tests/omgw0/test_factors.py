r"""Tests for the $\Omega_{\text{gw},0}$ factors and the constants they are computed from."""

import math
import unittest

import numpy as np
import pytest
from uncertainties import ufloat, unumpy

from pttools.omgw0 import const
from pttools.omgw0.dof import ge0, gs0
from pttools.omgw0.factors import H_FACTOR_SI, H_SI, F_gw0_h2, H, a_ratio
from pttools.type_hints import UFloat, UFloatArr1D

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
H_REF: float = 0.678

#: $F_{\text{gw},0} h^2$, obtained from :py:data:`F_GW0_REF` with the $h$ assumed by :caprini_2020:`\ `
F_GW0_H2_REF: float = F_GW0_REF * H_REF**2

#: Uncertainty of :py:data:`F_GW0_H2_REF`
F_GW0_H2_REF_ERR: float = F_GW0_REF_ERR * H_REF**2

#: $\Omega_{\gamma,0} h^2$, Planck 2018 / PDG value
OMEGA_PHOTON_H2_REF: float = 2.473e-5


class ARatioTest(unittest.TestCase):
    r"""Tests for $\frac{a_{\ast}}{a_0}$."""

    @staticmethod
    def test_uncertainty() -> None:
        r"""The uncertainty of $T_0$ should propagate to $\frac{a_{\ast}}{a_0} \propto T_0$."""
        result: UFloat = a_ratio(T0=const.T_CMB_U)
        assert result.nominal_value == a_ratio()
        assert result.std_dev / result.nominal_value \
            == pytest.approx(const.T_CMB_U.std_dev / const.T_CMB, rel=1e-12)


class DofTest(unittest.TestCase):
    """Tests for the degrees of freedom today."""

    @staticmethod
    def test_uncertainty() -> None:
        r"""An uncertainty of $N_\nu$ should propagate to $g_{e0}$ and $g_{s0}$."""
        n_nu = ufloat(3.044, 0.001)
        gs0_u: UFloat = gs0(n_nu=n_nu)
        assert gs0_u.nominal_value == gs0()
        assert gs0_u.std_dev == pytest.approx(7 / 8 * 2 * 4 / 11 * 3 / 4 * (3.044 / 3)**(-1/4) * 0.001, rel=1e-12)
        ge0_u: UFloat = ge0(n_nu=n_nu)
        assert ge0_u.nominal_value == ge0()
        assert ge0_u.std_dev == pytest.approx(7 / 8 * 2 * (4 / 11)**(4/3) * 0.001, rel=1e-12)


class HubbleTest(unittest.TestCase):
    """Tests for the Hubble rate."""

    @staticmethod
    def test_H_SI() -> None:
        r"""Test that $H = \sqrt{\frac{8 \pi G e}{3 c^2}}$ with the default $G$ is unchanged by the $G$ argument."""
        e = np.array([1e-3, 0.5, 2.])
        np.testing.assert_array_equal(H_SI(e), H_FACTOR_SI * np.sqrt(e))
        assert H_SI(0.5) == H_FACTOR_SI * math.sqrt(0.5)
        assert H_SI(0.5, G=4 * const.G) == pytest.approx(2 * H_SI(0.5), rel=1e-15)

    @staticmethod
    def test_H_SI_uncertainty() -> None:
        r"""The relative uncertainty of $H \propto \sqrt{G e}$ should be half of that of $G e$."""
        e = ufloat(0.5, 0.005)
        result = H_SI(e, G=const.G_U)
        assert result.nominal_value == H_SI(0.5)
        rel_ge = math.hypot(const.G_U.std_dev / const.G, 0.01)
        assert result.std_dev / result.nominal_value == pytest.approx(rel_ge / 2, rel=1e-12)
        result_H = H(e, G=const.G_U)
        assert result_H.std_dev / result_H.nominal_value == pytest.approx(rel_ge / 2, rel=1e-12)

    @staticmethod
    def test_H_SI_uncertainty_array() -> None:
        r"""$H$ should be computable for arrays of floats with uncertainties, and for arrays of mixed floats."""
        e_nom = np.array([1e-3, 0.5, 2.])
        e = unumpy.uarray(e_nom, 0.01 * e_nom)
        result: UFloatArr1D = H_SI(e)
        np.testing.assert_array_equal(unumpy.nominal_values(result), H_SI(e_nom))
        np.testing.assert_allclose(unumpy.std_devs(result) / unumpy.nominal_values(result), 0.005, rtol=1e-12)
        mixed = np.array([1e-3, ufloat(0.5, 0.005)], dtype=object)
        result_mixed: UFloatArr1D = H(mixed)
        assert result_mixed[0] == H(1e-3)
        assert result_mixed[1].std_dev / result_mixed[1].nominal_value == pytest.approx(0.005, rel=1e-12)

    @staticmethod
    def test_H() -> None:
        r"""Test that $H = \sqrt{\frac{8 \pi G e}{3}}$, where $G = 1$ by default."""
        e = np.array([1e-3, 0.5, 2.])
        np.testing.assert_allclose(H(e), np.sqrt(8 * np.pi * e / 3), rtol=1e-15)
        np.testing.assert_allclose(H(e, G=2.5), np.sqrt(8 * np.pi * 2.5 * e / 3), rtol=1e-15)
        assert H(0.5, G=4.) == pytest.approx(2 * H(0.5), rel=1e-15)


class ConstTest(unittest.TestCase):
    r"""Tests for the constants from which $F_{\text{gw},0}$ is computed."""

    def test_a_rad(self) -> None:
        """The radiation constant should match the CODATA value."""
        assert const.A_RADIATION / A_RAD_REF == pytest.approx(1, abs=5e-7)  # noqa: SIM300

    def test_gs0(self) -> None:
        r"""The computed $g_{s0}$ should match the literature.

        It should match the $3.91$ of :caprini_2020:`\ ` p. 12 for $N_\nu = 3$,
        and $h_\text{eff}$ of :escudero_2026:`\ ` table 1 for the default $N_\nu$.
        """
        assert gs0(gs0_photon=2, n_nu=3) == pytest.approx(43 / 11, rel=1e-15)
        assert gs0(gs0_photon=2, n_nu=3) == pytest.approx(3.91, abs=5e-3)
        assert gs0() == pytest.approx(const.GS0, abs=5e-4)
        assert const.GS0 == pytest.approx(3.91, abs=0.0201)  # noqa: SIM300

    def test_omega_photon_h2(self) -> None:
        r"""$\Omega_{\gamma,0} h^2$ should match the literature value."""
        assert const.OMEGA_PHOTON_H2 / OMEGA_PHOTON_H2_REF == pytest.approx(1, abs=5e-5)  # noqa: SIM300

    def test_omega_photon_h_scaling(self) -> None:
        r"""$\Omega_{\gamma,0}$ should be $\Omega_{\gamma,0} h^2$ divided by $h^2$."""
        assert const.OMEGA_PHOTON * const.H2 / const.OMEGA_PHOTON_H2 == pytest.approx(1, abs=5e-13)  # noqa: SIM300

    def test_h0_hz(self) -> None:
        r"""$H_0$ should be $h {H}_{100}$."""
        assert const.H0_HZ / (const.H * const.H0_100_HZ) == pytest.approx(1, abs=5e-13)  # noqa: SIM300


class ConstUncertaintyTest(unittest.TestCase):
    """Tests for the uncertainties of the constants."""

    def test_literature(self) -> None:
        """The uncertainties should be the standard uncertainties of the literature."""
        # CODATA 2018
        assert const.G_U.std_dev == pytest.approx(0.00015e-11, rel=1e-12)
        # Fixsen (2009), table 2
        assert const.T_CMB_U.std_dev == pytest.approx(0.00057, rel=1e-12)
        # Planck 2018, table 2: H_0 = 67.66 +- 0.42 km/s/Mpc
        assert const.H0_KM_S_MPC_U.nominal_value == pytest.approx(67.66, rel=1e-12)
        assert const.H0_KM_S_MPC_U.std_dev == pytest.approx(0.42, rel=1e-12)

    def test_nominal_values(self) -> None:
        """The float constants should be the nominal values of the constants with uncertainties."""
        for name in const.__all__:
            if not name.endswith("_U"):
                continue
            with self.subTest(name=name):
                value = getattr(const, name)
                assert getattr(const, name.removesuffix("_U")) == value.nominal_value
                assert value.std_dev > 0

    def test_derived(self) -> None:
        """The derived constants should have the uncertainties of linear error propagation."""
        h_rel = const.H_U.std_dev / const.H
        assert const.H2_U.std_dev / const.H2 == pytest.approx(2 * h_rel, rel=1e-12)
        assert const.H0_HZ_U.std_dev / const.H0_HZ == pytest.approx(h_rel, rel=1e-12)
        G_rel = const.G_U.std_dev / const.G
        T_rel = const.T_CMB_U.std_dev / const.T_CMB
        assert const.PLANCK_LENGTH_U.std_dev / const.PLANCK_LENGTH == pytest.approx(G_rel / 2, rel=1e-12)
        om_h2_rel = math.hypot(G_rel, 4 * T_rel)
        assert const.OMEGA_PHOTON_H2_U.std_dev / const.OMEGA_PHOTON_H2 == pytest.approx(om_h2_rel, rel=1e-12)
        assert const.OMEGA_PHOTON_U.std_dev / const.OMEGA_PHOTON \
            == pytest.approx(math.hypot(om_h2_rel, 2 * h_rel), rel=1e-12)

    def test_correlations(self) -> None:
        r"""The correlations should be taken into account, so that the $h$ of $\Omega_{\gamma,0}$ cancels out."""
        result = const.OMEGA_PHOTON_U * const.H2_U
        assert result.nominal_value == pytest.approx(const.OMEGA_PHOTON_H2, rel=1e-15)
        assert result.std_dev == pytest.approx(const.OMEGA_PHOTON_H2_U.std_dev, rel=1e-12)


class FGw0Test(unittest.TestCase):
    r"""Tests for $F_{\text{gw},0}$."""

    def test_reference_value(self) -> None:
        r"""The computed $F_{\text{gw},0}$ should be consistent with the value of :caprini_2020:`\ ` eq. 20.

        This is the test that ties the computed constants to the literature.
        """
        computed = F_gw0_h2(ge_star=100., om_gamma0_h2=const.OMEGA_PHOTON_H2)
        assert computed == pytest.approx(F_GW0_H2_REF, abs=F_GW0_H2_REF_ERR)
        # The agreement should in fact be better than the uncertainty of the reference value.
        assert computed / F_GW0_H2_REF == pytest.approx(1, abs=0.0068)

    def test_gs_star_default(self) -> None:
        r"""Omitting $g_{s\ast}$ should be equivalent to setting $g_{s\ast} = {g}_\ast$."""
        for g_star in (10., 100., 106.75):
            with self.subTest(g_star=g_star):
                assert F_gw0_h2(ge_star=g_star) == F_gw0_h2(ge_star=g_star, gs_star=g_star)

    @staticmethod
    def test_g_star_scaling() -> None:
        r"""For $g_{s\ast} = {g}_\ast$ the scaling should be $\left( \frac{100}{{g}_\ast} \right)^\frac{1}{3}$."""
        g_star = np.array([1., 10., 100., 106.75, 1000.])
        expected = F_gw0_h2(ge_star=100.) * (100 / g_star)**(1/3)
        np.testing.assert_allclose(F_gw0_h2(ge_star=g_star), expected, rtol=1e-12)

    def test_uncertainty(self) -> None:
        r"""The uncertainty of $\Omega_{\gamma,0} h^2$ should propagate to $F_{\text{gw},0} h^2$."""
        computed: UFloat = F_gw0_h2(ge_star=100., om_gamma0_h2=const.OMEGA_PHOTON_H2_U)
        assert computed.nominal_value == F_gw0_h2(ge_star=100.)
        assert computed.std_dev / computed.nominal_value \
            == pytest.approx(const.OMEGA_PHOTON_H2_U.std_dev / const.OMEGA_PHOTON_H2, rel=1e-12)
        # The uncertainty of the constants is much smaller than that of the reference value.
        assert computed.std_dev / const.H2 < F_GW0_REF_ERR / 10

    @staticmethod
    def test_uncertainty_array() -> None:
        r"""$F_{\text{gw},0} h^2$ should be computable for arrays of floats with uncertainties."""
        g_star = np.array([10., 100., 106.75])
        g_star_u = unumpy.uarray(g_star, 1.)
        computed: UFloatArr1D = F_gw0_h2(ge_star=g_star_u, gs_star=g_star, om_gamma0_h2=const.OMEGA_PHOTON_H2_U)
        np.testing.assert_array_equal(unumpy.nominal_values(computed), F_gw0_h2(ge_star=g_star, gs_star=g_star))
        # F_gw0_h2 is proportional to ge_star and om_gamma0_h2 when gs_star is fixed.
        rel = np.sqrt((1 / g_star)**2 + (const.OMEGA_PHOTON_H2_U.std_dev / const.OMEGA_PHOTON_H2)**2)
        np.testing.assert_allclose(unumpy.std_devs(computed) / unumpy.nominal_values(computed), rel, rtol=1e-12)

    def test_gs_star_dependence(self) -> None:
        r"""$F_{\text{gw},0}$ should scale as $g_{s\ast}^{-\frac{4}{3}}$."""
        ratio = F_gw0_h2(ge_star=100., gs_star=50.) / F_gw0_h2(ge_star=100., gs_star=100.)
        assert ratio == pytest.approx(2**(4/3), abs=5e-13)


if __name__ == "__main__":
    unittest.main()
