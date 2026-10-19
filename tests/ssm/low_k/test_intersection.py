r"""Tests for the crossover wavenumber $z_\times$ of the low- and intermediate-frequency approximations."""

import unittest

import numpy as np
from scipy.optimize import brentq

from pttools.math import assert_allclose
from pttools.ssm.const import CS0
from pttools.ssm.low_k import (
    IV_ANALYTICAL,
    kernel_int_bracket,
    power_spectrum_integration_int,
    power_spectrum_integration_low,
    z_cross,
    z_cross_approx,
    z_cross_junction,
)
from tests.ssm.low_k.common import NUS, P_TILDE_V, X, cs_from_nu

R_STAR: float = 0.1
ETA_RATIOS: tuple[float, ...] = (0.5, 2., 20.)


class ZCrossTest(unittest.TestCase):
    r"""Tests for the crossover functions.

    :py:func:`pttools.ssm.low_k.intersection.z_cross` and :py:func:`pttools.ssm.low_k.intersection.z_cross_junction`
    """

    def test_crossing(self) -> None:
        r"""Test that $\tilde{P}_\text{gw}^\text{low}(z_{\times}) = \tilde{P}_\text{gw}^\text{int}(z_{\times})$.

        This is the definition of $k_\times$ below :giombi_2024_cs:`\ ` eq. 4.9.
        """
        for nu in (0., *NUS):
            for eta_ratio in ETA_RATIOS:
                with self.subTest(nu=nu, eta_ratio=eta_ratio):
                    cs = CS0 if nu == 0 else cs_from_nu(nu)
                    tau_star = (1 + nu) / R_STAR
                    tau_end = tau_star * (1 + eta_ratio)
                    z_x = np.array([z_cross_junction(cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end)])
                    p_low = power_spectrum_integration_low(
                        x_data=X, Pv_data=P_TILDE_V, z=z_x, cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end)
                    p_int = power_spectrum_integration_int(z=z_x, cs=cs, tau_star=tau_star, Iv=IV_ANALYTICAL)
                    # The low-frequency integral is resampled to 1000 points.
                    assert_allclose(p_low, p_int, rtol=1e-5)

    def test_z_cross(self) -> None:
        r"""Test that the two implementations give the same $z_\times$ with $\tau_* = \frac{1 + \nu}{r_*}$."""
        for nu in (0., *NUS):
            for eta_ratio in ETA_RATIOS:
                with self.subTest(nu=nu, eta_ratio=eta_ratio):
                    cs = CS0 if nu == 0 else cs_from_nu(nu)
                    tau_star = (1 + nu) / R_STAR
                    tau_end = tau_star * (1 + eta_ratio)
                    assert_allclose(
                        z_cross(HLf=R_STAR, cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end),
                        z_cross_junction(cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end),
                        rtol=1e-12
                    )


class ZCrossApproxTest(unittest.TestCase):
    r"""Tests for :py:func:`pttools.ssm.low_k.intersection.z_cross_approx`."""

    @staticmethod
    def p_low_long(nu: float, tau_star: float, eta_ratio: float) -> float:
        r"""$\tilde{P}_\text{gw}^\text{low}$ for a long-lasting source, :giombi_2024_cs:`\ ` eq. 3.6.

        $$\tilde{P}_\text{gw}^\text{low} = \frac{1}{\nu^2}
        \left[ 1 - \left( 1 + \frac{\Delta \eta_\text{v}}{\eta_*} \right)^{-\nu} \right]^2
        \frac{8}{15} \tau_* \mathcal{I}_v$$
        For $\nu = 0$ the limit is $\ln^2 \left( 1 + \frac{\Delta \eta_\text{v}}{\eta_*} \right)$.
        Please note that this does not depend on $z$.
        """
        factor = np.log(1 + eta_ratio)**2 if nu == 0 else (1 - (1 + eta_ratio)**(-nu))**2 / nu**2
        return factor * 8 / 15 * tau_star * IV_ANALYTICAL

    @staticmethod
    def p_int_long(z: float, cs: float, tau_star: float) -> float:
        r"""$\tilde{P}_\text{gw}^\text{int}$ for a long-lasting source, :giombi_2024_cs:`\ ` eq. 3.11."""
        return 4 / (3 * cs**4) * kernel_int_bracket(cs) * IV_ANALYTICAL / (tau_star * z**2)

    def test_root(self) -> None:
        r"""Test $z_\times = k_{\times} R_*$ against the root of :giombi_2024_cs:`\ ` eq. 3.6 = eq. 3.11.

        This should give :giombi_2024_cs:`\ ` eq. 4.10.
        """
        for nu in (0., *NUS):
            for eta_ratio in ETA_RATIOS:
                with self.subTest(nu=nu, eta_ratio=eta_ratio):
                    cs = CS0 if nu == 0 else cs_from_nu(nu)
                    tau_star = (1 + nu) / R_STAR
                    root = brentq(
                        lambda z: np.log(
                            self.p_low_long(nu, tau_star, eta_ratio) / self.p_int_long(z, cs, tau_star)),  # noqa: B023
                        1e-6, 1e3, xtol=1e-15, rtol=1e-14
                    )
                    assert_allclose(z_cross_approx(cs=cs, nu=nu, eta_ratio=eta_ratio, r_star=R_STAR), root, rtol=1e-10)

    def test_bag(self) -> None:
        r"""Test that for the bag model $z_\times$ is finite and equals that of the junction.

        For $\nu = 0$ the low-frequency kernels of :giombi_2024_cs:`\ ` and :giombi_2026:`\ ` are the same,
        $\frac{1}{4} \ln^2 \left( \frac{\tau_\text{end}}{\tau_*} \right)$.
        For $\nu > 0$ they differ, as :giombi_2026:`\ ` uses the super-horizon expansion of the Green's function.
        """
        for eta_ratio in ETA_RATIOS:
            with self.subTest(eta_ratio=eta_ratio):
                z_x = z_cross_approx(cs=CS0, nu=0., eta_ratio=eta_ratio, r_star=R_STAR)
                assert np.isfinite(z_x)
                assert_allclose(
                    z_x,
                    z_cross_junction(cs=CS0, nu=0., tau_star=1 / R_STAR, tau_end=(1 + eta_ratio) / R_STAR),
                    rtol=1e-12
                )
