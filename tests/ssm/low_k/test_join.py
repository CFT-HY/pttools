r"""Tests for joining the frequency regimes of the GW power spectrum."""

import unittest

import numpy as np

from pttools.math import assert_allclose
from pttools.ssm.const import CS0
from pttools.ssm.low_k import (
    IV_ANALYTICAL,
    Iv_resampled,
    Pv_analytical,
    gw_junction,
    pow_gw_approximation,
    power_spectrum_integration_high,
    power_spectrum_integration_int,
    power_spectrum_integration_low,
    z_cross_junction,
    z_int_high,
    zp_gw_approx,
)
import pttools.type_hints as th
from tests.ssm.low_k.common import NUS, P_TILDE_V, ZP_SW, X, cs_from_nu

R_STAR: float = 0.1
ETA_RATIO: float = 2.


def params(nu: float) -> tuple[float, float, float]:
    r"""$c_s$, $\tau_*$ and $\tau_\text{end}$ for the given $\nu$."""
    tau_star = (1 + nu) / R_STAR
    return (CS0 if nu == 0 else cs_from_nu(nu)), tau_star, tau_star * (1 + ETA_RATIO)


def regimes(z: th.FloatArr1D, nu: float) -> tuple[th.FloatArr1D, th.FloatArr1D, th.FloatArr1D]:
    r"""$\tilde{P}_\text{gw}$ in the low-, intermediate- and high-frequency regimes for the analytical $\tilde{P}_v$."""
    cs, tau_star, tau_end = params(nu)
    p_low = power_spectrum_integration_low(
        x_data=X, Pv_data=P_TILDE_V, z=z, cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end)
    p_int = power_spectrum_integration_int(z=z, cs=cs, tau_star=tau_star, Iv=IV_ANALYTICAL)
    p_high = power_spectrum_integration_high(x_data=X, Pv_data=P_TILDE_V, z=z, cs=cs)
    return p_low, p_int, p_high


class GWJunctionTest(unittest.TestCase):
    r"""Tests for :py:func:`pttools.ssm.low_k.join.gw_junction`, a modified :giombi_2024_cs:`\ ` eq. 4.9."""

    def test_limits(self) -> None:
        r"""Test that the junction reduces to the correct regime far from $z_\times$ and $z_{\text{int,high}}$."""
        for nu in (0., *NUS):
            with self.subTest(nu=nu):
                cs, tau_star, tau_end = params(nu)
                z_x = z_cross_junction(cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end)
                z = np.array([z_x / 3, 2 * z_x, 100.])
                p_low, p_int, p_high = regimes(z, nu)
                p_gw = gw_junction(
                    z=z, Pgw_low=p_low, Pgw_int=p_int, Pgw_high=p_high,
                    cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end)
                assert_allclose(p_gw, [p_low[0], p_int[1], p_high[2]], rtol=1e-6)

    def test_continuity(self) -> None:
        r"""Test that the junction is continuous at $z_\times$.

        There $\tilde{P}_\text{gw}^\text{low} = \tilde{P}_\text{gw}^\text{int}$.
        """
        for nu in (0., *NUS):
            with self.subTest(nu=nu):
                cs, tau_star, tau_end = params(nu)
                z = np.array([z_cross_junction(cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end)])
                p_low, p_int, p_high = regimes(z, nu)
                p_gw = gw_junction(
                    z=z, Pgw_low=p_low, Pgw_int=p_int, Pgw_high=p_high,
                    cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end)
                assert_allclose(p_gw, p_low, rtol=1e-5)

    def test_z_int_high(self) -> None:
        r"""Test that the intermediate regime is switched off where $\tilde{P}_\text{gw}^\text{high}$ exceeds it.

        This crossing $z_{\text{int,high}}$ is below the estimate of the GW peak $k_\star = 2 c_s k_p$
        (:giombi_2024_cs:`\ ` eq. 3.15), at about $c_s k_p$.
        """
        nu = 0.25
        cs, tau_star, tau_end = params(nu)
        z = np.logspace(-2, 2, 2001)
        p_low, p_int, p_high = regimes(z, nu)
        z_ih = z[np.argmax(p_high > p_int)]
        assert_allclose(z_int_high(z=z, Pgw_int=p_int, Pgw_high=p_high), z_ih, rtol=1e-14)
        assert 0.7 < z_ih / (cs * ZP_SW) < 1.4
        assert z_ih < zp_gw_approx(cs=cs)
        p_gw = gw_junction(
            z=z, Pgw_low=p_low, Pgw_int=p_int, Pgw_high=p_high,
            cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end)
        i = np.searchsorted(z, 2 * z_ih)
        assert_allclose(p_gw[i], p_high[i], rtol=1e-12)

    def test_z_int_high_fallback(self) -> None:
        r"""Test the fallbacks of $z_{\text{int,high}}$ when the regimes do not cross.

        The default is $c_s x_{p,v}$, where $x_{p,v}$ is $x$ at the maximum of $x \tilde{P}_v(x)$.
        With the switch, or if $x_{p,v}$ is nan, it is the estimate of the GW peak
        $k_\star R_* = 2 c_s k_p R_* = 4 \pi c_s$ (:giombi_2024_cs:`\ ` eq. 3.15 with $k_p R_* = 2\pi$).
        """
        nu = 0.25
        cs, tau_star, tau_end = params(nu)
        xp_v = 1.3
        for xp_v_arg, fallback_zp_gw, z_fallback in (
                (xp_v, False, cs * xp_v),
                (xp_v, True, 4 * np.pi * cs),
                (np.nan, False, 4 * np.pi * cs)):
            with self.subTest(xp_v=xp_v_arg, fallback_zp_gw=fallback_zp_gw):
                z = z_fallback * np.array([0.5, 2.])
                p_low, p_int, _ = regimes(z, nu)
                # Without the low-frequency regime, which is not negligible at c_s x_{p,v}
                p_low = np.zeros_like(z)
                p_high = np.zeros_like(z)
                with self.assertLogs("pttools.ssm.low_k.join", level="WARNING"):
                    p_gw = gw_junction(
                        z=z, Pgw_low=p_low, Pgw_int=p_int, Pgw_high=p_high,
                        cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end,
                        xp_v=xp_v_arg, fallback_zp_gw=fallback_zp_gw)
                assert_allclose(p_gw, [p_int[0], 0.], atol=1e-12 * p_int[0])


class PowGWApproximationTest(unittest.TestCase):
    r"""Tests for :py:func:`pttools.ssm.low_k.join.pow_gw_approximation`."""

    def test_regimes(self) -> None:
        r"""Test that the result is the junction of the three regimes."""
        nu = 0.25
        cs, tau_star, tau_end = params(nu)
        z = np.logspace(-2, 2, 200)
        eps = 1e-8
        # The x grid of pow_gw_approximation
        x = np.logspace(
            np.log10(z.min() * (1 - cs) / (2 * cs) - eps), np.log10(z.max() * (1 + cs) / (2 * cs) + eps), z.size)
        spec_den_v = Pv_analytical(x, kp_sw=ZP_SW, ubarf2=1.)
        p_gw = pow_gw_approximation(
            z=z, spec_den_v=spec_den_v, cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end, eps=eps)
        assert np.all(np.isfinite(p_gw))
        assert np.all(p_gw > 0)
        p_low = power_spectrum_integration_low(
            x_data=x, Pv_data=spec_den_v, z=z, cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end)
        p_int = power_spectrum_integration_int(
            z=z, cs=cs, tau_star=tau_star, Iv=Iv_resampled(x=x, P_tilde_v=spec_den_v))
        p_high = power_spectrum_integration_high(x_data=x, Pv_data=spec_den_v, z=z, cs=cs)
        assert_allclose(
            p_gw,
            4 / 3 * gw_junction(
                z=z, Pgw_low=p_low, Pgw_int=p_int, Pgw_high=p_high,
                cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end),
            rtol=1e-12
        )
