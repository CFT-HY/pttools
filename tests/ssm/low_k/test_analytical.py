r"""Tests for the analytical approximations of the GW power spectrum with the analytical $\tilde{P}_v$."""

import unittest

import numpy as np

from pttools.ssm.const import CS0
from pttools.ssm.low_k import (
    IV_ANALYTICAL,
    Pgw_approx,
    gw_spectral_density_approx_high,
    gw_spectral_density_approx_int,
    gw_spectral_density_approx_low,
    power_spectrum_integration_high,
    power_spectrum_integration_int,
    power_spectrum_integration_low,
    z_cross,
)
from pttools.utils import assert_allclose
from tests.ssm.low_k.common import NUS, P_TILDE_V, X, cs_from_nu

#: $H L_f = r_*$
HLF: float = 0.1
ETA_RATIO: float = 2.


def params(nu: float) -> tuple[float, float, float]:
    r"""$c_s$, $\tau_*$ and $\tau_\text{end}$ for the given $\nu$."""
    tau_star = (1 + nu) / HLF
    return (CS0 if nu == 0 else cs_from_nu(nu)), tau_star, tau_star * (1 + ETA_RATIO)


class AnalyticalTest(unittest.TestCase):
    r"""Test the analytical approximations against the integrals of the analytical $\tilde{P}_v$."""

    def test_low(self) -> None:
        r"""Test $\tilde{P}_\text{gw}^\text{low}$, :giombi_2026:`\ ` eq. 3.5a and :giombi_2024_cs:`\ ` eq. 3.6."""
        z = np.logspace(-3, -1, 5)
        for nu in (0., *NUS):
            with self.subTest(nu=nu):
                cs, tau_star, tau_end = params(nu)
                assert_allclose(
                    gw_spectral_density_approx_low(z=z, HLf=HLF, cs=cs, tau_end=tau_end),
                    power_spectrum_integration_low(
                        x_data=X, Pv_data=P_TILDE_V, z=z, cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end),
                    rtol=1e-5
                )

    def test_int(self) -> None:
        r"""Test $\tilde{P}_\text{gw}^\text{int}$, :giombi_2024_cs:`\ ` eq. 3.11."""
        z = np.logspace(-1, 1, 5)
        for nu in (0., *NUS):
            with self.subTest(nu=nu):
                cs, tau_star, _ = params(nu)
                assert_allclose(
                    gw_spectral_density_approx_int(z=z, HLf=HLF, cs=cs),
                    power_spectrum_integration_int(z=z, cs=cs, tau_star=tau_star, Iv=IV_ANALYTICAL),
                    rtol=1e-14
                )

    def test_high(self) -> None:
        r"""Test $\tilde{P}_\text{gw}^\text{high}$, :giombi_2024_cs:`\ ` eq. 3.13.

        $$\tilde{P}_\text{gw}^\text{high} = \frac{1}{4 \pi z c_s} \left( \frac{1 - c_s^2}{c_s^2} \right)^2
        \Upsilon_{1+2\nu} \left( \frac{\eta_*}{\eta_\text{end}} \right)
        \int_{x_{-}}^{x_{+}} \frac{dx}{x} \frac{(x - x_{+})^2 (x - x_{-})^2}{x_{+} + x_{-} - x}
        \tilde{P}_v(x) \tilde{P}_v(x_{+} + x_{-} - x)$$
        """
        z = np.array([0.5, 2., 5., 20.])
        for nu in (0., *NUS):
            with self.subTest(nu=nu):
                cs, tau_star, tau_end = params(nu)
                upsilon = (1 - (tau_star / tau_end)**(1 + 2 * nu)) / (1 + 2 * nu)
                assert_allclose(
                    [gw_spectral_density_approx_high(z=z_i, HLf=HLF, cs=cs, tau_end=tau_end) for z_i in z],
                    upsilon * power_spectrum_integration_high(x_data=X, Pv_data=P_TILDE_V, z=z, cs=cs),
                    rtol=1e-4
                )

    def test_high_short_source(self) -> None:
        r"""Test that $\tilde{P}_\text{gw}^\text{high}$ grows linearly with a short source duration.

        :giombi_2024_cs:`\ ` eq. 3.14
        """
        z = 5.
        delta = 1e-4
        for nu in (0., *NUS):
            with self.subTest(nu=nu):
                cs, tau_star, _ = params(nu)
                assert_allclose(
                    gw_spectral_density_approx_high(z=z, HLf=HLF, cs=cs, tau_end=tau_star * (1 + delta)),
                    delta * power_spectrum_integration_high(x_data=X, Pv_data=P_TILDE_V, z=np.array([z]), cs=cs)[0],
                    rtol=1e-3
                )


class PgwApproxTest(unittest.TestCase):
    r"""Tests for the global approximation of :giombi_2024_cs:`\ ` eq. 4.9."""

    def test_limits(self) -> None:
        r"""Test that the approximation reduces to the correct regime far from the crossovers.

        Below $k_\times$ the approximation is $\tilde{P}_\text{gw}^\text{low}$,
        and above $k_\star = 2 c_s k_p$, where $k_p R_* = 2\pi$, only $\tilde{P}_\text{gw}^\text{high}$ remains.
        The high-frequency term has no switch, as it is negligible at low frequencies.
        The tolerance is set by the tails of the error functions.
        """
        for nu in (0., *NUS):
            with self.subTest(nu=nu):
                cs, tau_star, tau_end = params(nu)
                z_x = z_cross(HLf=HLF, cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end)
                z_star = 4 * np.pi * cs

                z_low = z_x / 3
                assert_allclose(
                    Pgw_approx(z=z_low, HLf=HLF, cs=cs, tau_star=tau_star, tau_end=tau_end),
                    gw_spectral_density_approx_low(z=z_low, HLf=HLF, cs=cs, tau_end=tau_end)
                    + gw_spectral_density_approx_high(z=z_low, HLf=HLF, cs=cs, tau_end=tau_end),
                    rtol=1e-6
                )
                z_int = np.sqrt(z_x * z_star)
                assert_allclose(
                    Pgw_approx(z=z_int, HLf=HLF, cs=cs, tau_star=tau_star, tau_end=tau_end),
                    gw_spectral_density_approx_int(z=z_int, HLf=HLF, cs=cs)
                    + gw_spectral_density_approx_high(z=z_int, HLf=HLF, cs=cs, tau_end=tau_end),
                    rtol=1e-6
                )
                z_high = 1.5 * z_star
                assert_allclose(
                    Pgw_approx(z=z_high, HLf=HLF, cs=cs, tau_star=tau_star, tau_end=tau_end),
                    gw_spectral_density_approx_high(z=z_high, HLf=HLF, cs=cs, tau_end=tau_end),
                    rtol=1e-6
                )

    def test_continuity(self) -> None:
        r"""Test that the approximation is continuous at $k_\times$.

        There $\tilde{P}_\text{gw}^\text{low} = \tilde{P}_\text{gw}^\text{int}$.
        """
        for nu in (0., *NUS):
            with self.subTest(nu=nu):
                cs, tau_star, tau_end = params(nu)
                z_x = z_cross(HLf=HLF, cs=cs, nu=nu, tau_star=tau_star, tau_end=tau_end)
                assert_allclose(
                    Pgw_approx(z=z_x, HLf=HLF, cs=cs, tau_star=tau_star, tau_end=tau_end),
                    gw_spectral_density_approx_low(z=z_x, HLf=HLF, cs=cs, tau_end=tau_end)
                    + gw_spectral_density_approx_high(z=z_x, HLf=HLF, cs=cs, tau_end=tau_end),
                    rtol=1e-6
                )
