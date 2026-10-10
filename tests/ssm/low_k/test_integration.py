r"""Tests for the source integrals and the GW power spectrum integration in the different frequency regimes."""

import unittest

import numpy as np
from scipy.integrate import quad

from pttools.ssm.const import CS0
from pttools.ssm.low_k import (
    IV_ANALYTICAL,
    JV_ANALYTICAL,
    Iv,
    Iv_resampled,
    Jv,
    Jv_resampled,
    Pv_analytical,
    kernel_int_bracket,
    power_spectrum_integration_high,
    power_spectrum_integration_int,
    power_spectrum_integration_low,
)
from pttools.ssm.spec_den_gw import spec_den_gw
from pttools.utils import assert_allclose
from tests.ssm.low_k.common import KP, NUS, P_TILDE_V, X, cs_from_nu, p_tilde_gw_low

R_STAR: float = 0.1
ETA_RATIO: float = 2.


class PvAnalyticalTest(unittest.TestCase):
    r"""Tests for the analytical velocity spectral density of :giombi_2024_cs:`\ ` eq. 3.1."""

    def test_parseval(self) -> None:
        r"""Test the normalisation of $\tilde{P}_v$ with Parseval's theorem.

        $$\bar{U}_f^2 = \int \frac{d^3 \mathbf{p}}{(2\pi)^3} C_{\tilde{v}\tilde{v}}(p, \eta_*, \eta_*)$$
        :giombi_2024_cs:`\ ` below eq. 3.1,
        where $C_{\tilde{v}\tilde{v}}(p, \eta_*, \eta_*) = 2 P_v(p)$ (eq. 2.30).
        This gives $\frac{1}{\pi^2} \int_0^\infty dx x^2 \tilde{P}_v(x) = 1$.
        """
        for kp in (1., KP, 10.):
            with self.subTest(kp=kp):
                integral = quad(
                    lambda x, kp=kp: x**2 * Pv_analytical(x, kp=kp, ubarf2=1.), 0, np.inf, epsrel=1e-12)[0]
                assert_allclose(integral / np.pi**2, 1., rtol=1e-10)

    def test_power_laws(self) -> None:
        r"""Test the causal $k^2$ tail and the $k^{-4}$ decay of a flow with shocks, :giombi_2024_cs:`\ ` eq. 3.1."""
        x_low = np.logspace(-4, -3, 5)
        x_high = np.logspace(4, 5, 5)
        assert_allclose(np.polyfit(np.log(x_low), np.log(Pv_analytical(x_low, KP, 1.)), 1)[0], 2.)
        assert_allclose(np.polyfit(np.log(x_high), np.log(Pv_analytical(x_high, KP, 1.)), 1)[0], -4.)

    def test_peak(self) -> None:
        r"""Test that $x \tilde{P}_v(x)$ has its maximum at $k_p$, which gives $k_\star = 2 c_s k_p$.

        :giombi_2024_cs:`\ ` eq. 3.15 and its footnote
        """
        x = KP * np.linspace(0.9, 1.1, 20001)
        assert_allclose(x[np.argmax(x * Pv_analytical(x, KP, 1.))], KP, rtol=1e-4)


class SourceIntegralTest(unittest.TestCase):
    r"""Tests for $\mathcal{I}_v$ and $\mathcal{J}_v$."""

    def test_iv(self) -> None:
        r"""$\mathcal{I}_v = \frac{1}{32 \pi^2}$ for the analytical $\tilde{P}_v$.

        :giombi_2024_cs:`\ ` below eq. 3.7,
        :giombi_2026:`\ ` below eq. 3.6
        """
        assert_allclose(IV_ANALYTICAL, 1 / (32 * np.pi**2), rtol=1e-15)
        assert_allclose(Iv(x=X, P_tilde_v=P_TILDE_V), IV_ANALYTICAL, rtol=1e-9)

    def test_jv(self) -> None:
        r"""$\mathcal{J}_v = \frac{\mathcal{I}_v}{(2\pi)^2}$ for the analytical $\tilde{P}_v$.

        :giombi_2026:`\ ` below eq. 3.6
        """
        assert_allclose(JV_ANALYTICAL, IV_ANALYTICAL / (2 * np.pi)**2, rtol=1e-15)
        assert_allclose(Jv(x=X, P_tilde_v=P_TILDE_V), JV_ANALYTICAL, rtol=1e-9)

    def test_kp_scaling(self) -> None:
        r"""Test the scaling with the peak wavenumber.

        Substituting $u = x / k_p$ gives $\mathcal{I}_v = \frac{\pi}{4 (k_p R_*)^3} \bar{U}_f^4$
        and $\mathcal{J}_v = \frac{\pi}{4 (k_p R_*)^5} \bar{U}_f^4$.
        """
        ubarf2 = 0.01
        for kp in (1., 3., 20.):
            with self.subTest(kp=kp):
                p_tilde_v = Pv_analytical(X, kp=kp, ubarf2=ubarf2)
                assert_allclose(Iv(x=X, P_tilde_v=p_tilde_v), np.pi / (4 * kp**3) * ubarf2**2, rtol=1e-8)
                assert_allclose(Jv(x=X, P_tilde_v=p_tilde_v), np.pi / (4 * kp**5) * ubarf2**2, rtol=1e-8)

    def test_resampled(self) -> None:
        """Test that resampling a coarse grid gives the integral of the fine grid."""
        x = np.logspace(-3, 4, 300)
        p_tilde_v = Pv_analytical(x, kp=KP, ubarf2=1.)
        assert_allclose(Iv_resampled(x=x, P_tilde_v=p_tilde_v, nx=10000), IV_ANALYTICAL, rtol=5e-3)
        assert_allclose(Jv_resampled(x=x, P_tilde_v=p_tilde_v, nx=10000), JV_ANALYTICAL, rtol=5e-3)


class IntegrationLowTest(unittest.TestCase):
    r"""Tests for $\tilde{P}_\text{gw}^\text{low}$."""

    def test_reference(self) -> None:
        r"""Test against :giombi_2026:`\ ` eq. 3.5a with the analytical $\tilde{P}_v$."""
        z = np.logspace(-3, -1, 5)
        for nu in NUS:
            with self.subTest(nu=nu):
                tau_star = (1 + nu) / R_STAR
                assert_allclose(
                    power_spectrum_integration_low(
                        x_data=X, Pv_data=P_TILDE_V, z=z, cs=cs_from_nu(nu), nu=nu,
                        tau_star=tau_star, tau_end=tau_star * (1 + ETA_RATIO)),
                    p_tilde_gw_low(z=z, nu=nu, r_star=R_STAR, eta_ratio=ETA_RATIO, iv=IV_ANALYTICAL),
                    # The integral is resampled to 1000 points.
                    rtol=1e-5
                )

    def test_bag(self) -> None:
        r"""Test the bag model against :giombi_2024_cs:`\ ` eq. 3.5 with the kernel of eq. 3.4.

        $$\tilde{P}_\text{gw}^\text{low} = \frac{8}{15} \tau_* \mathcal{I}_v
        \ln^2 \left( \frac{\tau_\text{end}}{\tau_*} \right)$$
        This is also the limit $\nu \rightarrow 0$ of eq. 3.6.
        As the kernel does not depend on $z$, neither does $\tilde{P}_\text{gw}^\text{low}$,
        and $\mathcal{P}_\text{gw} \propto k^3$.
        """
        z = np.logspace(-3, -1, 5)
        tau_star = 1 / R_STAR
        tau_end = tau_star * (1 + ETA_RATIO)
        p_gw = power_spectrum_integration_low(
            x_data=X, Pv_data=P_TILDE_V, z=z, cs=CS0, nu=0., tau_star=tau_star, tau_end=tau_end)
        assert p_gw.shape == z.shape
        assert_allclose(p_gw, 8 / 15 * tau_star * IV_ANALYTICAL * np.log(1 + ETA_RATIO)**2, rtol=1e-5)
        # The nu > 0 branch gives the same result at nu = 0.
        assert_allclose(
            power_spectrum_integration_low(
                x_data=X, Pv_data=P_TILDE_V, z=z, cs=CS0 - 1e-6, nu=0., tau_star=tau_star, tau_end=tau_end),
            p_gw, rtol=1e-12
        )


class IntegrationIntTest(unittest.TestCase):
    r"""Tests for $\tilde{P}_\text{gw}^\text{int}$."""

    def test_reference(self) -> None:
        r"""Test against :giombi_2026:`\ ` eq. 3.8a.

        $$\mathcal{P}_\text{gw,sw}^\text{lo,int} = 3 \left( \Gamma \bar{U}_f^2 \right)^2 r_*^2
        \left( \frac{a_*}{a_r} \right)^\frac{2\nu}{1+\nu}
        \frac{z}{\pi^2} \frac{2}{3 c_s^4}
        \left[ 3 - 2c_s^2 - \frac{3}{c_s}(1 - c_s^2) \operatorname{arctanh}(c_s) \right] \mathcal{I}_v$$
        divided by the prefactor of eq. 2.35.
        """
        z = np.logspace(-1, 1, 5)
        for nu in (0., *NUS):
            with self.subTest(nu=nu):
                cs = cs_from_nu(nu)
                ref = R_STAR**2 * z / np.pi**2 * 2 / (3 * cs**4) * kernel_int_bracket(cs) * IV_ANALYTICAL \
                    / ((1 + nu) * R_STAR * z**3 / (2 * np.pi**2))
                assert_allclose(
                    power_spectrum_integration_int(z=z, cs=cs, tau_star=(1 + nu) / R_STAR, Iv=IV_ANALYTICAL),
                    ref, rtol=1e-12
                )

    def test_mu_integral(self) -> None:
        r"""Test against the numerical integration of :giombi_2024_cs:`\ ` eq. 3.10 with the kernel of eq. 3.9c.

        $$\tilde{P}_\text{gw}^\text{int} = 2 \tau_* \mathcal{I}_v \int_{-1}^1 d\mu (1 - \mu^2)^2
        \frac{1}{4 (z \tau_*)^2} \frac{1 + c_s^2 \mu^2}{(1 - c_s^2 \mu^2)^2}$$
        """
        z = 0.5
        tau_star = 12.
        for cs in (0.3, 0.5, CS0):
            with self.subTest(cs=cs):
                integral = quad(
                    lambda mu: (1 - mu**2)**2 / (4 * (z * tau_star)**2)
                    * (1 + cs**2 * mu**2) / (1 - cs**2 * mu**2)**2,  # noqa: B023
                    -1, 1, epsrel=1e-13)[0]
                assert_allclose(
                    power_spectrum_integration_int(z=z, cs=cs, tau_star=tau_star, Iv=IV_ANALYTICAL),
                    2 * tau_star * IV_ANALYTICAL * integral, rtol=1e-10
                )

    def test_scaling(self) -> None:
        r"""Test that $\mathcal{P}_\text{gw}^\text{int} \propto k$, :giombi_2024_cs:`\ ` sec. 3.2."""
        z = np.logspace(-1, 1, 5)
        p_gw = z**3 * power_spectrum_integration_int(z=z, cs=0.5, tau_star=10., Iv=IV_ANALYTICAL)
        assert_allclose(np.polyfit(np.log(z), np.log(p_gw), 1)[0], 1.)


class IntegrationHighTest(unittest.TestCase):
    r"""Tests for $\tilde{P}_\text{gw}^\text{high}$."""

    def test_spec_den_gw(self) -> None:
        r"""Test that the result equals that of the SSM, :py:func:`pttools.ssm.spec_den_gw.spec_den_gw`.

        :giombi_2024_cs:`\ ` eq. 3.13 without the source lifetime factor
        """
        z = np.logspace(-1, 2, 50)
        for cs in (0.3, 0.5, CS0):
            with self.subTest(cs=cs):
                assert_allclose(
                    power_spectrum_integration_high(x_data=X, Pv_data=P_TILDE_V, z=z, cs=cs),
                    spec_den_gw(
                        z_lookup=X, P_tilde_v_lookup=P_TILDE_V, y=z, cs=cs, nx_P_tilde_gw=1000, parallel=False)[0],
                    rtol=1e-5
                )

    def test_power_laws(self) -> None:
        r"""Test the power laws $\mathcal{P}_\text{gw} \propto k^9$ below and $k^{-3}$ above the peak.

        :giombi_2024_cs:`\ ` eq. 3.18,
        :giombi_2026:`\ ` eq. 3.10.
        """
        cs = 0.5
        z_low = np.logspace(-2, -1, 5)
        z_high = np.logspace(2.5, 3, 5)
        p_low = z_low**3 * power_spectrum_integration_high(x_data=X, Pv_data=P_TILDE_V, z=z_low, cs=cs)
        p_high = z_high**3 * power_spectrum_integration_high(x_data=X, Pv_data=P_TILDE_V, z=z_high, cs=cs)
        assert_allclose(np.polyfit(np.log(z_low), np.log(p_low), 1)[0], 9., rtol=1e-6)
        assert_allclose(np.polyfit(np.log(z_high), np.log(p_high), 1)[0], -3., rtol=1e-4)

    def test_peak(self) -> None:
        r"""Test the peak against the rough estimates of :giombi_2024_cs:`\ ` eq. 3.15 and 3.16.

        The estimates approximate $x \tilde{P}_v(x)$ with a Dirac delta function,
        which gives the peak location $k_\star \simeq 2 c_s k_p$ and the amplitude
        $\frac{z^3}{2\pi^2} \tilde{P}_\text{gw}^\text{high}(z_{\star}) \simeq
        \left( \frac{1 - c_s^2}{c_s^2} \right)^2 \frac{9}{(2\pi)^3} c_s^5$
        without the source lifetime factor.
        As these are rough estimates, the tolerances are loose.
        """
        z = np.logspace(0, 2, 1001)
        for cs in (0.3, 0.5, CS0):
            with self.subTest(cs=cs):
                p_gw = z**3 / (2 * np.pi**2) * power_spectrum_integration_high(x_data=X, Pv_data=P_TILDE_V, z=z, cs=cs)
                i_max = np.argmax(p_gw)
                assert 1 < z[i_max] / (2 * cs * KP) < 1.4
                assert_allclose(p_gw[i_max], ((1 - cs**2) / cs**2)**2 * 9 / (2 * np.pi)**3 * cs**5, rtol=0.4)

    def test_int_dtype(self) -> None:
        """Test that an integer z array does not truncate the result."""
        z = np.array([1, 2, 5])
        assert_allclose(
            power_spectrum_integration_high(x_data=X, Pv_data=P_TILDE_V, z=z, cs=0.5),
            power_spectrum_integration_high(x_data=X, Pv_data=P_TILDE_V, z=z.astype(np.float64), cs=0.5)
        )
