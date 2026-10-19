r"""Tests for the low-k approximation as a part of :py:class:`pttools.ssm.SSMSpectrum`."""

import typing as tp
import unittest

import numpy as np

from pttools.bubble import Bubble
from pttools.math import assert_allclose
from pttools.models import BagModel, ConstCSModel
from pttools.ssm import SSMSpectrum
from pttools.ssm.low_k import Iv_resampled, z_cross_junction
from tests.ssm.low_k.common import nu_from_cs, p_tilde_gw_low

Y: np.ndarray = np.logspace(-3, 2, 200)
R_STAR: float = 0.1


class LowKSpectrumTest(unittest.TestCase):
    r"""Test the low-k approximation with the SSM $\tilde{P}_v$ of a bubble for the bag model and for $c_s^2 = 1/4$."""

    spectra: tp.ClassVar[list[SSMSpectrum]]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        bag = BagModel(alpha_n_min=0.05)
        const_cs = ConstCSModel(css2=1/3, csb2=1/4, a_s=5, a_b=1, V_s=1, alpha_n_min=0.1)
        cls.spectra = [
            SSMSpectrum(Bubble(bag, v_wall=0.5, alpha_n=0.05), y=Y, r_star=R_STAR),
            SSMSpectrum(Bubble(const_cs, v_wall=0.9, alpha_n=0.2), y=Y, r_star=R_STAR),
        ]

    def test_nu(self) -> None:
        r"""Test that $\nu$ corresponds to the sound speed, :giombi_2024_cs:`\ ` eq. 2.11."""
        for spectrum, nu in zip(self.spectra, (0, 1/7), strict=True):
            with self.subTest(nu=nu):
                assert_allclose(spectrum.nu_gdh2024, nu, atol=1e-12)
                assert_allclose(spectrum.nu_gdh2024, nu_from_cs(spectrum.cs), atol=1e-12)

    def test_reference(self) -> None:
        r"""Test $\tilde{P}_\text{gw}^\text{low}$ against :giombi_2026:`\ ` eq. 3.5a.

        The $\mathcal{I}_v$ is computed from the $\tilde{P}_v$ of the SSM.

        This checks the conventions for $\tau_* = \frac{1 + \nu}{r_*}$ and
        $\frac{\tau_\text{end}}{\tau_*} = 1 + \frac{\Delta \eta_\text{v}}{\eta_*}$.
        """
        for spectrum in self.spectra:
            nu = spectrum.nu_gdh2024
            with self.subTest(nu=nu):
                iv = Iv_resampled(x=spectrum.z_lookup, P_tilde_v=spectrum.spec_den_v_lookup)
                if nu == 0:
                    ref = 8 / 15 * spectrum.tau_star * iv * np.log(1 + spectrum.eta_ratio)**2 * np.ones_like(Y)
                else:
                    ref = p_tilde_gw_low(z=Y, nu=nu, r_star=R_STAR, eta_ratio=spectrum.eta_ratio, iv=iv)
                assert_allclose(spectrum.spec_den_gw_low, ref, rtol=1e-12)

    def test_power_laws(self) -> None:
        r"""Test that $\mathcal{P}_\text{gw} \propto k^{3 - 2\nu}$ at low and $\propto k$ at intermediate $k$.

        :giombi_2026:`\ ` eq. 3.7,
        :giombi_2024_cs:`\ ` sec. 3.2.
        """
        for spectrum in self.spectra:
            nu = spectrum.nu_gdh2024
            with self.subTest(nu=nu):
                log_y = np.log(Y)
                assert_allclose(np.polyfit(log_y, np.log(Y**3 * spectrum.spec_den_gw_low), 1)[0], 3 - 2 * nu)
                assert_allclose(np.polyfit(log_y, np.log(Y**3 * spectrum.spec_den_gw_int), 1)[0], 1.)

    def test_crossing(self) -> None:
        r"""Test that the low- and intermediate-frequency regimes cross at $z_\times$.

        As both use the same $\mathcal{I}_v$, this is exact.
        """
        for spectrum in self.spectra:
            nu = spectrum.nu_gdh2024
            with self.subTest(nu=nu):
                z_x = z_cross_junction(cs=spectrum.cs, nu=nu, tau_star=spectrum.tau_star, tau_end=spectrum.tau_end)
                p_low = spectrum.spec_den_gw_low[0] * (z_x / Y[0])**(-2 * nu)
                p_int = spectrum.spec_den_gw_int[0] * (z_x / Y[0])**(-2)
                assert_allclose(p_low, p_int, rtol=1e-12)

    def test_junction(self) -> None:
        r"""Test that the expanded spectrum is the low-frequency tail at low $k$ and the SSM spectrum at high $k$."""
        for spectrum in self.spectra:
            with self.subTest(nu=spectrum.nu_gdh2024):
                assert_allclose(spectrum.spec_den_gw_expanded[:10], spectrum.spec_den_gw_low[:10], rtol=1e-12)
                assert_allclose(spectrum.spec_den_gw_expanded[-10:], spectrum.spec_den_gw_ssm[-10:], rtol=1e-12)
                assert np.all(spectrum.spec_den_gw_expanded > 0)


class CsFrictionSpectrumTest(unittest.TestCase):
    r"""Test the switch for computing the acoustic friction from $c_s$ in :py:class:`pttools.ssm.SSMSpectrum`."""

    bubble: tp.ClassVar[Bubble]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.bubble = Bubble(BagModel(alpha_n_min=0.05), v_wall=0.5, alpha_n=0.05)

    def test_omega_differs(self) -> None:
        r"""Test the switch for $c_s^2 = \frac{1}{3}$ but $\omega = \frac{1}{4}$, i.e. $\nu = \frac{1}{7}$.

        By default the friction exponent is $d = \nu$ (:giombi_2026:`\ ` eq. 3.5a).
        With the switch, there is no acoustic friction, and $\Upsilon_{2\nu}$ is replaced with
        $\Upsilon_0 = \ln \left( \frac{\tau_\text{end}}{\tau_*} \right)$.
        """
        spectra = [
            SSMSpectrum(self.bubble, y=Y, r_star=R_STAR, omega_barotropic=0.25, low_k_cs_friction=switch)
            for switch in (False, True)
        ]
        nu = 1/7
        for spectrum, switch in zip(spectra, (False, True), strict=True):
            assert spectrum.low_k_cs_friction is switch
            assert_allclose(spectrum.nu_gdh2024, nu, rtol=1e-14)
            assert_allclose(spectrum.cs2, 1/3, rtol=1e-12)
        iv = Iv_resampled(x=spectra[0].z_lookup, P_tilde_v=spectra[0].spec_den_v_lookup)
        default = p_tilde_gw_low(z=Y, nu=nu, r_star=R_STAR, eta_ratio=spectra[0].eta_ratio, iv=iv)
        assert_allclose(spectra[0].spec_den_gw_low, default, rtol=1e-12)
        upsilon_2nu = (1 - (1 + spectra[0].eta_ratio)**(-2 * nu)) / (2 * nu)
        assert_allclose(
            spectra[1].spec_den_gw_low,
            default * (np.log(1 + spectra[1].eta_ratio) / upsilon_2nu)**2,
            rtol=1e-12
        )

    def test_omega_equal(self) -> None:
        r"""Test that the switch makes no difference for $c_s^2 = \omega$."""
        spectra = [
            SSMSpectrum(self.bubble, y=Y, r_star=R_STAR, low_k_cs_friction=switch) for switch in (False, True)
        ]
        assert_allclose(spectra[1].spec_den_gw_low, spectra[0].spec_den_gw_low, rtol=1e-12)
        assert_allclose(spectra[1].spec_den_gw_expanded, spectra[0].spec_den_gw_expanded, rtol=1e-12)
