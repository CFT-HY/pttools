"""Tests for the SSMSpectrum class."""

import typing as tp
import unittest
from unittest import mock

import numpy as np
import pytest

from pttools.bubble import Bubble
from pttools.models import ConstCSModel
from pttools.ssm import DEFAULT_NUC_TYPE, SSMSpectrum, spec_den_gw
from tests.utils import TEST_JSON_PATH


class SSMSpectrumTest(unittest.TestCase):
    """Tests for the SSMSpectrum class."""

    spectrum: tp.ClassVar[SSMSpectrum]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        model = ConstCSModel(css2=1/3-0.01, csb2=1/3-0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
        cls.spectrum = SSMSpectrum(bubble, beta_tilde=1.23)

    def test_beta(self) -> None:
        """Test that the nucleation rate parameter beta is finite for a given Hubble rate."""
        assert np.isfinite(self.spectrum.beta(1.23))

    def test_bubble_spacing_enlargement_factor(self) -> None:
        """Test that the bubble spacing enlargement factor is larger than 1."""
        assert self.spectrum.bubble_spacing_enlargement_factor > 1.

    def test_cs2_Tn(self) -> None:
        r"""In the constant sound speed model $c_s^2$ is constant in each phase."""
        assert self.spectrum.css2_Tn == pytest.approx(1/3 - 0.01, abs=5e-13)
        assert self.spectrum.csb2_Tn == pytest.approx(1/3 - 0.011, abs=5e-13)
        assert self.spectrum.css2_Tn == self.spectrum.bubble.css2_Tn

    def test_dilution_of_e(self) -> None:
        """Test that the dilution of the energy density is between 0 and 1."""
        assert self.spectrum.dilution_of_e > 0
        assert self.spectrum.dilution_of_e <= 1

    def test_eta_ratio(self) -> None:
        """Test that the source duration in units of the conformal time eta_* is larger than 1."""
        assert self.spectrum.eta_ratio > 1

    def test_export(self) -> None:
        """Test that the spectrum can be exported as JSON."""
        self.spectrum.export(TEST_JSON_PATH / "ssm-spectrum.json")

    def test_H_star_eta_star(self) -> None:
        """Test that the conformal Hubble rate times the conformal time at the start of the acoustics is positive."""
        assert self.spectrum.H_star_eta_star > 0

    def test_H_star_eta_sh(self) -> None:
        """Test that the Hubble-scaled shock formation time is positive."""
        assert self.spectrum.H_star_eta_sh > 0

    def test_H_star_eta_v(self) -> None:
        """Test that the Hubble-scaled effective lifetime of the source is positive."""
        assert self.spectrum.H_star_eta_v > 0

    def test_H_star_eta_v_old(self) -> None:
        """Test that the old approximation of the effective lifetime of the source is positive."""
        assert self.spectrum.H_star_eta_v_old > 0

    def test_hx(self) -> None:
        """Test that the fractional volume hx at which bubble nucleation stops is positive."""
        assert self.spectrum.hx > 0

    def test_k_peak_eta_star(self) -> None:
        """Test that the peak wavenumber scaled by the conformal time at GW formation is positive."""
        assert self.spectrum.k_peak_eta_star > 0

    def test_J(self) -> None:
        """Test that the combined lifetime factor J is positive."""
        assert self.spectrum.J > 0

    def test_nucleation_f(self) -> None:
        """Test that the nucleation suppression function f is positive."""
        assert self.spectrum.nucleation_f > 0

    def test_pow_gw(self) -> None:
        """Test that the GW power spectrum has no nan values."""
        assert not np.any(np.isnan(self.spectrum.pow_gw))

    def test_pow_gw_expanded(self) -> None:
        """Test that the expanded GW power spectrum has no nan values."""
        assert not np.any(np.isnan(self.spectrum.pow_gw_expanded))

    def test_pow_gw_int(self) -> None:
        """Test that the integrated GW power spectrum has no nan values."""
        assert not np.any(np.isnan(self.spectrum.pow_gw_int))

    def test_pow_gw_low(self) -> None:
        """Test that the low-frequency GW power spectrum has no nan values."""
        assert not np.any(np.isnan(self.spectrum.pow_gw_low))

    def test_pow_gw_ssm(self) -> None:
        """Test that the SSM GW power spectrum has no nan values."""
        assert not np.any(np.isnan(self.spectrum.pow_gw_ssm))

    def test_pow_v(self) -> None:
        """Test that the velocity power spectrum has no nan values."""
        assert not np.any(np.isnan(self.spectrum.pow_v))

    def test_pow_v_tilde(self) -> None:
        """Test that the power spectrum of the velocity field v_tilde has no nan values."""
        assert not np.any(np.isnan(self.spectrum.pow_v_tilde))

    def test_source_lifetime_factor(self) -> None:
        """Test that the source lifetime factor is positive."""
        assert self.spectrum.source_lifetime_factor > 0

    def test_spec_den_gw_scaling(self) -> None:
        """Test that the scaling factor of the GW spectral density is positive."""
        assert self.spectrum.spec_den_gw_scaling > 0

    def test_spec_den_v_tilde(self) -> None:
        """Test that the spectral density of the velocity field has no nan values."""
        assert not np.any(np.isnan(self.spectrum.spec_den_v_tilde))

    def test_suppression_factor(self) -> None:
        """Test that the suppression factor is positive."""
        assert self.spectrum.suppression_factor > 0

    def test_tau_end(self) -> None:
        """Test that the conformal time when the anisotropic stress turns off is positive."""
        assert self.spectrum.tau_end > 0

    def test_tau_star(self) -> None:
        """Test that the conformal time when the anisotropic stress turns on is positive."""
        assert self.spectrum.tau_star > 0

    def test_ubarf(self) -> None:
        """Test that the RMS fluid velocity is positive also for a custom nucleation type."""
        assert self.spectrum.ubarf > 0
        assert self.spectrum.ubarf_custom_nucleation(nuc_type=DEFAULT_NUC_TYPE) > 0

    def test_tau_order(self) -> None:
        """Test that the anisotropic stress turns off after it turns on."""
        assert self.spectrum.tau_end > self.spectrum.tau_star

    def test_z_cross_approx(self) -> None:
        """Test that the approximate crossover point of the low and intermediate frequency forms is positive."""
        assert self.spectrum.z_cross_approx > 0


class SSMSpectrumWithParamsTest(unittest.TestCase):
    """Tests for creating new spectra by reusing the velocity spectra of an existing spectrum."""

    bubble: tp.ClassVar[Bubble]
    spectrum: tp.ClassVar[SSMSpectrum]
    #: Low-resolution settings for faster tests
    KWARGS: tp.ClassVar[dict[str, tp.Any]] = {"y": np.logspace(-1, 3, 300), "nT": 2000, "n_z_lookup": 2000}

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        model = ConstCSModel(css2=1/3-0.01, csb2=1/3-0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        cls.bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
        cls.spectrum = SSMSpectrum(cls.bubble, beta_tilde=10, **cls.KWARGS)

    def assert_same(self, spectrum: SSMSpectrum, ref: SSMSpectrum) -> None:
        """Assert that a spectrum created with with_params() is the same as a fully computed one."""
        assert spectrum.r_star == ref.r_star
        assert spectrum.beta_tilde == ref.beta_tilde
        assert spectrum.label_unicode == ref.label_unicode
        for name in ("nucleation_f", "hx", "bubble_spacing_enlargement_factor", "ubarf2", "cs"):
            assert getattr(spectrum, name) == getattr(ref, name), name
        for name in (
                "spec_den_v", "spec_den_v_lookup", "a2", "a2_lookup", "z_lookup",
                "spec_den_gw", "spec_den_gw_ssm", "spec_den_gw_low", "spec_den_gw_int", "spec_den_gw_expanded",
                "pow_gw", "pow_gw_rate"):
            np.testing.assert_allclose(getattr(spectrum, name), getattr(ref, name), rtol=1e-14, err_msg=name)

    def test_r_star(self) -> None:
        r"""A new $r_*$ should give the same result as a new spectrum, without recomputing the velocity spectra."""
        with mock.patch("pttools.ssm.spectrum.compute") as compute_mock:
            spectrum = self.spectrum.with_params(r_star=0.3)
            compute_mock.assert_not_called()
        # The velocity spectra are shared
        assert spectrum.base_res is self.spectrum.base_res
        assert spectrum.beta_tilde is None
        assert spectrum.nucleation_f == 0
        assert spectrum.bubble_spacing_enlargement_factor == 1
        self.assert_same(spectrum, SSMSpectrum(self.bubble, r_star=0.3, **self.KWARGS))
        # The original spectrum should not be modified
        assert self.spectrum.beta_tilde == 10

    def test_beta_tilde(self) -> None:
        r"""A new $\tilde{\beta}$ should also give a new $r_*$ and bubble spacing enlargement factor."""
        spectrum = self.spectrum.with_params(beta_tilde=20)
        assert spectrum.r_star < self.spectrum.r_star
        self.assert_same(spectrum, SSMSpectrum(self.bubble, beta_tilde=20, **self.KWARGS))

    def test_other_params(self) -> None:
        """The other parameters that do not affect the velocity spectra can also be changed."""
        spectrum = self.spectrum.with_params(N_sh=3, a_star_a_r_ratio=0.5, omega_barotropic=0.3)
        assert spectrum.beta_tilde == 10
        self.assert_same(
            spectrum,
            SSMSpectrum(self.bubble, beta_tilde=10, N_sh=3, a_star_a_r_ratio=0.5, omega_barotropic=0.3, **self.KWARGS)
        )

    def test_source_lifetime_factor_scaling(self) -> None:
        r"""The source lifetime factor $\Upsilon_\ell$ should be a mere scaling factor of $\tilde{P}_\text{gw,ssm}$."""
        spectrum = self.spectrum
        np.testing.assert_array_equal(
            spectrum.spec_den_gw_ssm, spectrum.source_lifetime_factor * spectrum.spec_den_gw_ssm_unscaled)
        ref, _ = spec_den_gw(
            z_lookup=spectrum.z_lookup, P_tilde_v_lookup=spectrum.spec_den_v_lookup, y=spectrum.y,
            cs=spectrum.cs, source_lifetime_factor=spectrum.source_lifetime_factor
        )
        np.testing.assert_allclose(spectrum.spec_den_gw_ssm, ref, rtol=1e-14)

    def test_covers(self) -> None:
        """The range of the y array should be checked with a relative tolerance for the endpoints."""
        y = self.spectrum.y
        assert self.spectrum.covers(y)
        assert self.spectrum.covers(y[10:-10])
        assert self.spectrum.covers(np.array([y[0] * (1 - 1e-12), y[-1] * (1 + 1e-12)]))
        assert not self.spectrum.covers(np.array([y[0] * 0.99]))
        assert not self.spectrum.covers(np.array([y[-1] * 1.01]))
        assert not self.spectrum.covers(np.array([]))

    def test_y_nodes(self) -> None:
        """A new y array at the points of this spectrum should give the same spectra, except near the junction.

        The interpolated arrays are exact at the points,
        whereas the junction of the low-k approximation depends on the points of the y array.
        """
        indices = slice(10, -10, 3)
        y = self.spectrum.y[indices]
        with mock.patch("pttools.ssm.spectrum.compute") as compute_mock:
            spectrum = self.spectrum.with_params(r_star=0.3, y=y)
            compute_mock.assert_not_called()
        ref = self.spectrum.with_params(r_star=0.3)
        np.testing.assert_array_equal(spectrum.y, y)
        assert spectrum.a2 is self.spectrum.a2
        for name in ("spec_den_v", "spec_den_gw_ssm", "spec_den_gw_low", "spec_den_gw_int"):
            np.testing.assert_allclose(getattr(spectrum, name), getattr(ref, name)[indices], rtol=1e-14, err_msg=name)

    def test_y_interpolated(self) -> None:
        r"""A new y array should give the same spectrum as a new spectrum computed for it, within the accuracy.

        The spectra differ by the interpolation error and by the dependence of the lookup arrays on the range of y,
        and they are therefore compared only near the peak of $\mathcal{P}_\text{gw}$.
        """
        y = np.logspace(-0.5, 2.5, 200)
        spectrum = self.spectrum.with_params(beta_tilde=20, y=y)
        kwargs: dict[str, tp.Any] = {**self.KWARGS, "y": y}
        ref = SSMSpectrum(self.bubble, beta_tilde=20, **kwargs)
        assert spectrum.r_star == ref.r_star
        np.testing.assert_array_equal(spectrum.y, y)
        significant = ref.pow_gw > 1e-2 * ref.pow_gw.max()
        np.testing.assert_allclose(spectrum.pow_gw[significant], ref.pow_gw[significant], rtol=1e-2)
        # The original spectrum should not be modified
        np.testing.assert_array_equal(self.spectrum.y, self.KWARGS["y"])

    def test_y_out_of_range(self) -> None:
        """A new y array outside the range of the spectrum cannot be interpolated."""
        with pytest.raises(ValueError, match="not within the range"):
            self.spectrum.with_params(y=np.logspace(-2, 2, 10))

    def test_not_computed(self) -> None:
        """A spectrum that has not been computed cannot be reused."""
        spectrum = SSMSpectrum(self.bubble, r_star=0.1, compute=False)
        with pytest.raises(RuntimeError, match="has to be computed"):
            spectrum.with_params(r_star=0.2)


if __name__ == "__main__":
    unittest.main()
