"""Tests for the Spectrum class."""

import typing as tp
import unittest
from unittest import mock

import numpy as np
import pytest

from pttools.bubble import Bubble
from pttools.bubble.phase import Phase
from pttools.models import ConstCSModel, FullModel, StandardModel
from pttools.omgw0 import Spectrum, freq
from pttools.omgw0.compute import SNRSettings
from pttools.omgw0.factors import F_gw0_h2
from tests.utils import LOW_ACCURACY_SPECTRUM_KWARGS, TEST_JSON_PATH


class SpectrumTest(unittest.TestCase):
    """Tests for the Spectrum class.

    The spectra are computed with low accuracy, as they are not compared with reference values.
    """

    spectrum: tp.ClassVar[Spectrum]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        model = ConstCSModel.cached(css2=1/3-0.01, csb2=1/3-0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        bubble = Bubble.cached(model, v_wall=0.5, alpha_n=0.2)
        cls.spectrum = Spectrum.cached(bubble, r_star=0.1, **LOW_ACCURACY_SPECTRUM_KWARGS)

    def test_export(self) -> None:
        """Test that the spectrum can be exported as JSON."""
        self.spectrum.export(TEST_JSON_PATH / "spectrum.json")

    def test_f_min_max(self) -> None:
        """Test that the minimum and maximum frequencies correspond to the frequency array."""
        f = self.spectrum.f()
        assert self.spectrum.f_min == f.min()
        assert self.spectrum.f_max == f.max()
        assert self.spectrum.f_max > self.spectrum.f_min

    def test_f_computed_once(self) -> None:
        """The frequencies of the y array should be computed only once, and a custom z should not affect them."""
        spectrum = Spectrum(self.spectrum.bubble, r_star=0.1, compute=False)
        z = np.array([1., 10.])
        with mock.patch("pttools.omgw0.base.freq.f", wraps=freq.f) as f_mock:
            f = spectrum.f()
            f_z = spectrum.f(z)
            f_min = spectrum.f_min
            f_max = spectrum.f_max
            assert spectrum.f() is f
            # One call for the y array and one for the custom z
            assert f_mock.call_count == 2
        np.testing.assert_array_equal(f, freq.f(z=spectrum.y, r_star=spectrum.r_star, f_star0=spectrum.f_star0))
        np.testing.assert_array_equal(f_z, freq.f(z=z, r_star=spectrum.r_star, f_star0=spectrum.f_star0))
        assert f_min == f.min()
        assert f_max == f.max()

    def test_f_given(self) -> None:
        """The frequencies given as an argument should be returned as is, and correspond to the y array."""
        f = np.logspace(-6, -1, 100)
        params = ((0.1, None, None, None), (None, 100, None, None), (0.2, None, 1000, 50))
        for r_star, beta_tilde, T_star, g_star in params:
            with self.subTest(r_star=r_star, beta_tilde=beta_tilde, T_star=T_star, g_star=g_star):
                spectrum = Spectrum(
                    self.spectrum.bubble, r_star=r_star, beta_tilde=beta_tilde, f=f,
                    T_star=T_star, g_star=g_star, compute=False)
                assert spectrum.f() is f
                assert spectrum.f_min == f.min()
                assert spectrum.f_max == f.max()
                np.testing.assert_allclose(
                    freq.f(z=spectrum.y, r_star=spectrum.r_star, f_star0=spectrum.f_star0), f, rtol=1e-14)
                np.testing.assert_allclose(spectrum.y, spectrum.z_from_f(f), rtol=1e-14)

    def test_f_given_spectrum(self) -> None:
        """A spectrum given the frequencies of another spectrum should be the same as the other spectrum."""
        f = self.spectrum.f()
        spectrum = Spectrum(self.spectrum.bubble, r_star=0.1, f=f, **LOW_ACCURACY_SPECTRUM_KWARGS)
        assert spectrum.f() is f
        np.testing.assert_allclose(spectrum.y, self.spectrum.y, rtol=1e-14)
        np.testing.assert_allclose(spectrum.omgw0(), self.spectrum.omgw0(), rtol=1e-10)

    def test_f_given_invalid(self) -> None:
        """Test that giving both y and f, or giving non-finite frequencies, raises an error."""
        with pytest.raises(ValueError, match="Either y or f can be provided, but not both"):
            Spectrum(self.spectrum.bubble, r_star=0.1, y=np.array([1., 10.]), f=np.array([1e-3, 1e-2]), compute=False)
        with pytest.raises(ValueError, match="must not contain nan values"):
            Spectrum(self.spectrum.bubble, r_star=0.1, f=np.array([1e-3, np.nan]), compute=False)

    def test_degrees_of_freedom(self) -> None:
        r"""Test that $g_{e,*}$ and $g_{s,*}$ are used for the redshift of the frequencies and the power.

        The given $g_*$ is that of pressure, and $g_{e,*} = \frac{1}{3}(4 g_{s,*} - g_*)$.
        """
        T_star = 200.
        g_star = 90.
        gs_star = 100.
        ge_star = (4 * gs_star - g_star) / 3
        spectrum = Spectrum(
            self.spectrum.bubble, r_star=0.1, T_star=T_star, g_star=g_star, gs_star=gs_star, compute=False)
        assert spectrum.ge_star == pytest.approx(ge_star, abs=5e-13)
        np.testing.assert_allclose(
            spectrum.f_star0, freq.f_star0(T_star=T_star, ge_star=ge_star, gs_star=gs_star), rtol=1e-14)
        np.testing.assert_allclose(spectrum.F_gw0_h2(), F_gw0_h2(ge_star=ge_star, gs_star=gs_star), rtol=1e-14)
        z = np.array([1., 10.])
        np.testing.assert_allclose(
            spectrum.z_from_f(spectrum.f(z)),
            freq.z_full(f=spectrum.f(z), T_star=T_star, r_star=spectrum.r_star, ge_star=ge_star, gs_star=gs_star),
            rtol=1e-14
        )
        np.testing.assert_allclose(spectrum.z_from_f(spectrum.f(z)), z, rtol=1e-14)

    def test_degrees_of_freedom_g_star_only(self) -> None:
        r"""Test that $g_{s,*} = g_{e,*} = g_*$ when only $g_*$ is given."""
        spectrum = Spectrum(self.spectrum.bubble, r_star=0.1, T_star=200., g_star=106.75, compute=False)
        assert spectrum.gs_star == 106.75
        assert spectrum.ge_star == pytest.approx(106.75, abs=5e-13)
        np.testing.assert_allclose(spectrum.f_star0, freq.f_star0(T_star=200., ge_star=106.75), rtol=1e-14)
        np.testing.assert_allclose(spectrum.F_gw0_h2(), F_gw0_h2(ge_star=106.75), rtol=1e-14)

    def test_noise(self) -> None:
        """Test that the signal-to-noise ratio is positive."""
        assert self.spectrum.snr()[0] > 0

    def test_noise_instrument(self) -> None:
        """Test that the signal-to-noise ratio for the instrument noise is positive."""
        assert self.spectrum.snr_ins()[0] > 0

    def test_peak(self) -> None:
        """Test that the peak frequency is positive and the peak amplitude is between 0 and 1."""
        peak = self.spectrum.omgw0_peak()
        assert peak[0] > 0
        assert peak[1] > 0
        assert peak[1] < 1

    def test_H_star(self) -> None:
        r"""Test that $H_* = \sqrt{\frac{8\pi}{3} e_*}$ with $G = 1$, and that $R_* = r_* / H_*$."""
        e_star = np.pi**2 / 30 * self.spectrum.ge_star * self.spectrum.T_star**4
        assert self.spectrum.H_star == pytest.approx(np.sqrt(8 * np.pi * e_star / 3), rel=1e-14)
        assert self.spectrum.R_star == pytest.approx(self.spectrum.r_star / self.spectrum.H_star, rel=1e-14)

    def test_R_star(self) -> None:
        r"""Test that $0 < R_* < 1 \text{mm}$."""
        assert self.spectrum.R_star > 0
        assert self.spectrum.R_star < 1e-3

    def test_spectrum(self) -> None:
        """Test that the spectrum has no nan values."""
        assert np.isnan(self.spectrum.omgw0()).sum() == 0

    def test_total(self) -> None:
        """Test that the total power is the integral of the spectrum over the frequency."""
        val = self.spectrum.omgw0_total()
        ref = np.trapezoid(y=self.spectrum.omgw0(), x=self.spectrum.f())
        assert val == pytest.approx(ref, abs=5e-8)


class SpectrumComputeTest(unittest.TestCase):
    """Tests for reusing the computed spectra and for computing the signal-to-noise ratio together with them."""

    bubble: tp.ClassVar[Bubble]
    spectrum: tp.ClassVar[Spectrum]
    #: Low-resolution settings for faster tests
    KWARGS: tp.ClassVar[dict[str, tp.Any]] = {"nT": 2000, "n_z_lookup": 2000}

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        model = ConstCSModel.cached(css2=1/3-0.01, csb2=1/3-0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        cls.bubble = Bubble.cached(model, v_wall=0.5, alpha_n=0.2)
        cls.spectrum = Spectrum.cached(
            cls.bubble, r_star=0.1, f=np.logspace(-5, -1, 300), T_star=200, g_star=100, compute_snr=True, **cls.KWARGS)

    def test_snr_precomputed(self) -> None:
        """The precomputed signal-to-noise ratio should be the same as one computed afterwards."""
        with mock.patch("pttools.omgw0.base.signal_to_noise_ratio") as snr_mock:
            snr = self.spectrum.snr()
            snr_mock.assert_not_called()
        assert snr[0] > 0
        assert self.spectrum.snr_settings.matches(SNRSettings())
        spectrum = Spectrum.cached(
            self.bubble, r_star=0.1, f=self.spectrum.f(), T_star=200, g_star=100, compute_snr=False, **self.KWARGS)
        ref = spectrum.snr()
        for val, ref_val in zip(snr, ref, strict=True):
            np.testing.assert_allclose(val, ref_val, rtol=1e-14)
        np.testing.assert_allclose(self.spectrum.omgw0_h2(), spectrum.omgw0_h2(), rtol=1e-14)

    def test_snr_custom_settings(self) -> None:
        """The precomputed result should be used only when the settings match."""
        spectrum = self.spectrum.with_params(r_star=0.2)
        spectrum.compute(compute_snr=True, snr_settings=SNRSettings(noise_eb=False, noise_gb=False))
        with mock.patch("pttools.omgw0.base.signal_to_noise_ratio") as snr_mock:
            snr_ins = spectrum.snr_ins()[0]
            snr_mock.assert_not_called()
        snr = spectrum.snr()[0]
        assert snr < snr_ins
        spectrum.compute(compute_snr=False)
        assert spectrum.snr_ins()[0] == pytest.approx(snr_ins, rel=1e-14)
        assert spectrum.snr()[0] == pytest.approx(snr, rel=1e-14)

    def test_with_params(self) -> None:
        r"""Changing $r_*$, $T_*$ and $g_*$ should give the same result as a new spectrum with the same $z$."""
        with mock.patch("pttools.omgw0.spectrum.compute") as compute_mock:
            spectrum = self.spectrum.with_params(r_star=0.2, T_star=1000, g_star=50)
            compute_mock.assert_not_called()
        assert spectrum.base_res is self.spectrum.base_res
        assert spectrum.compute_snr
        assert (spectrum.r_star, spectrum.T_star, spectrum.g_star, spectrum.gs_star) == (0.2, 1000, 50, 50)
        # The z array is kept, and therefore the frequencies change.
        np.testing.assert_array_equal(spectrum.y, self.spectrum.y)
        assert not spectrum.f_given
        ref = Spectrum.cached(
            self.bubble, r_star=0.2, y=self.spectrum.y, T_star=1000, g_star=50, compute_snr=True, **self.KWARGS)
        np.testing.assert_allclose(spectrum.f(), ref.f(), rtol=1e-14)
        np.testing.assert_allclose(spectrum.omgw0_h2(), ref.omgw0_h2(), rtol=1e-14)
        assert spectrum.snr()[0] == pytest.approx(ref.snr()[0], rel=1e-14)
        # The original spectrum should not be modified
        assert (self.spectrum.r_star, self.spectrum.T_star, self.spectrum.g_star) == (0.1, 200, 100)

    def test_with_params_f(self) -> None:
        """Giving the frequencies should interpolate the spectrum to them with the parameters of the new spectrum.

        The spectra are compared only near the peak, see :py:meth:`tests.ssm.test_spectrum.SSMSpectrumWithParamsTest`.
        """
        f = np.logspace(-4.5, -2, 150)
        with mock.patch("pttools.omgw0.spectrum.compute") as compute_mock:
            spectrum = self.spectrum.with_params(r_star=0.2, g_star=50, f=f)
            compute_mock.assert_not_called()
        assert spectrum.f_given
        np.testing.assert_array_equal(spectrum.f(), f)
        np.testing.assert_allclose(spectrum.y, spectrum.z_from_f(f), rtol=1e-14)
        assert spectrum.a2 is self.spectrum.a2
        ref = Spectrum.cached(self.bubble, r_star=0.2, f=f, T_star=200, g_star=50, compute_snr=True, **self.KWARGS)
        np.testing.assert_allclose(spectrum.y, ref.y, rtol=1e-14)
        omgw0_h2 = spectrum.omgw0_h2()
        omgw0_h2_ref = ref.omgw0_h2()
        significant = omgw0_h2_ref > 1e-2 * omgw0_h2_ref.max()
        np.testing.assert_allclose(omgw0_h2[significant], omgw0_h2_ref[significant], rtol=1e-2)
        assert spectrum.snr()[0] == pytest.approx(ref.snr()[0], rel=2e-3)

    def test_with_params_y(self) -> None:
        """Giving the y array should interpolate the spectrum to it, and y and f cannot be given together."""
        y = self.spectrum.y[10:-10]
        spectrum = self.spectrum.with_params(r_star=0.2, y=y)
        assert not spectrum.f_given
        np.testing.assert_array_equal(spectrum.y, y)
        with pytest.raises(ValueError, match="not both"):
            self.spectrum.with_params(y=y, f=self.spectrum.f())

    def test_with_params_f_out_of_range(self) -> None:
        """Frequencies that do not correspond to the range of y cannot be interpolated."""
        with pytest.raises(ValueError, match="not within the range"):
            self.spectrum.with_params(r_star=0.2, f=self.spectrum.f())

    def test_with_params_noise(self) -> None:
        """The signal-to-noise ratio of the new spectrum should use the custom noise of the original spectrum."""
        f_noise = np.logspace(-4, -2, 50)
        settings = SNRSettings(noise=np.full(f_noise.size, 1e-12), f_noise=f_noise)
        spectrum = self.spectrum.with_params(r_star=0.3)
        spectrum.compute(snr_settings=settings)
        spectrum2 = spectrum.with_params(r_star=0.2)
        assert spectrum2.snr_settings is settings
        snr2 = spectrum2.snr(noise=settings.noise, f_noise=f_noise)
        np.testing.assert_array_equal(snr2[3], f_noise)
        spectrum2.compute(compute_snr=False)
        assert spectrum2.snr(noise=settings.noise, f_noise=f_noise)[0] == pytest.approx(snr2[0], rel=1e-14)


class StandardModelSpectrumTest(unittest.TestCase):
    r"""Tests for a spectrum of the Standard Model, whose temperature is in physical units of MeV.

    The temperature and the degrees of freedom at the time of GW production should be taken from the bubble,
    and the temperature should be converted to GeV.
    """

    bubble: tp.ClassVar[Bubble]
    spectrum: tp.ClassVar[Spectrum]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        sm = StandardModel.cached(V_s=5e12, g_mult_s=1 + 1e-9, silence_temp=True)
        model = FullModel.cached(sm, T_crit_guess=100e3)
        cls.bubble = Bubble.cached(model, v_wall=0.3, alpha_n=0.05)
        cls.spectrum = Spectrum(cls.bubble, r_star=0.1, compute=False)

    def test_model(self) -> None:
        """Test that the full model takes the temperature properties of the Standard Model."""
        assert self.bubble.model.temperature_is_physical
        assert self.bubble.model.temperature_unit_gev == 1e-3

    def test_T_star(self) -> None:
        r"""Test that $T_*$ is taken from the bubble and converted from MeV to GeV."""
        assert self.spectrum.T_star == pytest.approx(self.bubble.T_star * 1e-3, abs=5e-13)

    def test_g_star(self) -> None:
        r"""Test that $g_*$, $g_{s,*}$ and $g_{e,*}$ are those of the model after the bubble nucleation."""
        model = self.bubble.model
        w = self.bubble.va_enthalpy_density
        assert self.spectrum.g_star == self.bubble.g_star
        assert self.spectrum.gs_star == self.bubble.gs_star
        assert self.spectrum.g_star == pytest.approx(model.gp(w, Phase.BROKEN), abs=5e-11)
        assert self.spectrum.gs_star == pytest.approx(model.gs(w, Phase.BROKEN), abs=5e-11)
        assert self.spectrum.ge_star == pytest.approx(model.ge(w, Phase.BROKEN), abs=5e-9)


if __name__ == "__main__":
    unittest.main()
