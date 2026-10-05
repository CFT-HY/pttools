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
from pttools.omgw0.factors import F_gw0_h2
from tests.utils import TEST_JSON_PATH


class SpectrumTest(unittest.TestCase):
    """Tests for the Spectrum class."""

    spectrum: tp.ClassVar[Spectrum]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        model = ConstCSModel(css2=1/3-0.01, csb2=1/3-0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
        cls.spectrum = Spectrum(bubble, r_star=0.1)

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
        with mock.patch("pttools.omgw0.spectrum.freq.f", wraps=freq.f) as f_mock:
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
        spectrum = Spectrum(self.spectrum.bubble, r_star=0.1, f=f)
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
            freq.z(f=spectrum.f(z), T_star=T_star, r_star=spectrum.r_star, ge_star=ge_star, gs_star=gs_star),
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
        sm = StandardModel(V_s=5e12, g_mult_s=1 + 1e-9, silence_temp=True)
        model = FullModel(sm, T_crit_guess=100e3)
        cls.bubble = Bubble(model, v_wall=0.3, alpha_n=0.05)
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
