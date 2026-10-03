"""Tests for the Spectrum class."""

import typing as tp
import unittest
from unittest import mock

import numpy as np

from pttools.bubble import Bubble
from pttools.models import ConstCSModel
from pttools.omgw0 import Spectrum, freq
from tests.utils import TEST_JSON_PATH


class SpectrumTest(unittest.TestCase):
    """Tests for the Spectrum class."""

    spectrum: Spectrum

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
        self.assertEqual(self.spectrum.f_min, f.min())
        self.assertEqual(self.spectrum.f_max, f.max())
        self.assertGreater(self.spectrum.f_max, self.spectrum.f_min)

    def test_f_computed_once(self) -> None:
        """The frequencies of the y array should be computed only once, and a custom z should not affect them."""
        spectrum = Spectrum(self.spectrum.bubble, r_star=0.1, compute=False)
        z = np.array([1., 10.])
        with mock.patch("pttools.omgw0.spectrum.freq.f", wraps=freq.f) as f_mock:
            f = spectrum.f()
            f_z = spectrum.f(z)
            f_min = spectrum.f_min
            f_max = spectrum.f_max
            self.assertIs(spectrum.f(), f)
            # One call for the y array and one for the custom z
            self.assertEqual(f_mock.call_count, 2)
        np.testing.assert_array_equal(f, freq.f(z=spectrum.y, r_star=spectrum.r_star, f_star0=spectrum.f_star0))
        np.testing.assert_array_equal(f_z, freq.f(z=z, r_star=spectrum.r_star, f_star0=spectrum.f_star0))
        self.assertEqual(f_min, f.min())
        self.assertEqual(f_max, f.max())

    def test_f_given(self) -> None:
        """The frequencies given as an argument should be returned as is, and correspond to the y array."""
        f = np.logspace(-6, -1, 100)
        params = ((0.1, None, None, None), (None, 100, None, None), (0.2, None, 1000, 50))
        for r_star, beta_tilde, T_star, g_star in params:
            with self.subTest(r_star=r_star, beta_tilde=beta_tilde, T_star=T_star, g_star=g_star):
                spectrum = Spectrum(
                    self.spectrum.bubble, r_star=r_star, beta_tilde=beta_tilde, f=f,
                    T_star=T_star, g_star=g_star, compute=False)
                self.assertIs(spectrum.f(), f)
                self.assertEqual(spectrum.f_min, f.min())
                self.assertEqual(spectrum.f_max, f.max())
                np.testing.assert_allclose(
                    freq.f(z=spectrum.y, r_star=spectrum.r_star, f_star0=spectrum.f_star0), f, rtol=1e-14)
                np.testing.assert_allclose(spectrum.y, spectrum.z_from_f(f), rtol=1e-14)

    def test_f_given_spectrum(self) -> None:
        """A spectrum given the frequencies of another spectrum should be the same as the other spectrum."""
        f = self.spectrum.f()
        spectrum = Spectrum(self.spectrum.bubble, r_star=0.1, f=f)
        self.assertIs(spectrum.f(), f)
        np.testing.assert_allclose(spectrum.y, self.spectrum.y, rtol=1e-14)
        np.testing.assert_allclose(spectrum.omgw0(), self.spectrum.omgw0(), rtol=1e-10)

    def test_f_given_invalid(self) -> None:
        """Test that giving both y and f, or giving non-finite frequencies, raises an error."""
        with self.assertRaises(ValueError):
            Spectrum(self.spectrum.bubble, r_star=0.1, y=np.array([1., 10.]), f=np.array([1e-3, 1e-2]), compute=False)
        with self.assertRaises(ValueError):
            Spectrum(self.spectrum.bubble, r_star=0.1, f=np.array([1e-3, np.nan]), compute=False)

    def test_noise(self) -> None:
        """Test that the signal-to-noise ratio is positive."""
        self.assertGreater(self.spectrum.snr()[0], 0)

    def test_noise_instrument(self) -> None:
        """Test that the signal-to-noise ratio for the instrument noise is positive."""
        self.assertGreater(self.spectrum.snr_ins()[0], 0)

    def test_peak(self) -> None:
        """Test that the peak frequency is positive and the peak amplitude is between 0 and 1."""
        peak = self.spectrum.omgw0_peak()
        self.assertGreater(peak[0], 0)
        self.assertGreater(peak[1], 0)
        self.assertLess(peak[1], 1)

    def test_R_star(self) -> None:
        r"""Test that $0 < R_* < 1 \text{mm}$."""
        self.assertGreater(self.spectrum.R_star, 0)
        self.assertLess(self.spectrum.R_star, 1e-3)

    def test_spectrum(self) -> None:
        """Test that the spectrum has no nan values."""
        self.assertEqual(np.isnan(self.spectrum.omgw0()).sum(), 0)

    def test_total(self) -> None:
        """Test that the total power is the integral of the spectrum over the frequency."""
        val = self.spectrum.omgw0_total()
        ref = np.trapezoid(y=self.spectrum.omgw0(), x=self.spectrum.f())
        self.assertAlmostEqual(val, ref)


if __name__ == "__main__":
    unittest.main()
