"""Tests for the Spectrum class."""

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
    def setUpClass(cls) -> None:
        model = ConstCSModel(css2=1/3-0.01, csb2=1/3-0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        bubble = Bubble(model, v_wall=0.5, alpha_n=0.2)
        cls.spectrum = Spectrum(bubble, r_star=0.1)

    def test_export(self) -> None:
        self.spectrum.export(TEST_JSON_PATH / "spectrum.json")

    def test_f_min_max(self) -> None:
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

    def test_noise(self) -> None:
        self.assertGreater(self.spectrum.snr()[0], 0)

    def test_noise_instrument(self) -> None:
        self.assertGreater(self.spectrum.snr_ins()[0], 0)

    def test_peak(self) -> None:
        peak = self.spectrum.omgw0_peak()
        self.assertGreater(peak[0], 0)
        self.assertGreater(peak[1], 0)
        self.assertLess(peak[1], 1)

    def test_R_star(self) -> None:
        r"""Test that $0 < R_* < 1 \text{mm}$."""
        self.assertGreater(self.spectrum.R_star, 0)
        self.assertLess(self.spectrum.R_star, 1e-3)

    def test_spectrum(self) -> None:
        self.assertEqual(np.isnan(self.spectrum.omgw0()).sum(), 0)

    def test_total(self) -> None:
        val = self.spectrum.omgw0_total()
        ref = np.trapezoid(y=self.spectrum.omgw0(), x=self.spectrum.f())
        self.assertAlmostEqual(val, ref)


if __name__ == "__main__":
    unittest.main()
