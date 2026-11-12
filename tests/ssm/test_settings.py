"""Tests for the accuracy settings of the bubbles and spectra."""

import unittest

import numpy as np

from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.ssm import SSMSettings, SSMSpectrum


class SSMSettingsTest(unittest.TestCase):
    """Tests for :py:class:`pttools.ssm.settings.SSMSettings`."""

    @staticmethod
    def test_defaults() -> None:
        """The default settings should be the defaults of the bubbles and spectra."""
        settings = SSMSettings()
        bubble = Bubble.cached(BagModel.cached(a_s=1.1, a_b=1, V_s=1), v_wall=0.5, alpha_n=0.1)
        spectrum = SSMSpectrum(bubble, r_star=0.1, y=np.logspace(-1, 2, 50), compute=False)
        assert settings.n_xi == bubble.n_xi
        assert (spectrum.n_z_lookup, spectrum.nx_P_tilde_gw, spectrum.nT) == (
            settings.n_z_lookup, settings.nx_P_tilde_gw, settings.nT)

    @staticmethod
    def test_spectrum_kwargs() -> None:
        """The settings should be given to the spectra as arguments."""
        accuracy = SSMSettings(n_xi=1000, n_z_lookup=500, nx_P_tilde_gw=100, nT=300)
        assert accuracy.spectrum_kwargs() == {"n_z_lookup": 500, "nx_P_tilde_gw": 100, "nT": 300}
        bubble = Bubble.cached(BagModel.cached(a_s=1.1, a_b=1, V_s=1), v_wall=0.5, alpha_n=0.1, n_xi=accuracy.n_xi)
        spectrum = SSMSpectrum(
            bubble, r_star=0.1, y=np.logspace(-1, 2, 50), compute=False, **accuracy.spectrum_kwargs())
        assert (spectrum.n_z_lookup, spectrum.nx_P_tilde_gw, spectrum.nT) == (500, 100, 300)


if __name__ == "__main__":
    unittest.main()
