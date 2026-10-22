"""Tests for the accuracy settings."""

import unittest

import numpy as np

from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.ssm import Accuracy, SSMSpectrum


class AccuracyTest(unittest.TestCase):
    """Tests for :py:class:`pttools.ssm.accuracy.Accuracy`."""

    @staticmethod
    def test_spectrum_kwargs() -> None:
        """The settings should be given to the spectra as arguments."""
        accuracy = Accuracy(n_xi=1000, n_z_lookup=500, nx_P_tilde_gw=100, nT=300)
        assert accuracy.spectrum_kwargs() == {"n_z_lookup": 500, "nx_P_tilde_gw": 100, "nT": 300}
        bubble = Bubble.cached(BagModel.cached(a_s=1.1, a_b=1, V_s=1), v_wall=0.5, alpha_n=0.1, n_xi=accuracy.n_xi)
        spectrum = SSMSpectrum(
            bubble, r_star=0.1, y=np.logspace(-1, 2, 50), compute=False, **accuracy.spectrum_kwargs())
        assert (spectrum.n_z_lookup, spectrum.nx_P_tilde_gw, spectrum.nT) == (500, 100, 300)


if __name__ == "__main__":
    unittest.main()
