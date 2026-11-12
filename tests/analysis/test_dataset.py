"""Tests for computing the spectra of the datasets."""

import typing as tp
import unittest

import numpy as np
import pytest

from pttools.analysis.dataset import compute_spectra, create_shared_spectrum
from pttools.analysis.grid import SpectrumParams
from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.ssm.nucleation import NucType
from pttools.ssm.settings import SSMSettings

F: np.ndarray = np.logspace(-5, -1, 81)
SETTINGS: SSMSettings = SSMSettings(n_z_lookup=1000, nT=300, nx_P_tilde_gw=100)
PARAMS: list[SpectrumParams] = [
    SpectrumParams(nuc_type=nuc_type, beta_tilde=beta_tilde, T_star=T_star, g_star=g_star)
    for nuc_type in NucType for beta_tilde in (100., 1000.) for T_star in (50., 500.) for g_star in (80., 130.)
]


class ComputeSpectraTest(unittest.TestCase):
    """Tests for :py:func:`pttools.analysis.dataset.compute_spectra`."""

    bubble: tp.ClassVar[Bubble]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        """Create the bubble of the spectra."""
        cls.bubble = Bubble.cached(BagModel.cached(alpha_n_min=0.01), v_wall=0.5, alpha_n=0.1)

    def test_shared(self) -> None:
        """The spectra created from shared spectra should be close to the separately computed spectra."""
        shared, n_skipped_shared = compute_spectra(self.bubble, PARAMS, F, SETTINGS, share_spectra=True)
        separate, n_skipped_separate = compute_spectra(self.bubble, PARAMS, F, SETTINGS, share_spectra=False)
        assert n_skipped_shared == n_skipped_separate == 0
        assert len(shared) == len(separate) == len(PARAMS)
        for params, spectrum, ref in zip(PARAMS, shared, separate, strict=True):
            assert (spectrum.nuc_type, spectrum.beta_tilde, spectrum.T_star, spectrum.g_star) == (
                params.nuc_type, params.beta_tilde, params.T_star, params.g_star)
            assert spectrum.f_given
            np.testing.assert_array_equal(spectrum.f(), F)
            assert spectrum.n_z_lookup > SETTINGS.n_z_lookup
            assert ref.n_z_lookup == SETTINGS.n_z_lookup
            omgw0_h2 = spectrum.omgw0_h2()
            omgw0_h2_ref = ref.omgw0_h2()
            peak = omgw0_h2_ref >= 1e-3 * omgw0_h2_ref.max()
            np.testing.assert_allclose(omgw0_h2[peak], omgw0_h2_ref[peak], rtol=1e-2)

    def test_shared_nuc_types(self) -> None:
        """A shared spectrum can be created only for a single nucleation type."""
        with pytest.raises(ValueError, match="nucleation type"):
            create_shared_spectrum(self.bubble, PARAMS, F, SETTINGS)
        spectrum = create_shared_spectrum(
            self.bubble, [params for params in PARAMS if params.nuc_type == NucType.SIMULTANEOUS], F, SETTINGS)
        assert spectrum.nuc_type == NucType.SIMULTANEOUS

    def test_skipped(self) -> None:
        """The spectra that cannot be computed should be skipped."""
        params = [*PARAMS[:2], SpectrumParams(nuc_type=NucType.EXPONENTIAL, beta_tilde=-1., T_star=100, g_star=100)]
        for share_spectra in (True, False):
            spectra, n_skipped = compute_spectra(self.bubble, params, F, SETTINGS, share_spectra=share_spectra)
            assert n_skipped == 1
            assert [spectrum.beta_tilde for spectrum in spectra] == [p.beta_tilde for p in params[:2]]


if __name__ == "__main__":
    unittest.main()
