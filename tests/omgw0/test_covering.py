"""Tests for the $z$ arrays and accuracy settings that cover given frequencies."""

import dataclasses
import typing as tp
import unittest

import numpy as np
import pytest

from pttools.bubble import Bubble
from pttools.models import BagModel
from pttools.omgw0 import Spectrum, freq
from pttools.ssm.nucleation import DEFAULT_NUC_TYPE
from pttools.ssm.settings import SSMSettings

F: np.ndarray = np.logspace(-5, -1, 81)
T_STAR: float = 100
G_STAR: float = 100
SETTINGS: SSMSettings = SSMSettings(n_z_lookup=1000, nT=1000, nx_P_tilde_gw=200)


class CoveringTest(unittest.TestCase):
    """Tests for the methods of :py:class:`pttools.omgw0.spectrum.Spectrum` that cover given frequencies."""

    bubble: tp.ClassVar[Bubble]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        """Create the bubble of the spectra."""
        cls.bubble = Bubble.cached(BagModel.cached(alpha_n_min=0.01), v_wall=0.5, alpha_n=0.1)

    def test_n_z_lookup_covering(self) -> None:
        """The number of points should be scaled by the ratio of the decades, but not reduced."""
        assert Spectrum.n_z_lookup_covering(z=np.logspace(0, 8, 10), f=F, n_z_lookup=1000) == 2000
        assert Spectrum.n_z_lookup_covering(z=np.logspace(0, 2, 10), f=F, n_z_lookup=1000) == 1000

    def test_settings_covering(self) -> None:
        """Only n_z_lookup should be scaled for a z array that spans more decades than the frequencies."""
        z = Spectrum.z_covering(f=F, bubble=self.bubble, T_star=T_STAR, g_star=G_STAR, r_star=np.array([1e-3, 0.1]))
        settings = Spectrum.settings_covering(z=z, f=F, settings=SETTINGS)
        assert settings.n_z_lookup == Spectrum.n_z_lookup_covering(z=z, f=F, n_z_lookup=SETTINGS.n_z_lookup)
        assert settings.n_z_lookup > SETTINGS.n_z_lookup
        assert settings == dataclasses.replace(SETTINGS, n_z_lookup=settings.n_z_lookup)
        assert Spectrum.settings_covering(z=z, f=F).n_z_lookup > SSMSettings().n_z_lookup

    def test_z_covering(self) -> None:
        """The z array should cover the frequencies for all the given values, ignoring the invalid ones."""
        f_star0 = freq.f_star0(T_star=T_STAR, ge_star=G_STAR, gs_star=G_STAR) * 1.1
        for name, values in (("beta_tilde", (10., 100., 1000.)), ("r_star", (1e-3, 0.01, 0.1))):
            arr = np.array([*values, np.nan, -1.])
            z = Spectrum.z_covering(
                f=F, bubble=self.bubble, T_star=T_STAR, g_star=G_STAR, f_star0_factor=1.1,
                beta_tilde=arr if name == "beta_tilde" else None,
                r_star=arr if name == "r_star" else None
            )
            assert np.all(np.diff(z) > 0)
            r_stars = values if name == "r_star" else [
                Spectrum.validate_nucleation(bubble=self.bubble, nuc_type=DEFAULT_NUC_TYPE, r_star=None, beta_tilde=b)
                for b in values
            ]
            # The smallest and largest r_star determine the endpoints.
            assert z[0] == freq.z(f=F[0], r_star=min(r_stars), f_star0=f_star0)
            assert z[-1] == freq.z(f=F[-1], r_star=max(r_stars), f_star0=f_star0)
            # The default density is that of the frequencies.
            np.testing.assert_allclose(
                (z.size - 1) / np.log10(z[-1] / z[0]), (F.size - 1) / np.log10(F[-1] / F[0]), rtol=1e-2)
        z = Spectrum.z_covering(
            f=F, bubble=self.bubble, T_star=T_STAR, g_star=G_STAR, r_star=np.array([0.1]), points_per_decade=5)
        assert z.size == 21

    def test_z_covering_invalid(self) -> None:
        """Exactly one of beta_tilde and r_star should be given, and at least one value should be valid."""
        with pytest.raises(ValueError, match="Exactly one"):
            Spectrum.z_covering(f=F, bubble=self.bubble, T_star=T_STAR, g_star=G_STAR)
        with pytest.raises(ValueError, match="Exactly one"):
            Spectrum.z_covering(
                f=F, bubble=self.bubble, T_star=T_STAR, g_star=G_STAR,
                beta_tilde=np.array([10.]), r_star=np.array([0.1]))
        with pytest.raises(ValueError, match="None of"):
            Spectrum.z_covering(f=F, bubble=self.bubble, T_star=T_STAR, g_star=G_STAR, beta_tilde=np.array([np.nan]))


if __name__ == "__main__":
    unittest.main()
