"""Tests for importing bubbles and spectra from a file without recomputing them."""

import typing as tp
import unittest

from matplotlib.figure import Figure
import numpy as np
import pytest

from pttools.bubble import BaseNucleatedBubble, Bubble, ComputedBubble
from pttools.export import (
    Exporter,
    ImportedBubble,
    ImportedSpectrum,
    ImportedSSMSpectrum,
    Importer,
    NotImportedError,
    Preset,
    Table,
)
from pttools.models import ConstCSModel
from pttools.omgw0 import Spectrum
from pttools.omgw0.base_ssm import BaseSSMOmgw0
from pttools.plot.bubbles import plot_bubbles
from pttools.plot.spectra import plot_spectra
from pttools.ssm import SSMSpectrum
from pttools.ssm.spectrum_base_ssm import BaseSSMSpectrum
from tests.export.test_export import F_SPECTRUM_KWARGS, Y_SPECTRUM_KWARGS, new_path

#: Quantities of the bubbles that are computed from the stored profiles and parameters
BUBBLE_QUANTITIES: tuple[str, ...] = (
    "alpha_n", "v_wall", "sol_type", "wn", "Tn", "v_sh", "css2_Tn", "csb2_Tn", "en", "e_bar", "kappa", "kappa_giese",
    "omega", "ubarf", "ubarf2", "w_bar", "T_star", "g_star", "gs_star", "nu_gdh2024", "omega_barotropic",
    "mean_adiabatic_index", "kinetic_energy_fraction", "thermal_energy_fraction", "entropy_density_diff",
    "va_enthalpy_density", "va_trace_anomaly_diff", "vp_vm_tilde_ratio", "vp_vm_tilde_ratio_giese", "v_mu",
)
#: Quantities of the spectra that are computed from the stored fields and the bubble
SPECTRUM_QUANTITIES: tuple[str, ...] = (
    "r_star", "beta_tilde", "nuc_type", "cs2", "T_star", "g_star", "gs_star", "H_star", "R_star", "f_star0",
    "delta_tau_v", "eta_ratio", "H_star_eta_sh", "nu_gdh2024", "suppression_factor", "tau_star", "tau_end",
    "source_lifetime_factor", "dilution_of_e", "f_min", "f_max",
)


class ImportedTest(unittest.TestCase):
    """Tests for :py:class:`pttools.export.bubble.ImportedBubble` and the imported spectra."""

    bubble: tp.ClassVar[Bubble]
    f_spectrum: tp.ClassVar[Spectrum]
    y_spectrum: tp.ClassVar[Spectrum]
    ssm_spectrum: tp.ClassVar[SSMSpectrum]
    path: tp.ClassVar[str]
    path_full: tp.ClassVar[str]
    path_ssm: tp.ClassVar[str]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        """Compute a bubble and its spectra, and export them with the minimal and full presets."""
        model = ConstCSModel.cached(css2=0.3, csb2=0.29, a_s=1.5, a_b=1, V_s=1, alpha_n_min=0.05)
        cls.bubble = Bubble.cached(model, v_wall=0.5, alpha_n=0.3)
        cls.f_spectrum = Spectrum.cached(cls.bubble, beta_tilde=100, T_star=200, g_star=100, **F_SPECTRUM_KWARGS)
        cls.y_spectrum = Spectrum.cached(cls.bubble, r_star=0.1, **Y_SPECTRUM_KWARGS)
        cls.ssm_spectrum = SSMSpectrum.cached(cls.bubble, r_star=0.1, **Y_SPECTRUM_KWARGS)
        cls.path = str(new_path("imported"))
        with Exporter(cls.path) as exporter:
            exporter.add_many([cls.f_spectrum, cls.y_spectrum])
        cls.path_full = str(new_path("imported_full"))
        with Exporter(cls.path_full, bubble_fields=Preset.FULL, spectrum_fields=Preset.FULL) as exporter:
            exporter.add_many([cls.f_spectrum, cls.y_spectrum])
        cls.path_ssm = str(new_path("imported_ssm"))
        with Exporter(cls.path_ssm) as exporter:
            exporter.add(cls.ssm_spectrum)

    def test_bubble(self) -> None:
        """The quantities of the imported bubble should be those of the computed bubble."""
        with Importer(self.path) as importer:
            bubble = importer.import_bubble(0)
        assert isinstance(bubble, ImportedBubble)
        assert isinstance(bubble, BaseNucleatedBubble)
        assert not isinstance(bubble, ComputedBubble)
        assert bubble.id == self.bubble.id
        assert bubble.solved
        assert bubble.has_profile
        assert bubble.label_unicode == self.bubble.label_unicode
        for name in ("v", "w", "xi", "phase", "e", "T"):
            np.testing.assert_array_equal(getattr(bubble, name), getattr(self.bubble, name), err_msg=name)
        for name in BUBBLE_QUANTITIES:
            assert getattr(bubble, name) == pytest.approx(getattr(self.bubble, name), rel=1e-12), name

    def test_bubbles_shared(self) -> None:
        """The spectra of the same bubble should share the imported bubble."""
        with Importer(self.path) as importer:
            spectra = [*importer.import_spectra(table=Table.SPECTRA_F), *importer.import_spectra(table=Table.SPECTRA_Y)]
            bubbles = importer.import_bubbles()
        assert len(bubbles) == 1
        assert all(spectrum.bubble is bubbles[0] for spectrum in spectra)

    def test_not_imported(self) -> None:
        """The quantities that need data that was not exported should raise NotImportedError."""
        with Importer(self.path) as importer:
            spectrum = importer.import_spectrum(0, table=Table.SPECTRA_F)
        with pytest.raises(NotImportedError, match="spec_den_v"):
            _ = spectrum.spec_den_v
        # The error of a property tells the missing data that it needs.
        with pytest.raises(NotImportedError, match="ubarf2"):
            _ = spectrum.pow_U_par
        with pytest.raises(NotImportedError, match="sm_sh"):
            _ = spectrum.bubble.sm_sh
        assert not hasattr(spectrum, "spec_den_v")
        assert hasattr(spectrum, "r_star")

    def test_full(self) -> None:
        """The data of the full preset should make the spectral densities available."""
        with Importer(self.path_full) as importer:
            spectrum = importer.import_spectrum(0, table=Table.SPECTRA_Y)
        assert spectrum.label_latex == self.y_spectrum.label_latex
        np.testing.assert_allclose(spectrum.pow_gw, self.y_spectrum.pow_gw, rtol=1e-12)
        np.testing.assert_allclose(spectrum.pow_U_par, self.y_spectrum.pow_U_par, rtol=1e-12)
        assert spectrum.bubble.sm_sh == self.bubble.sm_sh

    def test_plot(self) -> None:
        """The plotting functions should take both computed and imported objects."""
        with Importer(self.path) as importer:
            spectrum = importer.import_spectrum(0, table=Table.SPECTRA_F)
        assert isinstance(spectrum, ImportedSpectrum)
        assert isinstance(plot_bubbles([self.bubble, spectrum.bubble]), Figure)
        fig, _ = plot_spectra([self.f_spectrum, spectrum])
        assert isinstance(fig, Figure)

    def test_spectrum_f(self) -> None:
        """The imported spectrum with the given frequencies should have the stored frequencies and spectrum."""
        with Importer(self.path) as importer:
            spectrum = importer.import_spectrum(0, table=Table.SPECTRA_F)
        assert isinstance(spectrum, ImportedSpectrum)
        assert isinstance(spectrum, BaseSSMOmgw0)
        assert spectrum.f_given
        np.testing.assert_array_equal(spectrum.f(), self.f_spectrum.f())
        np.testing.assert_allclose(spectrum.y, self.f_spectrum.y, rtol=1e-12)
        np.testing.assert_allclose(spectrum.omgw0_h2(), self.f_spectrum.omgw0_h2(), rtol=1e-12)
        np.testing.assert_allclose(spectrum.omgw0_h2(gs0=4), self.f_spectrum.omgw0_h2(gs0=4), rtol=1e-12)
        np.testing.assert_allclose(spectrum.pow_gw, self.f_spectrum.pow_gw, rtol=1e-12)
        assert spectrum.snr()[0] == pytest.approx(self.f_spectrum.snr()[0], rel=1e-12)
        assert spectrum.label_unicode == self.f_spectrum.label_unicode
        for name in SPECTRUM_QUANTITIES:
            assert getattr(spectrum, name) == pytest.approx(getattr(self.f_spectrum, name), rel=1e-12), name

    def test_spectrum_y(self) -> None:
        """The imported spectrum with the given y array should have the frequencies of the y array."""
        with Importer(self.path) as importer:
            spectrum = importer.import_spectrum(0, table=Table.SPECTRA_Y)
        assert isinstance(spectrum, ImportedSpectrum)
        assert not spectrum.f_given
        np.testing.assert_array_equal(spectrum.y, self.y_spectrum.y)
        np.testing.assert_allclose(spectrum.f(), self.y_spectrum.f(), rtol=1e-12)
        np.testing.assert_allclose(spectrum.omgw0_h2(), self.y_spectrum.omgw0_h2(), rtol=1e-12)

    def test_ssm_spectrum(self) -> None:
        """The spectra of SSMSpectrum should be imported as ImportedSSMSpectrum."""
        with Importer(self.path_ssm) as importer:
            spectrum = importer.import_spectrum(0)
        assert isinstance(spectrum, ImportedSSMSpectrum)
        assert isinstance(spectrum, BaseSSMSpectrum)
        assert not isinstance(spectrum, BaseSSMOmgw0)
        np.testing.assert_array_equal(spectrum.pow_gw, self.ssm_spectrum.pow_gw)
        assert spectrum.H_star_eta_sh == pytest.approx(self.ssm_spectrum.H_star_eta_sh, rel=1e-12)


if __name__ == "__main__":
    unittest.main()
