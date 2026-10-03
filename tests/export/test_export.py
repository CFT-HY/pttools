"""Tests for exporting and importing models, bubbles and spectra as HDF5 files."""

from pathlib import Path
import pickle
import shutil
import typing as tp
import unittest
from unittest import mock

import h5py
import numpy as np

from pttools.bubble import Bubble
from pttools.export import (
    ChecksumError,
    Exporter,
    ExportFormatError,
    Extractor,
    Importer,
    Preset,
    Table,
    checksum_path,
    verify_checksum,
)
from pttools.models import BagModel, ConstCSModel
from pttools.omgw0 import Spectrum
from pttools.ssm import SSMSpectrum
from tests.export.test_fields import SPECTRUM_MINIMAL_PARAMS
from tests.utils import TEST_RESULT_PATH

EXPORT_PATH: Path = TEST_RESULT_PATH / "export"
EXPORT_PATH.mkdir(parents=True, exist_ok=True)

#: Low-accuracy settings for fast tests
Y_SPECTRUM_KWARGS: dict[str, tp.Any] = {"y": np.logspace(-1, 3, 100), "nT": 1000, "n_z_lookup": 1000}
#: Low-accuracy settings for the spectra that are given the frequencies.
#: The length of f differs from that of y to test that the two sets of spectra can coexist in a file.
F_SPECTRUM_KWARGS: dict[str, tp.Any] = {"f": np.logspace(-5, -1, 80), "nT": 1000, "n_z_lookup": 1000}


def new_path(name: str) -> Path:
    """Path for a new export file, from which any previous files have been removed."""
    path = EXPORT_PATH / f"{name}.h5"
    path.unlink(missing_ok=True)
    checksum_path(path).unlink(missing_ok=True)
    return path


class ExportTest(unittest.TestCase):
    """Tests for exporting and importing spectra."""

    bag: BagModel
    const_cs: ConstCSModel
    bubbles: list[Bubble]
    y_spectra: list[Spectrum]
    f_spectra: list[Spectrum]
    path: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.bag = BagModel(a_s=1.1, a_b=1, V_s=1)
        cls.const_cs = ConstCSModel(css2=1/3 - 0.01, csb2=1/3 - 0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        cls.bubbles = [
            Bubble(cls.bag, v_wall=0.5, alpha_n=0.1),
            Bubble(cls.const_cs, v_wall=0.7, alpha_n=0.1),
        ]
        cls.y_spectra = [
            Spectrum(cls.bubbles[0], r_star=0.1, **Y_SPECTRUM_KWARGS),
            Spectrum(cls.bubbles[0], r_star=0.2, **Y_SPECTRUM_KWARGS),
            Spectrum(cls.bubbles[1], beta_tilde=100, **Y_SPECTRUM_KWARGS),
        ]
        # These share the bubbles with the spectra above, but have the frequencies as their grid.
        cls.f_spectra = [
            Spectrum(cls.bubbles[0], r_star=0.1, **F_SPECTRUM_KWARGS),
            Spectrum(cls.bubbles[1], beta_tilde=100, T_star=1000, **F_SPECTRUM_KWARGS),
        ]
        cls.path = new_path("spectra")
        with Exporter(cls.path) as exporter:
            exporter.add_many(cls.y_spectra)
            exporter.add_many(cls.f_spectra)
            # Copies that have been sent between processes should not be duplicated.
            exporter.add(pickle.loads(pickle.dumps(cls.y_spectra[0])))
            exporter.add(pickle.loads(pickle.dumps(exporter.extractor.extract(cls.y_spectra[2]))))
            exporter.add(pickle.loads(pickle.dumps(exporter.extractor.extract(cls.f_spectra[1]))))

    def test_arrays(self) -> None:
        with Importer(self.path) as importer:
            omgw0_h2 = importer.read(Table.SPECTRA_Y, "omgw0_h2")
            self.assertEqual(omgw0_h2.shape, (3, 100))
            for i, spectrum in enumerate(self.y_spectra):
                np.testing.assert_array_equal(omgw0_h2[i], spectrum.omgw0_h2())
            np.testing.assert_array_equal(importer.read(Table.SPECTRA_Y, "omgw0_h2", [2, 0])[0], omgw0_h2[2])
            np.testing.assert_array_equal(importer.read(Table.SPECTRA_Y, "y"), Y_SPECTRUM_KWARGS["y"])

    def test_checksum(self) -> None:
        self.assertTrue(verify_checksum(self.path))

    # -----
    # Spectra that have been given the frequencies
    # -----

    def test_f_arrays(self) -> None:
        with Importer(self.path) as importer:
            self.assertEqual(importer.n_spectra_f, 2)
            self.assertEqual(importer.n_bubbles, 2)
            np.testing.assert_array_equal(importer.parent_indices(Table.SPECTRA_F), [0, 1])
            np.testing.assert_array_equal(importer.read(Table.SPECTRA_F, "f"), F_SPECTRUM_KWARGS["f"])
            omgw0_h2 = importer.read(Table.SPECTRA_F, "omgw0_h2")
            self.assertEqual(omgw0_h2.shape, (2, 80))
            for i, spectrum in enumerate(self.f_spectra):
                np.testing.assert_array_equal(omgw0_h2[i], spectrum.omgw0_h2())
            np.testing.assert_array_equal(importer.read(Table.SPECTRA_F, "T_star"), [100, 1000])
            fields = importer.fields(Table.SPECTRA_F)
            self.assertIn("f", fields)
            # y depends on the parameters of each spectrum, and is therefore not stored by default.
            self.assertNotIn("y", fields)
            self.assertEqual(importer.field_info(Table.SPECTRA_F, "omgw0_h2")["axis"], "f")
            # The spectra that share y are not affected.
            self.assertEqual(importer.read(Table.SPECTRA_Y, "omgw0_h2").shape, (3, 100))
            self.assertNotIn("f", importer.fields(Table.SPECTRA_Y))

    def test_f_full(self) -> None:
        path = new_path("f_full")
        with Exporter(path, spectrum_fields=Preset.FULL) as exporter:
            exporter.add_many(self.f_spectra)
        with Importer(path) as importer:
            self.assertEqual(importer.n_spectra_y, 0)
            y = importer.read(Table.SPECTRA_F, "y")
            self.assertEqual(y.shape, (2, 80))
            for i, spectrum in enumerate(self.f_spectra):
                np.testing.assert_array_equal(y[i], spectrum.y)
            np.testing.assert_array_equal(importer.read(Table.SPECTRA_F, "f"), F_SPECTRUM_KWARGS["f"])

    def test_f_grid_mismatch(self) -> None:
        path = new_path("f_grid_mismatch")
        spectrum = Spectrum(self.bubbles[0], r_star=0.1, f=np.logspace(-5, -1, 50), compute=False)
        with Exporter(path) as exporter:
            exporter.add(self.f_spectra[0])
            with self.assertRaises(ValueError):
                exporter.add(spectrum)

    def test_f_load(self) -> None:
        with Importer(self.path, verify=True) as importer:
            spectra = importer.load_spectra(table=Table.SPECTRA_F)
            bubble = importer.load_bubble(0)
        self.assertIs(spectra[0].bubble, bubble)
        for loaded, orig in zip(spectra, self.f_spectra, strict=True):
            self.assertIsInstance(loaded, Spectrum)
            self.assertTrue(loaded.f_given)
            np.testing.assert_array_equal(loaded.f(), orig.f())
            self.assertEqual(loaded.r_star, orig.r_star)
            self.assertEqual(loaded.T_star, orig.T_star)
            np.testing.assert_array_equal(loaded.y, orig.y)
            np.testing.assert_allclose(loaded.omgw0_h2(), orig.omgw0_h2(), rtol=1e-12)

    def test_f_record(self) -> None:
        record = Extractor().extract(self.f_spectra[0])
        self.assertEqual(record.table, Table.SPECTRA_F)
        self.assertNotIn("y", record.data)
        self.assertEqual(Extractor().extract(self.y_spectra[0]).table, Table.SPECTRA_Y)

    def test_file_without_f_table(self) -> None:
        """Files created before the table of the spectra with the frequencies was added should still work."""
        path = new_path("without_f_table")
        shutil.copyfile(self.path, path)
        with h5py.File(path, "r+") as file:
            del file["spectra_f"]
        with Importer(path) as importer:
            self.assertEqual(importer.n_spectra_f, 0)
            self.assertEqual(importer.fields(Table.SPECTRA_F), ())
            self.assertEqual(importer.n_spectra_y, 3)
        with Exporter(path, mode="a") as exporter:
            self.assertEqual(exporter.add(self.f_spectra[0]), 0)
        with Importer(path) as importer:
            self.assertEqual(importer.n_spectra_f, 1)

    def test_checksum_corrupted(self) -> None:
        path = new_path("corrupted")
        shutil.copyfile(self.path, path)
        shutil.copyfile(checksum_path(self.path), checksum_path(path))
        with path.open("r+b") as file:
            file.seek(-100, 2)
            byte = file.read(1)
            file.seek(-100, 2)
            file.write(bytes([byte[0] ^ 0xFF]))
        self.assertFalse(verify_checksum(path))
        with self.assertRaises(ChecksumError):
            Importer(path, verify=True)

    def test_descriptions(self) -> None:
        with Importer(self.path) as importer:
            self.assertEqual(
                importer.field_info(Table.SPECTRA_Y, "omgw0_h2")["description"],
                r"Gravitational wave power spectrum today $\Omega_{\text{gw},0} h^2$"
            )
            self.assertEqual(
                importer.field_info(Table.SPECTRA_Y, "r_star")["description"],
                "$r_*$, Hubble-scaled mean bubble spacing"
            )

    def test_counts(self) -> None:
        with Importer(self.path) as importer:
            self.assertEqual(importer.n_models, 2)
            self.assertEqual(importer.n_bubbles, 2)
            self.assertEqual(importer.n_spectra_y, 3)
            np.testing.assert_array_equal(importer.parent_indices(Table.SPECTRA_Y), [0, 0, 1])
            np.testing.assert_array_equal(importer.parent_indices(Table.BUBBLES), [0, 1])
            self.assertEqual(list(importer.read(Table.SPECTRA_Y, "id")), [spectrum.id for spectrum in self.y_spectra])

    def test_load(self) -> None:
        with Importer(self.path, verify=True) as importer:
            spectra = importer.load_spectra()
        self.assertIs(spectra[0].bubble, spectra[1].bubble)
        self.assertIsInstance(spectra[0].bubble.model, BagModel)
        self.assertIsInstance(spectra[2].bubble.model, ConstCSModel)
        self.assertEqual(spectra[2].beta_tilde, 100)
        for loaded, orig in zip(spectra, self.y_spectra, strict=True):
            self.assertIsInstance(loaded, Spectrum)
            self.assertEqual(loaded.r_star, orig.r_star)
            self.assertEqual(loaded.T_star, orig.T_star)
            self.assertEqual(
                loaded.bubble.model.export(fields=Preset.INIT), orig.bubble.model.export(fields=Preset.INIT))
            np.testing.assert_array_equal(loaded.bubble.xi, orig.bubble.xi)
            np.testing.assert_array_equal(loaded.bubble.v, orig.bubble.v)
            np.testing.assert_allclose(loaded.omgw0_h2(), orig.omgw0_h2(), rtol=1e-12)

    def test_model_params(self) -> None:
        with Importer(self.path) as importer:
            params = importer.model_params(1)
            self.assertEqual(importer.read(Table.MODELS, "class", 1), "pttools.models.const_cs.ConstCSModel")
        self.assertEqual(params["name"], "const_cs")
        self.assertEqual(params["css2"], self.const_cs.css2)
        self.assertEqual(params["T_max"], np.inf)

    def test_profiles(self) -> None:
        with Importer(self.path) as importer:
            for i, bubble in enumerate(self.bubbles):
                np.testing.assert_array_equal(importer.read(Table.BUBBLES, "v", i), bubble.v)
            ws = importer.read(Table.BUBBLES, "w")
            xis = importer.read(Table.BUBBLES, "xi", [1, 0])
        for w, xi, bubble in zip(ws, reversed(xis), self.bubbles, strict=True):
            np.testing.assert_array_equal(w, bubble.w)
            np.testing.assert_array_equal(xi, bubble.xi)

    def test_scalars(self) -> None:
        with Importer(self.path) as importer:
            scalars = importer.read_scalars(Table.SPECTRA_Y)
            sol_types = importer.read(Table.BUBBLES, "sol_type")
        for name in SPECTRUM_MINIMAL_PARAMS:
            self.assertIn(name, scalars)
        np.testing.assert_array_equal(scalars["r_star"], [spectrum.r_star for spectrum in self.y_spectra])
        np.testing.assert_array_equal(scalars["v_wall"], [0.5, 0.5, 0.7])
        np.testing.assert_array_equal(scalars["beta_tilde"], [np.nan, np.nan, 100])
        np.testing.assert_array_equal(scalars["cs2"], [spectrum.cs2 for spectrum in self.y_spectra])
        self.assertEqual(list(sol_types), [bubble.sol_type.value for bubble in self.bubbles])
        self.assertEqual(scalars["nuc_type"][0], "exponential")

    # -----
    # Appending
    # -----

    def test_append(self) -> None:
        path = new_path("append")
        shutil.copyfile(self.path, path)
        with Exporter(path, mode="a") as exporter:
            self.assertEqual(exporter.n_spectra_y, 3)
            self.assertFalse(checksum_path(path).exists())
            self.assertEqual(exporter.add(self.y_spectra[1]), 1)
            new_spectrum = Spectrum(self.bubbles[1], r_star=0.3, **Y_SPECTRUM_KWARGS)
            self.assertEqual(exporter.add(new_spectrum), 3)
            self.assertEqual(exporter.n_bubbles, 2)
        self.assertTrue(verify_checksum(path))
        with Importer(path) as importer:
            self.assertEqual(importer.n_spectra_y, 4)
            np.testing.assert_array_equal(importer.parent_indices(Table.SPECTRA_Y), [0, 0, 1, 1])
            np.testing.assert_array_equal(importer.read(Table.SPECTRA_Y, "omgw0_h2", 3), new_spectrum.omgw0_h2())

    def test_append_different_fields(self) -> None:
        path = new_path("append_different_fields")
        shutil.copyfile(self.path, path)
        with Exporter(path, mode="a", spectrum_fields=[Preset.MINIMAL, "f"]) as exporter, \
                self.assertRaises(ExportFormatError):
            exporter.add(Spectrum(self.bubbles[1], r_star=0.3, **Y_SPECTRUM_KWARGS))

    def test_crash_recovery(self) -> None:
        """Rows that were written but not committed should be discarded."""
        path = new_path("crash_recovery")
        shutil.copyfile(self.path, path)
        with h5py.File(path, "r+") as file:
            for name in ("r_star", "omgw0_h2", "id"):
                dset = file["spectra_y"][name]
                dset.resize((5, *dset.shape[1:]))
            for name in ("v", "w", "xi"):
                dset = file["bubbles"][name]
                dset.resize((dset.shape[0] + 10,))
        with Importer(path) as importer:
            self.assertEqual(importer.read(Table.SPECTRA_Y, "r_star").size, 3)
        with Exporter(path, mode="a") as exporter:
            self.assertEqual(exporter.n_spectra_y, 3)
            exporter.add(Spectrum(self.bubbles[1], r_star=0.3, **Y_SPECTRUM_KWARGS))
            exporter.add(Spectrum(Bubble(self.bag, v_wall=0.6, alpha_n=0.05), r_star=0.1, **Y_SPECTRUM_KWARGS))
        with h5py.File(path, "r") as file:
            self.assertEqual(file["spectra_y"]["r_star"].shape, (5,))
            self.assertEqual(file["bubbles"]["v"].shape, (file["bubbles"]["xi_offsets"][-1],))
        with Importer(path) as importer:
            self.assertEqual(importer.n_bubbles, 3)
            np.testing.assert_array_equal(importer.read(Table.BUBBLES, "v", 1), self.bubbles[1].v)

    def test_write_failure(self) -> None:
        """After a failed write, no more data should be accepted, and the committed rows should remain valid."""
        path = new_path("write_failure")
        shutil.copyfile(self.path, path)
        exporter = Exporter(path, mode="a")
        exporter.add(Spectrum(self.bubbles[1], r_star=0.3, **Y_SPECTRUM_KWARGS))
        with mock.patch("pttools.export.exporter._TableWriter.write", side_effect=OSError("Disk full")), \
                self.assertRaises(OSError):
            exporter.flush()
        with self.assertRaises(RuntimeError):
            exporter.add(self.y_spectra[0])
        exporter.close()
        self.assertFalse(checksum_path(path).exists())
        with Importer(path) as importer:
            self.assertEqual(importer.n_spectra_y, 3)
            np.testing.assert_array_equal(importer.read(Table.SPECTRA_Y, "r_star"), [s.r_star for s in self.y_spectra])

    def test_exists(self) -> None:
        with self.assertRaises(FileExistsError):
            Exporter(self.path)

    # -----
    # Other field selections
    # -----

    def test_full(self) -> None:
        path = new_path("full")
        with Exporter(path, model_fields=Preset.FULL, bubble_fields=Preset.FULL, spectrum_fields=Preset.FULL) as exp:
            exp.add_many(self.y_spectra)
        with Importer(path) as importer:
            self.assertEqual(importer.n_spectra_y, 3)
            fields = importer.fields(Table.SPECTRA_Y)
            self.assertIn("snr", fields)
            self.assertEqual(importer.read(Table.SPECTRA_Y, "spec_den_gw_ssm").shape, (3, 100))
            for i, spectrum in enumerate(self.y_spectra):
                np.testing.assert_array_equal(importer.read(Table.SPECTRA_Y, "a2_lookup", i), spectrum.a2_lookup)
                np.testing.assert_array_equal(importer.read(Table.SPECTRA_Y, "z_lookup", i), spectrum.z_lookup)
            np.testing.assert_array_equal(importer.read(Table.BUBBLES, "T", 1), self.bubbles[1].T)
            self.assertIn("w_crit", importer.model_params(0))
            self.assertEqual(importer.read(Table.SPECTRA_Y, "label_unicode", 0), self.y_spectra[0].label_unicode)
            # Descriptions from the docstrings
            self.assertEqual(
                importer.field_info(Table.SPECTRA_Y, "R_star")["description"],
                "Mean bubble separation $R_*$, in units of $T^{-2}$"
            )
            self.assertEqual(
                importer.field_info(Table.BUBBLES, "T")["description"], r"Temperature profile $T(\xi)$")

    def test_grid_mismatch(self) -> None:
        path = new_path("grid_mismatch")
        spectrum = Spectrum(self.bubbles[0], r_star=0.1, y=np.logspace(-1, 3, 50), nT=1000, n_z_lookup=1000)
        with Exporter(path) as exporter:
            exporter.add(self.y_spectra[0])
            with self.assertRaises(ValueError):
                exporter.add(spectrum)

    def test_not_importable(self) -> None:
        path = new_path("not_importable")
        with Exporter(path, importable=False) as exporter:
            exporter.add(self.y_spectra[0])
        with Importer(path) as importer:
            self.assertEqual(set(importer.fields(Table.SPECTRA_Y)), {*SPECTRUM_MINIMAL_PARAMS, "y", "omgw0_h2"})
            with self.assertRaises(ValueError):
                importer.load_spectrum(0)

    def test_record_mismatch(self) -> None:
        path = new_path("record_mismatch")
        record = Extractor(spectrum_fields=Preset.FULL).extract(self.y_spectra[0])
        with Exporter(path) as exporter, self.assertRaises(ValueError):
            exporter.add(record)

    def test_ssm_spectrum(self) -> None:
        path = new_path("ssm_spectrum")
        spectrum = SSMSpectrum(self.bubbles[0], r_star=0.1, **Y_SPECTRUM_KWARGS)
        with Exporter(path) as exporter:
            exporter.add(spectrum)
            exporter.add(self.bubbles[1])
            with self.assertRaises(ExportFormatError):
                exporter.add(self.y_spectra[0])
        with Importer(path) as importer:
            self.assertEqual(importer.n_bubbles, 2)
            np.testing.assert_array_equal(importer.read(Table.SPECTRA_Y, "pow_gw", 0), spectrum.pow_gw)
            loaded = importer.load_spectrum(0)
            bubble = importer.load_bubble(1)
        self.assertIs(type(loaded), SSMSpectrum)
        np.testing.assert_allclose(loaded.pow_gw, spectrum.pow_gw, rtol=1e-12)
        np.testing.assert_array_equal(bubble.v, self.bubbles[1].v)


if __name__ == "__main__":
    unittest.main()
