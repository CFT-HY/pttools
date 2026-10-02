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
SPECTRUM_KWARGS: dict[str, tp.Any] = {"y": np.logspace(-1, 3, 100), "nT": 1000, "n_z_lookup": 1000}


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
    spectra: list[Spectrum]
    path: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls.bag = BagModel(a_s=1.1, a_b=1, V_s=1)
        cls.const_cs = ConstCSModel(css2=1/3 - 0.01, csb2=1/3 - 0.011, a_s=1.1, a_b=1, V_s=1, V_b=0)
        cls.bubbles = [
            Bubble(cls.bag, v_wall=0.5, alpha_n=0.1),
            Bubble(cls.const_cs, v_wall=0.7, alpha_n=0.1),
        ]
        cls.spectra = [
            Spectrum(cls.bubbles[0], r_star=0.1, **SPECTRUM_KWARGS),
            Spectrum(cls.bubbles[0], r_star=0.2, **SPECTRUM_KWARGS),
            Spectrum(cls.bubbles[1], beta_tilde=100, **SPECTRUM_KWARGS),
        ]
        cls.path = new_path("spectra")
        with Exporter(cls.path) as exporter:
            exporter.add_many(cls.spectra)
            # Copies that have been sent between processes should not be duplicated.
            exporter.add(pickle.loads(pickle.dumps(cls.spectra[0])))
            exporter.add(pickle.loads(pickle.dumps(exporter.extractor.extract(cls.spectra[2]))))

    def test_arrays(self) -> None:
        with Importer(self.path) as importer:
            omgw0_h2 = importer.read(Table.SPECTRA, "omgw0_h2")
            self.assertEqual(omgw0_h2.shape, (3, 100))
            for i, spectrum in enumerate(self.spectra):
                np.testing.assert_array_equal(omgw0_h2[i], spectrum.omgw0_h2())
            np.testing.assert_array_equal(importer.read(Table.SPECTRA, "omgw0_h2", [2, 0])[0], omgw0_h2[2])
            np.testing.assert_array_equal(importer.read(Table.SPECTRA, "y"), SPECTRUM_KWARGS["y"])

    def test_checksum(self) -> None:
        self.assertTrue(verify_checksum(self.path))

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

    def test_counts(self) -> None:
        with Importer(self.path) as importer:
            self.assertEqual(importer.n_models, 2)
            self.assertEqual(importer.n_bubbles, 2)
            self.assertEqual(importer.n_spectra, 3)
            np.testing.assert_array_equal(importer.parent_indices(Table.SPECTRA), [0, 0, 1])
            np.testing.assert_array_equal(importer.parent_indices(Table.BUBBLES), [0, 1])
            self.assertEqual(list(importer.read(Table.SPECTRA, "id")), [spectrum.id for spectrum in self.spectra])

    def test_load(self) -> None:
        with Importer(self.path, verify=True) as importer:
            spectra = importer.load_spectra()
        self.assertIs(spectra[0].bubble, spectra[1].bubble)
        self.assertIsInstance(spectra[0].bubble.model, BagModel)
        self.assertIsInstance(spectra[2].bubble.model, ConstCSModel)
        self.assertEqual(spectra[2].beta_tilde, 100)
        for loaded, orig in zip(spectra, self.spectra, strict=True):
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
            scalars = importer.read_scalars(Table.SPECTRA)
            sol_types = importer.read(Table.BUBBLES, "sol_type")
        for name in SPECTRUM_MINIMAL_PARAMS:
            self.assertIn(name, scalars)
        np.testing.assert_array_equal(scalars["r_star"], [spectrum.r_star for spectrum in self.spectra])
        np.testing.assert_array_equal(scalars["v_wall"], [0.5, 0.5, 0.7])
        np.testing.assert_array_equal(scalars["beta_tilde"], [np.nan, np.nan, 100])
        np.testing.assert_array_equal(scalars["cs2"], [spectrum.cs2 for spectrum in self.spectra])
        self.assertEqual(list(sol_types), [bubble.sol_type.value for bubble in self.bubbles])
        self.assertEqual(scalars["nuc_type"][0], "exponential")

    # -----
    # Appending
    # -----

    def test_append(self) -> None:
        path = new_path("append")
        shutil.copyfile(self.path, path)
        with Exporter(path, mode="a") as exporter:
            self.assertEqual(exporter.n_spectra, 3)
            self.assertFalse(checksum_path(path).exists())
            self.assertEqual(exporter.add(self.spectra[1]), 1)
            new_spectrum = Spectrum(self.bubbles[1], r_star=0.3, **SPECTRUM_KWARGS)
            self.assertEqual(exporter.add(new_spectrum), 3)
            self.assertEqual(exporter.n_bubbles, 2)
        self.assertTrue(verify_checksum(path))
        with Importer(path) as importer:
            self.assertEqual(importer.n_spectra, 4)
            np.testing.assert_array_equal(importer.parent_indices(Table.SPECTRA), [0, 0, 1, 1])
            np.testing.assert_array_equal(importer.read(Table.SPECTRA, "omgw0_h2", 3), new_spectrum.omgw0_h2())

    def test_append_different_fields(self) -> None:
        path = new_path("append_different_fields")
        shutil.copyfile(self.path, path)
        with Exporter(path, mode="a", spectrum_fields=[Preset.MINIMAL, "f"]) as exporter, \
                self.assertRaises(ExportFormatError):
            exporter.add(Spectrum(self.bubbles[1], r_star=0.3, **SPECTRUM_KWARGS))

    def test_crash_recovery(self) -> None:
        """Rows that were written but not committed should be discarded."""
        path = new_path("crash_recovery")
        shutil.copyfile(self.path, path)
        with h5py.File(path, "r+") as file:
            for name in ("r_star", "omgw0_h2", "id"):
                dset = file["spectra"][name]
                dset.resize((5, *dset.shape[1:]))
            for name in ("v", "w", "xi"):
                dset = file["bubbles"][name]
                dset.resize((dset.shape[0] + 10,))
        with Importer(path) as importer:
            self.assertEqual(importer.read(Table.SPECTRA, "r_star").size, 3)
        with Exporter(path, mode="a") as exporter:
            self.assertEqual(exporter.n_spectra, 3)
            exporter.add(Spectrum(self.bubbles[1], r_star=0.3, **SPECTRUM_KWARGS))
            exporter.add(Spectrum(Bubble(self.bag, v_wall=0.6, alpha_n=0.05), r_star=0.1, **SPECTRUM_KWARGS))
        with h5py.File(path, "r") as file:
            self.assertEqual(file["spectra"]["r_star"].shape, (5,))
            self.assertEqual(file["bubbles"]["v"].shape, (file["bubbles"]["xi_offsets"][-1],))
        with Importer(path) as importer:
            self.assertEqual(importer.n_bubbles, 3)
            np.testing.assert_array_equal(importer.read(Table.BUBBLES, "v", 1), self.bubbles[1].v)

    def test_write_failure(self) -> None:
        """After a failed write, no more data should be accepted, and the committed rows should remain valid."""
        path = new_path("write_failure")
        shutil.copyfile(self.path, path)
        exporter = Exporter(path, mode="a")
        exporter.add(Spectrum(self.bubbles[1], r_star=0.3, **SPECTRUM_KWARGS))
        with mock.patch("pttools.export.exporter._TableWriter.write", side_effect=OSError("Disk full")), \
                self.assertRaises(OSError):
            exporter.flush()
        with self.assertRaises(RuntimeError):
            exporter.add(self.spectra[0])
        exporter.close()
        self.assertFalse(checksum_path(path).exists())
        with Importer(path) as importer:
            self.assertEqual(importer.n_spectra, 3)
            np.testing.assert_array_equal(importer.read(Table.SPECTRA, "r_star"), [s.r_star for s in self.spectra])

    def test_exists(self) -> None:
        with self.assertRaises(FileExistsError):
            Exporter(self.path)

    # -----
    # Other field selections
    # -----

    def test_full(self) -> None:
        path = new_path("full")
        with Exporter(path, model_fields=Preset.FULL, bubble_fields=Preset.FULL, spectrum_fields=Preset.FULL) as exp:
            exp.add_many(self.spectra)
        with Importer(path) as importer:
            self.assertEqual(importer.n_spectra, 3)
            fields = importer.fields(Table.SPECTRA)
            self.assertIn("snr", fields)
            self.assertEqual(importer.read(Table.SPECTRA, "spec_den_gw_ssm").shape, (3, 100))
            for i, spectrum in enumerate(self.spectra):
                np.testing.assert_array_equal(importer.read(Table.SPECTRA, "a2_lookup", i), spectrum.a2_lookup)
                np.testing.assert_array_equal(importer.read(Table.SPECTRA, "z_lookup", i), spectrum.z_lookup)
            np.testing.assert_array_equal(importer.read(Table.BUBBLES, "T", 1), self.bubbles[1].T)
            self.assertIn("w_crit", importer.model_params(0))
            self.assertEqual(importer.read(Table.SPECTRA, "label_unicode", 0), self.spectra[0].label_unicode)

    def test_grid_mismatch(self) -> None:
        path = new_path("grid_mismatch")
        spectrum = Spectrum(self.bubbles[0], r_star=0.1, y=np.logspace(-1, 3, 50), nT=1000, n_z_lookup=1000)
        with Exporter(path) as exporter:
            exporter.add(self.spectra[0])
            with self.assertRaises(ValueError):
                exporter.add(spectrum)

    def test_not_importable(self) -> None:
        path = new_path("not_importable")
        with Exporter(path, importable=False) as exporter:
            exporter.add(self.spectra[0])
        with Importer(path) as importer:
            self.assertEqual(set(importer.fields(Table.SPECTRA)), {*SPECTRUM_MINIMAL_PARAMS, "y", "omgw0_h2"})
            with self.assertRaises(ValueError):
                importer.load_spectrum(0)

    def test_record_mismatch(self) -> None:
        path = new_path("record_mismatch")
        record = Extractor(spectrum_fields=Preset.FULL).extract(self.spectra[0])
        with Exporter(path) as exporter, self.assertRaises(ValueError):
            exporter.add(record)

    def test_ssm_spectrum(self) -> None:
        path = new_path("ssm_spectrum")
        spectrum = SSMSpectrum(self.bubbles[0], r_star=0.1, **SPECTRUM_KWARGS)
        with Exporter(path) as exporter:
            exporter.add(spectrum)
            exporter.add(self.bubbles[1])
            with self.assertRaises(ExportFormatError):
                exporter.add(self.spectra[0])
        with Importer(path) as importer:
            self.assertEqual(importer.n_bubbles, 2)
            np.testing.assert_array_equal(importer.read(Table.SPECTRA, "pow_gw", 0), spectrum.pow_gw)
            loaded = importer.load_spectrum(0)
            bubble = importer.load_bubble(1)
        self.assertIs(type(loaded), SSMSpectrum)
        np.testing.assert_allclose(loaded.pow_gw, spectrum.pow_gw, rtol=1e-12)
        np.testing.assert_array_equal(bubble.v, self.bubbles[1].v)


if __name__ == "__main__":
    unittest.main()
