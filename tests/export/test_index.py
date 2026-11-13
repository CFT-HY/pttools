"""Tests for finding the rows of an exported file by their parameter values."""

import itertools
from pathlib import Path
import random
import typing as tp
import unittest

import numpy as np
import pytest

from pttools.analysis.dataset import compute_spectra
from pttools.analysis.grid import SpectrumParams
from pttools.bubble import Bubble
from pttools.export import DatasetIndex, Exporter, Importer, Selection, Table, detect_axes
from pttools.models import ConstCSModel
from pttools.ssm.nucleation import NucType
from pttools.ssm.settings import SSMSettings
from tests.export.test_export import new_path

ALPHA_NS: tuple[float, ...] = (0.1, 0.2)
BETA_TILDES: tuple[float, ...] = (100., 1000.)
CSB2S: tuple[float, ...] = (0.3, 1/3)
F: np.ndarray = np.logspace(-5, -1, 40)
SETTINGS: SSMSettings = SSMSettings(n_z_lookup=200, nx_P_tilde_gw=50, nT=100)
V_WALLS: tuple[float, ...] = (0.5, 0.8)
#: The parameter values of the spectrum that is left out of the dataset
MISSING: dict[str, tp.Any] = {
    "csb2": CSB2S[1], "alpha_n": ALPHA_NS[0], "v_wall": V_WALLS[1],
    "beta_tilde": BETA_TILDES[0], "nuc_type": NucType.SIMULTANEOUS.value,
}


def create_dataset(name: str) -> Path:
    r"""Export a small dataset, in which the spectra are in a random order and one spectrum is missing.

    The parameters of the dataset are $c_{s,b}^2$, $\alpha_n$, $v_\text{wall}$, $\tilde{\beta}$
    and the nucleation type, and $T_*$ and $g_*$ have a single value.

    :param name: name of the file without the extension
    :return: path of the file
    """
    params = [
        SpectrumParams(nuc_type=nuc_type, beta_tilde=beta_tilde, T_star=100., g_star=100.)
        for nuc_type in NucType for beta_tilde in BETA_TILDES
    ]
    spectra = []
    for csb2, alpha_n, v_wall in itertools.product(CSB2S, ALPHA_NS, V_WALLS):
        model = ConstCSModel.cached(css2=1/3, csb2=csb2, a_s=1.5, a_b=1, V_s=1, alpha_n_min=0.05)
        bubble = Bubble.cached(model, v_wall=v_wall, alpha_n=alpha_n)
        spectra += compute_spectra(bubble, params, F, SETTINGS)[0]
    spectra = [spectrum for spectrum in spectra if {**values_of(spectrum), **MISSING} != values_of(spectrum)]
    random.Random(0).shuffle(spectra)
    path = new_path(name)
    with Exporter(path) as exporter:
        exporter.add_many(spectra)
    return path


def values_of(spectrum: tp.Any) -> dict[str, tp.Any]:
    """The values of the parameters of the dataset for a spectrum."""
    return {
        "csb2": spectrum.bubble.model.csb2,
        "alpha_n": spectrum.bubble.alpha_n,
        "v_wall": spectrum.bubble.v_wall,
        "beta_tilde": spectrum.beta_tilde,
        "T_star": spectrum.T_star,
        "g_star": spectrum.g_star,
        "nuc_type": spectrum.nuc_type.value,
    }


class DatasetIndexTest(unittest.TestCase):
    """Tests for :py:class:`pttools.export.index.DatasetIndex`."""

    importer: tp.ClassVar[Importer]
    index: tp.ClassVar[DatasetIndex]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        """Export a dataset and create its index."""
        cls.importer = Importer(create_dataset("index"))
        cls.addClassCleanup(cls.importer.close)
        cls.index = DatasetIndex(cls.importer)

    def test_axes(self) -> None:
        """The axes should have the distinct values of the parameters, and the model axes should be detected."""
        index = self.index
        assert index.table == Table.SPECTRA_F
        assert [axis.name for axis in index.model_axes] == ["csb2"]
        assert [axis.name for axis in index.axes] == [
            "csb2", "alpha_n", "v_wall", "beta_tilde", "T_star", "g_star", "nuc_type"]
        axes = {axis.name: axis for axis in index.axes}
        assert axes["csb2"].values == CSB2S
        assert axes["alpha_n"].values == ALPHA_NS
        assert axes["v_wall"].values == V_WALLS
        assert axes["beta_tilde"].values == BETA_TILDES
        assert axes["T_star"].values == (100.,)
        assert axes["nuc_type"].values == tuple(sorted(nuc_type.value for nuc_type in NucType))
        assert axes["alpha_n"].numeric
        assert not axes["nuc_type"].numeric
        assert axes["csb2"].table == Table.MODELS
        assert "c_{s,b}^2" in axes["csb2"].description
        assert "alpha_n" in axes["alpha_n"].description

    def test_default_values(self) -> None:
        """The default values should be those of an existing spectrum."""
        selection = self.index.lookup(self.index.default_values())
        assert selection.spectrum is not None

    def test_lookup(self) -> None:
        """Each combination of the values should give the rows that have those values, except for the missing one."""
        index = self.index
        importer = self.importer
        names = [axis.name for axis in index.axes]
        n_found = 0
        for combination in itertools.product(*(axis.values for axis in index.axes)):
            values = dict(zip(names, combination, strict=True))
            selection = index.lookup(values)
            assert selection.model is not None
            assert selection.bubble is not None
            assert importer.model_params(selection.model)["csb2"] == values["csb2"]
            assert importer.read(Table.BUBBLES, "model", selection.bubble) == selection.model
            assert importer.read(Table.BUBBLES, "alpha_n", selection.bubble) == values["alpha_n"]
            assert importer.read(Table.BUBBLES, "v_wall", selection.bubble) == values["v_wall"]
            if {**values, **MISSING} == values:
                assert selection.spectrum is None
                continue
            assert selection.spectrum is not None
            n_found += 1
            assert importer.read(index.table, "bubble", selection.spectrum) == selection.bubble
            assert importer.read(index.table, "beta_tilde", selection.spectrum) == values["beta_tilde"]
            assert importer.read(index.table, "nuc_type", selection.spectrum) == values["nuc_type"]
            assert index.values_of(selection.bubble, selection.spectrum) == values
        assert n_found == importer.n_spectra_f

    def test_lookup_unknown(self) -> None:
        """The values that are not in the dataset should give no rows."""
        values = self.index.default_values()
        assert self.index.lookup({**values, "alpha_n": 0.15}) == Selection(model=self.index.lookup(values).model)
        assert self.index.lookup({**values, "csb2": 0.2}) == Selection()

    def test_model_params(self) -> None:
        """The parameters of the models can be given, and have to exist."""
        index = DatasetIndex(self.importer, model_params=["csb2", "css2"])
        assert [axis.name for axis in index.model_axes] == ["csb2", "css2"]
        assert index.lookup(index.default_values()).spectrum is not None
        with pytest.raises(ValueError, match="nonexistent"):
            DatasetIndex(self.importer, model_params=["nonexistent"])

    def test_spectra_of(self) -> None:
        """The spectra of each bubble should be found regardless of their order in the file."""
        parents = self.importer.parent_indices(Table.SPECTRA_F)
        assert np.any(np.diff(parents) < 0)
        for bubble in range(self.importer.n_bubbles):
            np.testing.assert_array_equal(self.index.spectra_of(bubble), np.flatnonzero(parents == bubble))


class DetectAxesTest(unittest.TestCase):
    """Tests for :py:func:`pttools.export.index.detect_axes`."""

    @staticmethod
    def test_dependent() -> None:
        """The parameters of a grid should be preferred to the parameters that depend on them."""
        rows = [
            {"name": "model", "a": a + 0.1 * b, "b": b, "c": c, "d": 1.}
            for b in (1., 2., 3.) for c in (4., 5., 6.) for a in (b * c,)
        ]
        assert detect_axes(rows) == ("b", "c")

    @staticmethod
    def test_single() -> None:
        """A single parameter that identifies the rows should be selected, and the later one of equally good ones."""
        assert detect_axes([{"a": 1., "b": 1.}, {"a": 1., "b": 2.}, {"a": 2., "b": 3.}]) == ("b",)
        assert detect_axes([{"a": 1., "b": 1., "c": "x"}, {"a": 2., "b": 2., "c": "y"}]) == ("b",)

    @staticmethod
    def test_trivial() -> None:
        """A single row needs no parameters, and identical rows have none."""
        assert detect_axes([]) == ()
        assert detect_axes([{"a": 1.}]) == ()
        assert detect_axes([{"a": 1.}, {"a": 1.}]) == ()


if __name__ == "__main__":
    unittest.main()
