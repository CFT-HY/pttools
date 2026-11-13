"""Tests for the interactive viewer of datasets."""

import typing as tp
import unittest

import numpy as np
import pytest

from pttools.export import ImportedSpectrum, Importer, Table
from pttools.plot.dataset import bubble_fig, empty_fig, spectrum_fig
from tests.export.test_index import MISSING, create_dataset

if tp.TYPE_CHECKING:
    from pttools.plot.dataset.app import DatasetViewer


class FiguresTest(unittest.TestCase):
    """Tests for :py:mod:`pttools.plot.dataset.figures`."""

    importer: tp.ClassVar[Importer]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        """Export a dataset."""
        cls.importer = Importer(create_dataset("viewer_figures"))
        cls.addClassCleanup(cls.importer.close)

    def test_bubble_fig(self) -> None:
        """The figure of a bubble should have its fluid profiles."""
        bubble = self.importer.import_bubble(0)
        fig: tp.Any = bubble_fig(bubble)
        np.testing.assert_array_equal(fig.data[0].x, bubble.xi)
        np.testing.assert_array_equal(fig.data[0].y, bubble.v)
        np.testing.assert_array_equal(fig.data[1].y, bubble.w)
        assert fig.layout.title.text == bubble.label_unicode

    @staticmethod
    def test_empty_fig() -> None:
        """The empty figure should show its message."""
        fig: tp.Any = empty_fig("No data")
        assert fig.layout.annotations[0].text == "No data"

    def test_spectrum_fig(self) -> None:
        """The figure of a spectrum should have the spectrum and the noise on logarithmic axes."""
        spectrum = self.importer.import_spectrum(0, table=Table.SPECTRA_F)
        assert isinstance(spectrum, ImportedSpectrum)
        fig: tp.Any = spectrum_fig(spectrum)
        np.testing.assert_array_equal(fig.data[0].x, spectrum.f())
        np.testing.assert_array_equal(fig.data[0].y, spectrum.omgw0_h2())
        assert len(fig.data) == 2
        assert fig.layout.xaxis.type == fig.layout.yaxis.type == "log"
        assert "SNR" in fig.layout.title.text


class ViewerTest(unittest.TestCase):
    """Tests for :py:mod:`pttools.plot.dataset.app`."""

    importer: tp.ClassVar[Importer]
    viewer: tp.ClassVar[DatasetViewer]

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        """Export a dataset and create its viewer."""
        pytest.importorskip("dash")
        from pttools.plot.dataset.app import DatasetViewer  # noqa: PLC0415
        cls.importer = Importer(create_dataset("viewer_app"))
        cls.addClassCleanup(cls.importer.close)
        cls.viewer = DatasetViewer(cls.importer)

    def test_app(self) -> None:
        """The application should serve its layout and update it with the callback."""
        from pttools.plot.dataset.app import create_app  # noqa: PLC0415
        client = create_app(self.importer).server.test_client()
        index = client.get("/")
        assert index.status_code == 200
        # The stylesheet of the tables is served from the assets of the package.
        assert "viewer.css" in index.text
        layout = client.get("/_dash-layout")
        assert layout.status_code == 200
        assert "control-alpha_n" in layout.text
        assert "control-nuc_type" in layout.text
        # T_star has a single value, and therefore no control.
        assert "control-T_star" not in layout.text

        dependencies = client.get("/_dash-dependencies").json
        assert len(dependencies) == 1
        callback = dependencies[0]
        states = {"csb2": 1, "alpha_n": 1, "v_wall": 0, "beta_tilde": 1, "nuc_type": "exponential"}
        response = client.post("/_dash-update-component", json={
            "output": callback["output"],
            "outputs": [
                {"id": output.rsplit(".", 1)[0], "property": output.rsplit(".", 1)[1]}
                for output in callback["output"].strip(".").split("...")
            ],
            "inputs": [
                {**inp, "value": states[inp["id"].removeprefix("control-")]} for inp in callback["inputs"]],
            "changedPropIds": [],
        })
        assert response.status_code == 200, response.text
        result = response.json["response"]
        assert len(result["bubble-graph"]["figure"]["data"]) == 2
        assert len(result["spectrum-graph"]["figure"]["data"]) == 2
        assert "| `beta_tilde` | 1000 |" in result[f"scalars-{Table.SPECTRA_F}"]["children"]
        assert result["label-alpha_n"]["children"].endswith("**0.2**")

    def test_controls(self) -> None:
        """The parameters with several values should have controls, whose states are converted to the values."""
        viewer = self.viewer
        assert [axis.name for axis in viewer.controls] == ["csb2", "alpha_n", "v_wall", "beta_tilde", "nuc_type"]
        assert viewer.fixed == {"T_star": 100., "g_star": 100.}
        values = viewer.control_values([1, 0, 1, 0, "simultaneous"])
        assert values == {**MISSING, "T_star": 100., "g_star": 100.}

    def test_update(self) -> None:
        """The contents should have the figures and all the stored scalars of the selected rows."""
        viewer = self.viewer
        values = viewer.index.default_values()
        state = viewer.update(values)
        selection = state.selection
        assert selection == viewer.index.lookup(values)
        assert selection.spectrum is not None
        spectrum = self.importer.import_spectrum(selection.spectrum, table=Table.SPECTRA_F)
        assert isinstance(spectrum, ImportedSpectrum)
        fig_spectrum: tp.Any = state.spectrum_fig
        fig_bubble: tp.Any = state.bubble_fig
        np.testing.assert_array_equal(fig_spectrum.data[0].y, spectrum.omgw0_h2())
        np.testing.assert_array_equal(fig_bubble.data[0].y, spectrum.bubble.v)
        for table, row in ((Table.BUBBLES, selection.bubble), (Table.SPECTRA_F, selection.spectrum)):
            scalars = {scalar.name: scalar for scalar in state.scalars[table]}
            stored = self.importer.read_scalars(table, rows=row)
            assert list(scalars) == list(stored)
            for name, value in stored.items():
                np.testing.assert_equal(scalars[name].value, value)
            assert scalars["alpha_n"].description
        model = {scalar.name: scalar.value for scalar in state.scalars[Table.MODELS]}
        assert model["csb2"] == values["csb2"]
        assert model["class"] == "pttools.models.const_cs.ConstCSModel"

    def test_update_missing(self) -> None:
        """The parameter values without a spectrum should give the bubble and a message."""
        state = self.viewer.update({**MISSING, "T_star": 100., "g_star": 100.})
        assert state.selection.bubble is not None
        assert state.selection.spectrum is None
        fig_bubble: tp.Any = state.bubble_fig
        fig_spectrum: tp.Any = state.spectrum_fig
        assert len(fig_bubble.data) == 2
        assert not fig_spectrum.data
        assert Table.SPECTRA_F not in state.scalars
        assert any("No spectrum" in message for message in state.messages)
        state = self.viewer.update({**MISSING, "T_star": 100., "g_star": 100., "csb2": 0.2})
        assert state.selection.model is None
        assert not state.scalars


if __name__ == "__main__":
    unittest.main()
