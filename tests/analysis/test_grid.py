"""Tests for the parameter grids of datasets."""

import dataclasses
import unittest

import numpy as np
import pytest

from pttools.analysis.grid import Grid, SpectrumParams
from pttools.export import Table
from pttools.ssm.nucleation import NucType


def create_grid() -> Grid:
    """Create a small grid with 1-3 values for each parameter."""
    return Grid(
        model_axes={"css2": np.array([0.25, 0.3]), "csb2": np.array([0.26, 0.31])},
        alpha_ns=np.array([0.1, 0.2, 0.3]),
        v_walls=np.array([0.5, 0.7]),
        beta_tildes=np.array([10.]),
        T_stars=np.array([100., 200.]),
        g_stars=np.array([100.]),
    )


class GridTest(unittest.TestCase):
    """Tests for :py:class:`pttools.analysis.grid.Grid`."""

    @staticmethod
    def test_counts() -> None:
        """The numbers of the models, bubbles and spectra are the products of the numbers of the values."""
        grid = create_grid()
        assert grid.n_models == 4
        assert grid.n_bubbles == 4 * 6
        assert grid.n_spectra_per_bubble == len(NucType) * 2
        assert grid.n_spectra == 4 * 6 * len(NucType) * 2
        assert grid.n_rows() == {Table.MODELS: 4, Table.BUBBLES: 24, Table.SPECTRA_F: 24 * len(NucType) * 2}

    @staticmethod
    def test_empty() -> None:
        """A grid with a parameter without values is invalid."""
        with pytest.raises(ValueError, match="alpha_ns"):
            dataclasses.replace(create_grid(), alpha_ns=np.array([]))
        with pytest.raises(ValueError, match="csb2"):
            dataclasses.replace(create_grid(), model_axes={"css2": np.array([0.3]), "csb2": np.array([])})

    @staticmethod
    def test_params() -> None:
        """The parameters are the Cartesian products of the values, with the last parameter varying fastest."""
        grid = create_grid()
        assert list(grid.model_params()) == [
            {"css2": 0.25, "csb2": 0.26}, {"css2": 0.25, "csb2": 0.31},
            {"css2": 0.3, "csb2": 0.26}, {"css2": 0.3, "csb2": 0.31},
        ]
        assert list(grid.bubble_params())[:3] == [(0.1, 0.5), (0.1, 0.7), (0.2, 0.5)]
        spectra = grid.spectrum_params()
        assert len(spectra) == grid.n_spectra_per_bubble
        assert spectra[0] == SpectrumParams(nuc_type=next(iter(NucType)), beta_tilde=10, T_star=100, g_star=100)


if __name__ == "__main__":
    unittest.main()
