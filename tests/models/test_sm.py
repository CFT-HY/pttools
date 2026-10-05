"""Tests for the Standard Model."""

import typing as tp
import unittest

import numpy as np

from pttools import models
from tests.models.base_thermo import ThermoModelBaseCase


class TestStandardModel(ThermoModelBaseCase[models.StandardModel], unittest.TestCase):
    """Tests for the Standard Model."""

    temp_arr = np.logspace(models.StandardModel.GEFF_DATA[0, 0], models.StandardModel.GEFF_DATA[0, -1], 10)
    phase_arr = np.linspace(0, 1, temp_arr.size)

    @classmethod
    def setUpClass(cls, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Create the Standard Model and load its reference data."""
        thermo = models.StandardModel()
        super().setUpClass(thermo)

    def test_geff_arrays(self) -> None:
        """Test that the effective degrees of freedom data arrays have the correct dimensions.

        It's easy to accidentally make these into column vectors,
        which will mess up the dimensionality of the spliners.
        """
        assert self.thermo.GEFF_DATA.ndim == 2
        assert self.thermo.GEFF_DATA_GE.ndim == 1
        assert self.thermo.GEFF_DATA_GS.ndim == 1
        assert self.thermo.GEFF_DATA_GE_GS_RATIO.ndim == 1
        assert self.thermo.GEFF_DATA_LOG_TEMP.ndim == 1
        assert self.thermo.GEFF_DATA_TEMP.ndim == 1

    # def test_phase_invariance(self):
    #     raise NotImplementedError
