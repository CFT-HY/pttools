"""Base test case for thermodynamic models."""

import abc

import numpy as np

from pttools.models import ThermoModel
import pttools.type_hints as th
from tests.utils.const import TEST_DATA_PATH
from tests.utils.json import JsonTestCase


class ThermoModelBaseCase[T: ThermoModel](JsonTestCase, abc.ABC):
    """Base test case for thermodynamic models."""

    thermo: T
    temp_arr: th.FloatArr1D
    phase_arr: th.FloatArr1D

    EXPECT_MISSING_DATA = True
    SAVE_NEW_DATA = True

    @classmethod
    def setUpClass(cls, thermo: T) -> None:
        """Set the thermodynamic model to be tested and load its reference data.

        :param thermo: thermodynamic model to be tested
        """
        cls.thermo = thermo
        cls.REF_DATA_PATH = TEST_DATA_PATH / "models" / "thermo" / f"{thermo.name}.json"
        super().setUpClass()

    def test_class_is_valid(self) -> None:
        """Test that the test arrays have the same size."""
        sizes = np.array([self.temp_arr.size, self.phase_arr.size])
        if np.any(sizes != self.temp_arr.size):
            raise ValueError(f"Test arrays must have the same size. Got: {sizes}")

    def test_dge_dT(self) -> None:
        """Test the temperature derivative of the degrees of freedom for energy density against the reference data."""
        data = self.thermo.dge_dT(self.temp_arr, self.phase_arr)
        self.assert_json(data, "dge_dT")

    def test_dgs_dT(self) -> None:
        """Test the temperature derivative of the degrees of freedom for entropy density against the reference data."""
        data = self.thermo.dgs_dT(self.temp_arr, self.phase_arr)
        self.assert_json(data, "dgs_dT")

    def test_ge(self) -> None:
        """Test the degrees of freedom for energy density against the reference data."""
        data = self.thermo.ge(self.temp_arr, self.phase_arr)
        self.assert_json(data, "ge")

    def test_gs(self) -> None:
        """Test the degrees of freedom for entropy density against the reference data."""
        data = self.thermo.gs(self.temp_arr, self.phase_arr)
        self.assert_json(data, "gs")

    def test_dp_dt(self) -> None:
        """Test the temperature derivative of the pressure against the reference data."""
        data = self.thermo.dp_dt(self.temp_arr, self.phase_arr)
        self.assert_json(data, "dp_dt")

    def test_de_dt(self) -> None:
        """Test the temperature derivative of the energy density against the reference data."""
        data = self.thermo.de_dt(self.temp_arr, self.phase_arr)
        self.assert_json(data, "de_dt")
