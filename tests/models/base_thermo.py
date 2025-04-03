import abc
import os.path

import numpy as np


from pttools.models import ThermoModel
from tests.utils.const import TEST_DATA_PATH
from tests.utils.json import JsonTestCase


class ThermoModelBaseCase(JsonTestCase, abc.ABC):
    thermo: ThermoModel
    temp_arr: np.ndarray
    phase_arr: np.ndarray

    EXPECT_MISSING_DATA = True
    SAVE_NEW_DATA = True

    @classmethod
    def setUpClass(cls, thermo: ThermoModel):
        cls.thermo = thermo
        cls.REF_DATA_PATH = os.path.join(TEST_DATA_PATH, "models", "thermo", f"{thermo.name}.json")
        super().setUpClass()

    def test_class_is_valid(self):
        sizes = np.array([self.temp_arr.size, self.phase_arr.size])
        if np.any(sizes != self.temp_arr.size):
            raise ValueError(f"Test arrays must have the same size. Got: {sizes}")

    def test_dge_dT(self):
        data = self.thermo.dge_dT(self.temp_arr, self.phase_arr)
        self.assert_json(data, "dge_dT")

    def test_dgs_dT(self):
        data = self.thermo.dgs_dT(self.temp_arr, self.phase_arr)
        self.assert_json(data, "dgs_dT")

    def test_ge(self):
        data = self.thermo.ge(self.temp_arr, self.phase_arr)
        self.assert_json(data, "ge")

    def test_gs(self):
        data = self.thermo.gs(self.temp_arr, self.phase_arr)
        self.assert_json(data, "gs")

    def test_dp_dt(self):
        data = self.thermo.dp_dt(self.temp_arr, self.phase_arr)
        self.assert_json(data, "dp_dt")

    def test_de_dt(self):
        data = self.thermo.de_dt(self.temp_arr, self.phase_arr)
        self.assert_json(data, "de_dt")
