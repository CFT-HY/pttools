import abc
import os.path

import numpy as np


from pttools.models import Model
from tests.utils.const import TEST_DATA_PATH
from tests.utils.json import JsonTestCase


class ModelBaseCase(JsonTestCase, abc.ABC):
    model: Model

    EXPECT_MISSING_DATA = True
    SAVE_NEW_DATA = True

    REF_DATA_PATH: str
    TEST_ARR_SIZE: int = 10

    alpha_n = np.linspace(0.15, 0.5, 10)
    temp_arr = np.linspace(1, 100, TEST_ARR_SIZE)
    w_arr1 = temp_arr**4
    w_arr2 = temp_arr**3.9
    phase_arr = np.array([(1 + (-1)**i)/2 for i in range(TEST_ARR_SIZE)])

    @classmethod
    def setUpClass(cls, model: Model):
        cls.model = model
        cls.REF_DATA_PATH = os.path.join(TEST_DATA_PATH, "models", f"{model.name}.json")
        super().setUpClass()

    def test_class_is_valid(self):
        sizes = np.array([self.w_arr1.size, self.w_arr2.size, self.phase_arr.size, self.temp_arr.size])
        if np.any(sizes != self.w_arr1.size):
            raise ValueError(f"Test arrays must have the same shape. Got: {sizes}")

    def test_alpha_n(self):
        data = self.model.alpha_n(self.w_arr1, error_on_invalid=False, log_invalid=False)
        self.assert_json(data, "alpha_n")

    def test_alpha_plus(self):
        data = self.model.alpha_plus(self.w_arr1, self.w_arr2, error_on_invalid=False, log_invalid=False)
        self.assert_json(data, "alpha_plus")

    def test_critical_temp(self):
        data = self.model.critical_temp(guess=1)
        self.assert_json(data, "critical_temp")

    def test_cs2(self):
        data = self.model.cs2(self.w_arr1, self.phase_arr)
        self.assert_json(data, "cs2", rtol=3.4e-4)

    def test_e(self):
        data = self.model.e(self.w_arr1, self.phase_arr)
        self.assert_json(data, "e")

    def test_p(self):
        data = self.model.p(self.w_arr1, self.phase_arr)
        self.assert_json(data, "p", rtol=2.3e-4)

    def test_s(self):
        data = self.model.s(self.w_arr1, self.phase_arr)
        self.assert_json(data, "s")

    def test_theta(self):
        data = self.model.theta(self.w_arr1, self.phase_arr)
        self.assert_json(data, "theta", atol=1.2e-8)

    def test_e_temp(self):
        data = self.model.e_temp(self.temp_arr, self.phase_arr)
        self.assert_json(data, "e_temp")

    def test_export(self):
        self.model.export()
        # self.assertIn("name", data)
        # self.assertin("label", data)

    def test_p_temp(self):
        data = self.model.p_temp(self.temp_arr, self.phase_arr)
        self.assert_json(data, "p_temp", rtol=5e-5)

    def test_s_temp(self):
        data = self.model.s_temp(self.temp_arr, self.phase_arr)
        self.assert_json(data, "s_temp")

    def test_temp(self):
        data = self.model.temp(self.w_arr1, self.phase_arr)
        self.assert_json(data, "temp")

    def test_w(self):
        data = self.model.w(self.temp_arr, self.phase_arr)
        self.assert_json(data, "w")

    def test_w_n(self):
        data = self.model.w_n(self.alpha_n)
        self.assert_json(data, "w_n")
