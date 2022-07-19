import unittest

import numpy as np

from pttools import models
from tests.models.base_thermo import ThermoModelBaseCase


class TestStandardModel(ThermoModelBaseCase, unittest.TestCase):
    temp_arr = np.logspace(models.StandardModel.GEFF_DATA[0, 0], models.StandardModel.GEFF_DATA[0, -1], 10)
    phase_arr = np.linspace(0, 1, temp_arr.size)

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        thermo = models.StandardModel()
        super().setUpClass(thermo)
