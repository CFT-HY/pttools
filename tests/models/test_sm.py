import unittest

import numpy as np

from pttools import models
from tests.models.base_thermo import ThermoModelBaseCase


class TestStandardModel(ThermoModelBaseCase, unittest.TestCase):
    temp_arr = np.logspace(models.StandardModel.GEFF_DATA[0, 0], models.StandardModel.GEFF_DATA[0, -1], 10)
    phase_arr = np.linspace(0, 1, temp_arr.size)
    thermo: models.StandardModel

    @classmethod
    def setUpClass(cls, *args, **kwargs):
        thermo = models.StandardModel()
        super().setUpClass(thermo)

    def test_geff_arrays(self):
        """It's easy to accidentally make these into column vectors,
        which will mess up the dimensionality of the spliners."""
        self.assertEqual(self.thermo.GEFF_DATA.ndim, 2)
        self.assertEqual(self.thermo.GEFF_DATA_GE.ndim, 1)
        self.assertEqual(self.thermo.GEFF_DATA_GS.ndim, 1)
        self.assertEqual(self.thermo.GEFF_DATA_GE_GS_RATIO.ndim, 1)
        self.assertEqual(self.thermo.GEFF_DATA_LOG_TEMP.ndim, 1)
        self.assertEqual(self.thermo.GEFF_DATA_TEMP.ndim, 1)

    # def test_phase_invariance(self):
    #     raise NotImplementedError
