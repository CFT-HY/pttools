import unittest

from pttools import models
from tests.models.base_model import ModelBaseCase


class TestBag(ModelBaseCase, unittest.TestCase):
    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        model = models.BagModel(a_s=1, a_b=1, V_s=1)
        super().setUpClass(model)

    @unittest.expectedFailure
    def test_critical_temp(self):
        """Not yet implemented for this model"""
        super().test_critical_temp()


class TestConstCS(ModelBaseCase, unittest.TestCase):
    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        model = models.ConstCSModel(a_s=1, a_b=1, V_s=1, css2=0.4**2, csb2=1/3)
        super().setUpClass(model)


class TestFull(ModelBaseCase, unittest.TestCase):
    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        sm = models.StandardModel()
        model = models.FullModel(thermo=sm)
        super().setUpClass(model)
