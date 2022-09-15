import unittest

from pttools import models
from tests.models.base_model import ModelBaseCase
from tests.models.base_bag import BagBaseCase


class TestBag(BagBaseCase, unittest.TestCase):
    SAVE_NEW_DATA = True

    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        model = models.BagModel(a_s=1.2, a_b=1.1, V_s=1.3)
        super().setUpClass(model)


class TestConstCSLikeBag(BagBaseCase, unittest.TestCase):
    model: models.ConstCSModel

    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        # Use bag model reference data
        model = models.ConstCSModel(a_s=1.2, a_b=1.1, V_s=1.3, css2=1/3, csb2=1/3, name="bag")
        super().setUpClass(model)

    def test_constants(self):
        self.assertAlmostEqual(self.model.mu, 4)
        self.assertAlmostEqual(self.model.nu, 4)
        self.assertEqual(self.model.t_ref, 1)

    @unittest.expectedFailure
    def test_critical_temp(self):
        super().test_critical_temp()


class TestConstCS(ModelBaseCase, unittest.TestCase):
    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        model = models.ConstCSModel(a_s=1.2, a_b=1.1, V_s=1.3, css2=0.4**2, csb2=1/3)
        super().setUpClass(model)


class TestFull(ModelBaseCase, unittest.TestCase):
    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        sm = models.StandardModel()
        model = models.FullModel(thermo=sm, V_s=1.3)
        super().setUpClass(model)

    # @unittest.expectedFailure
    # def test_alpha_n(self):
    #     super().test_alpha_n()
    #
    # @unittest.expectedFailure
    # def test_alpha_plus(self):
    #     super().test_alpha_plus()
