import unittest

from pttools import models
from tests.models.base_model import ModelBaseCase


class TestBag(ModelBaseCase, unittest.TestCase):
    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        model = models.BagModel(a_s=1.2, a_b=1.1, V_s=1.3)
        super().setUpClass(model)

    def test_alphas_same(self):
        r""""The two definitions of the transition strength coincide
        only in the case of detonations within the bag model."

        See :notes:` \` p. 40
        """
        wn = 1.1
        alpha_n = self.model.alpha_n(wn=wn)
        alpha_plus = self.model.alpha_plus(wp=wn, wm=0.9)
        self.assertAlmostEqual(alpha_n, alpha_plus)


class TestConstCSLikeBag(ModelBaseCase, unittest.TestCase):
    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        model = models.ConstCSModel(a_s=1.1, a_b=1.2, V_s=1.3, css2=1/3, csb2=1/3)
        super().setUpClass(model)


class TestConstCS(ModelBaseCase, unittest.TestCase):
    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        model = models.ConstCSModel(a_s=1.1, a_b=1.2, V_s=1.3, css2=0.4**2, csb2=1/3)
        super().setUpClass(model)


class TestFull(ModelBaseCase, unittest.TestCase):
    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        sm = models.StandardModel()
        model = models.FullModel(thermo=sm, V_s=1.1)
        super().setUpClass(model)

    # @unittest.expectedFailure
    # def test_alpha_n(self):
    #     super().test_alpha_n()
    #
    # @unittest.expectedFailure
    # def test_alpha_plus(self):
    #     super().test_alpha_plus()
