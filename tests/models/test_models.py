import unittest

import numpy as np

from pttools.bubble.boundary import Phase
from pttools import models
from tests.models.base_model import ModelBaseCase
from tests.utils.assertions import assert_allclose


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

    def test_theta_constant(self):
        """The theta of the bag model is a constant"""
        theta_s = self.model.theta(self.w_arr1, Phase.SYMMETRIC)
        theta_b = self.model.theta(self.w_arr1, Phase.BROKEN)
        assert_allclose(theta_s, np.ones_like(self.w_arr1)*self.model.V_s, atol=1e-16)
        assert_allclose(theta_b, np.ones_like(self.w_arr1)*self.model.V_b, atol=1e-16)


class TestConstCSLikeBag(ModelBaseCase, unittest.TestCase):
    #: This test should use the bag model reference data instead of creating its own
    SAVE_NEW_DATA = False
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

    def test_cs2_like_bag(self):
        assert_allclose(self.model.cs2(self.w_arr1, self.phase_arr), 1/3*np.ones_like(self.w_arr1))


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
