"""Tests for various models"""

import unittest

from pttools import models
from tests.models.base_model import ModelBaseCase
from tests.models.base_bag import BagBaseCase


class TestBag(BagBaseCase, unittest.TestCase):
    """Tests for the bag model"""
    model: models.BagModel
    SAVE_NEW_DATA = True

    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        model = models.BagModel(**cls.PARAMS)
        super().setUpClass(model)

    def test_wn_full(self):
        data = self.model.wn(self.alpha_n, analytical=False)
        self.assert_json(data, "w_n", allow_save=False)

    def test_auto_potential(self):
        params = {**self.PARAMS, "V_s": None, "V_b": None, "auto_potential": True}
        model = models.BagModel(**params)
        self.assertAlmostEqual(model.critical_temp(), 1)

    # Model initialisation tests

    def test_alpha_n_min(self):
        models.BagModel(alpha_n_min=0.01)

    def test_g(self):
        models.BagModel(g_s=120, g_b=100)

    def test_g_s(self):
        models.BagModel(g_s=120)

    def test_g_b(self):
        models.BagModel(g_b=100)

    def test_a_g(self):
        with self.assertRaises(ValueError):
            models.BagModel(a_s=1.5, g_b=1)

    def test_g_a(self):
        with self.assertRaises(ValueError):
            models.BagModel(g_s=100, a_b=1)

    def test_a_g_all(self):
        with self.assertRaises(ValueError):
            models.BagModel(a_s=1.5, a_b=1, g_s=120, g_b=100)


class TestConstCSLikeBag(BagBaseCase, unittest.TestCase):
    """Tests for the constant sound speed model with css2=csb2=1/3"""
    model: models.ConstCSModel

    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        # Use bag model reference data
        model = models.ConstCSModel(**cls.PARAMS_FULL)
        super().setUpClass(model)

    def test_constants(self):
        self.assertAlmostEqual(self.model.mu_s, 4)
        self.assertAlmostEqual(self.model.mu_b, 4)
        self.assertEqual(self.model.T_ref, 1)

    # @unittest.expectedFailure
    def test_critical_temp(self):
        pass
        # super().test_critical_temp()

    def test_wn_full(self):
        data = self.model.wn(self.alpha_n, analytical=False)
        self.assert_json(data, "w_n", allow_save=False)


class TestConstCSThermoLikeBag(BagBaseCase, unittest.TestCase):
    """Tests for the ThermoModel-based constant sound speed model with css2=csb2=1/3"""
    model: models.FullModel

    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        model = models.FullModel(
            thermo=models.ConstCSThermoModel(**cls.PARAMS_FULL), name="bag")
        super().setUpClass(model)

    def test_constants(self):
        self.assertAlmostEqual(self.model.thermo.mu_s, 4)
        self.assertAlmostEqual(self.model.thermo.mu_b, 4)
        self.assertEqual(self.model.T_ref, 1)

    def test_cs2_full(self):
        data = self.model.thermo.cs2_full(self.w_arr1, self.phase_arr)
        self.assert_json(data, "cs2")

    # def test_critical_temp(self):
    #     pass


class TestConstCS(ModelBaseCase, unittest.TestCase):
    """Tests for the constant $c_s$ model"""
    model: models.ConstCSModel
    SAVE_NEW_DATA = True

    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        model = models.ConstCSModel(a_s=1.2, a_b=1.1, V_s=1.3, css2=0.4**2, csb2=1/3)
        super().setUpClass(model)

    def test_wn_full(self):
        data = self.model.wn(self.alpha_n, analytical=False)
        self.assert_json(data, "w_n", allow_save=False)


class TestConstCSThermo(ModelBaseCase, unittest.TestCase):
    """Tests for the ThermoModel-based constant $c_s$ model"""
    model: models.FullModel
    SAVE_NEW_DATA = False

    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        thermo = models.ConstCSThermoModel(a_s=1.2, a_b=1.1, V_s=1.3, css2=0.4**2, csb2=1/3)
        model = models.FullModel(thermo=thermo, name="const_cs")
        super().setUpClass(model)

    def test_cs2_full(self):
        data = self.model.thermo.cs2_full(self.w_arr1, self.phase_arr)
        self.assert_json(data, "cs2")


class TestSM(ModelBaseCase, unittest.TestCase):
    """Tests for the Standard Model-based FullModel"""
    @classmethod
    def setUpClass(cls, *args, **kwargs) -> None:
        sm = models.StandardModel(V_s=1.3, g_mult_s=1.3)
        model = models.FullModel(thermo=sm)
        super().setUpClass(model)

    # @unittest.expectedFailure
    # def test_alpha_n(self):
    #     super().test_alpha_n()
    #
    # @unittest.expectedFailure
    # def test_alpha_plus(self):
    #     super().test_alpha_plus()
