"""Tests for various models."""

import typing as tp
import unittest

import numpy as np

from pttools import models
from pttools.bubble.phase import Phase
from pttools.utils import assert_allclose
from tests.models.base_bag import BagBaseCase
from tests.models.base_model import ModelBaseCase


class TestBag(BagBaseCase[models.BagModel], unittest.TestCase):
    """Tests for the bag model."""

    SAVE_NEW_DATA = True

    @classmethod
    def setUpClass(cls, *args: tp.Any, **kwargs: tp.Any) -> None:
        model = models.BagModel(**cls.PARAMS)
        super().setUpClass(model)

    def test_wn_full(self) -> None:
        data = self.model.wn(self.alpha_n, analytical=False)
        self.assert_json(data, "w_n", allow_save=False)

    def test_auto_potential(self) -> None:
        params: dict[str, tp.Any] = {**self.PARAMS, "V_s": None, "V_b": None, "auto_potential": True}
        model = models.BagModel(**params)
        self.assertAlmostEqual(model.critical_temp(), 1)

    # Model initialisation tests

    def test_alpha_n_min(self) -> None:
        models.BagModel(alpha_n_min=0.01)

    def test_g(self) -> None:
        models.BagModel(g_s=120, g_b=100)

    def test_g_s(self) -> None:
        models.BagModel(g_s=120)

    def test_g_b(self) -> None:
        models.BagModel(g_b=100)

    def test_a_g(self) -> None:
        with self.assertRaises(ValueError):
            models.BagModel(a_s=1.5, g_b=1)

    def test_g_a(self) -> None:
        with self.assertRaises(ValueError):
            models.BagModel(g_s=100, a_b=1)

    def test_a_g_all(self) -> None:
        with self.assertRaises(ValueError):
            models.BagModel(a_s=1.5, a_b=1, g_s=120, g_b=100)


class TestConstCSLikeBag(BagBaseCase[models.ConstCSModel], unittest.TestCase):
    """Tests for the constant sound speed model with css2=csb2=1/3."""

    @classmethod
    def setUpClass(cls, *args: tp.Any, **kwargs: tp.Any) -> None:
        # Use bag model reference data
        model = models.ConstCSModel(**cls.PARAMS_FULL)
        super().setUpClass(model)

    def test_constants(self) -> None:
        self.assertAlmostEqual(self.model.mu_s, 4)
        self.assertAlmostEqual(self.model.mu_b, 4)
        self.assertEqual(self.model.T_ref, 1)

    # @unittest.expectedFailure
    def test_critical_temp(self) -> None:
        pass
        # super().test_critical_temp()

    def test_wn_full(self) -> None:
        data = self.model.wn(self.alpha_n, analytical=False)
        self.assert_json(data, "w_n", allow_save=False)


class TestConstCSThermoLikeBag(BagBaseCase[models.FullModel], unittest.TestCase):
    """Tests for the ThermoModel-based constant sound speed model with css2=csb2=1/3."""

    thermo: models.ConstCSThermoModel

    @classmethod
    def setUpClass(cls, *args: tp.Any, **kwargs: tp.Any) -> None:
        cls.thermo = models.ConstCSThermoModel(**cls.PARAMS_FULL)
        model = models.FullModel(thermo=cls.thermo, name="bag")
        super().setUpClass(model)

    def test_constants(self) -> None:
        self.assertAlmostEqual(self.thermo.mu_s, 4)
        self.assertAlmostEqual(self.thermo.mu_b, 4)
        self.assertEqual(self.model.T_ref, 1)

    def test_cs2_full(self) -> None:
        data = self.model.thermo.cs2_full(self.w_arr1, self.phase_arr)
        self.assert_json(data, "cs2")

    # def test_critical_temp(self):
    #     pass


class TestConstCS(ModelBaseCase[models.ConstCSModel], unittest.TestCase):
    """Tests for the constant $c_s$ model."""

    SAVE_NEW_DATA = True

    @classmethod
    def setUpClass(cls, *args: tp.Any, **kwargs: tp.Any) -> None:
        model = models.ConstCSModel(a_s=1.2, a_b=1.1, V_s=1.3, css2=0.4**2, csb2=1/3)
        super().setUpClass(model)

    def test_wn_full(self) -> None:
        data = self.model.wn(self.alpha_n, analytical=False)
        self.assert_json(data, "w_n", allow_save=False)

    def test_inverse_enthalpy_ratio(self) -> None:
        r"""$\Psi(T) = w_b(T) / w_s(T)$, :ai_2023:`\ ` eq. 19."""
        temp = np.linspace(1, 3, 5)
        data = self.model.inverse_enthalpy_ratio(temp)
        ref = self.model.w(temp, Phase.BROKEN) / self.model.w(temp, Phase.SYMMETRIC)
        assert_allclose(data, ref, rtol=1e-12)
        self.assertAlmostEqual(self.model.inverse_enthalpy_ratio(2.), ref[2])

    def test_Psi_scaling(self) -> None:
        r"""$\Psi_+ = \Psi_n \left( \frac{w_+}{w_n} \right)^{\nu/\mu - 1}$, :ai_2023:`\ ` eq. 21b."""
        wn = self.model.wn(0.3)
        wp = np.linspace(1, 2, 5) * wn
        data = self.model.Psi_n(wp)
        ref = self.model.Psi_n(wn) * (wp / wn) ** (self.model.mu_b / self.model.mu_s - 1)
        assert_allclose(data, ref, rtol=1e-12)

    def test_alpha_n_min_V_b(self) -> None:
        """If the given parameters already fulfill the target, they should be used as such."""
        params = {"css2": 1/4, "csb2": 1/4, "a_s": 5, "a_b": 1, "V_s": 1, "V_b": 0.1, "log_info": False}
        target = models.ConstCSModel(**params).alpha_n_min + 0.01
        model = models.ConstCSModel(**params, alpha_n_min=target)
        self.assertEqual((model.a_s, model.a_b, model.V_s, model.V_b), (5, 1, 1, 0.1))
        self.assertLessEqual(model.alpha_n_min, target)

    def test_invalid_cs2(self) -> None:
        for css2, csb2 in ((-0.1, 1/3), (1.1, 1/3), (1/3, -0.1), (1/3, 1.1)):
            with self.subTest(css2=css2, csb2=csb2), self.assertRaisesRegex(ValueError, "have to be"):
                models.ConstCSModel(a_s=1.2, a_b=1.1, V_s=1.3, css2=css2, csb2=csb2)


class TestConstCSThermo(ModelBaseCase[models.FullModel], unittest.TestCase):
    """Tests for the ThermoModel-based constant $c_s$ model."""

    SAVE_NEW_DATA = False

    @classmethod
    def setUpClass(cls, *args: tp.Any, **kwargs: tp.Any) -> None:
        thermo = models.ConstCSThermoModel(a_s=1.2, a_b=1.1, V_s=1.3, css2=0.4**2, csb2=1/3)
        model = models.FullModel(thermo=thermo, name="const_cs")
        super().setUpClass(model)

    def test_cs2_full(self) -> None:
        data = self.model.thermo.cs2_full(self.w_arr1, self.phase_arr)
        self.assert_json(data, "cs2")


class TestSM(ModelBaseCase[models.FullModel], unittest.TestCase):
    """Tests for the Standard Model-based FullModel."""

    # The Standard Model data starts at T = 1 MeV, and the enthalpy there is about 4.7 MeV^4.
    # The test values are chosen to be within the validity range of the model.
    temp_arr = np.linspace(2, 100, ModelBaseCase.TEST_ARR_SIZE)
    w_arr1 = temp_arr**4
    w_arr2 = temp_arr**3.9
    # The transition strength is limited by the minimum temperature of the model.
    alpha_n = np.linspace(0.1, 0.28, 10)

    @classmethod
    def setUpClass(cls, *args: tp.Any, **kwargs: tp.Any) -> None:
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
