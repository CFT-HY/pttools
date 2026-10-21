"""Tests for various models."""

import typing as tp
import unittest

import numpy as np
import pytest

from pttools import models
from pttools.bubble.phase import Phase
from pttools.math import assert_allclose
from tests.models.base_bag import BagBaseCase
from tests.models.base_model import ModelBaseCase

#: Parameters of constant sound speed models for testing the limits of $\alpha_n$.
#: They cover both $c_{s,b}^2 < c_{s,s}^2$ and $c_{s,b}^2 > c_{s,s}^2$, i.e. both signs of $p = \mu_b / \mu_s - 1$.
CONST_CS_PARAMS: tuple[dict[str, tp.Any], ...] = (
    {"css2": 1/4, "csb2": 0.3},
    {"css2": 0.1, "csb2": 0.3},
    {"css2": 1/4, "csb2": 1/4},
    {"css2": 1/3, "csb2": 1/4, "a_s": 5, "a_b": 1, "V_s": 1},
    {"css2": 0.4**2, "csb2": 1/3, "a_s": 1.2, "a_b": 1.1, "V_s": 1.3},
)


class TestBag(BagBaseCase[models.BagModel], unittest.TestCase):
    """Tests for the bag model."""

    SAVE_NEW_DATA = True

    @classmethod
    def setUpClass(cls, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Create the bag model and load its reference data."""
        model = models.BagModel.cached(**cls.PARAMS)
        super().setUpClass(model)

    def test_wn_full(self) -> None:
        r"""Test the numerically solved $w_n$ against the reference data."""
        data = self.model.wn(self.alpha_n, analytical=False)
        self.assert_json(data, "w_n", allow_save=False)

    def test_auto_potential(self) -> None:
        """Test that the automatically generated potential gives the critical temperature of one."""
        params: dict[str, tp.Any] = {**self.PARAMS, "V_s": None, "V_b": None, "auto_potential": True}
        model = models.BagModel.cached(**params)
        assert model.critical_temp() == pytest.approx(1, abs=5e-8)

    # Model initialisation tests

    def test_alpha_n_min(self) -> None:
        """Test that the bag model can be created with alpha_n_min."""
        models.BagModel(alpha_n_min=0.01)

    def test_alpha_n_max_analytical(self) -> None:
        r"""$\alpha_{n,\text{max}} = \frac{4 (V_s - V_b)}{3 w_{\text{min}}}$ for the bag model.

        $\theta = V$ in both phases of the bag model, and therefore $\alpha_n = \frac{4 (V_s - V_b)}{3 w_n}$,
        whose maximum is at the minimum enthalpy $w_{\text{min}} = 4 \max(a_s, a_b) T_{\text{min}}^4$.
        """
        model = models.BagModel(a_s=1.1, a_b=1, V_s=1)
        w_min = 4 * 1.1 * model.T_min**4
        assert model.w_at_alpha_n_max == pytest.approx(w_min, rel=1e-12, abs=0)
        assert model.alpha_n_max == pytest.approx(4 * (1 - 0) / (3 * w_min), rel=1e-12, abs=0)

    def test_g(self) -> None:
        """Test that the bag model can be created with the degrees of freedom of both phases."""
        models.BagModel(g_s=120, g_b=100)

    def test_g_s(self) -> None:
        """Test that the bag model can be created with the degrees of freedom of the symmetric phase."""
        models.BagModel(g_s=120)

    def test_g_b(self) -> None:
        """Test that the bag model can be created with the degrees of freedom of the broken phase."""
        models.BagModel(g_b=100)

    def test_a_g(self) -> None:
        """Test that giving both a_s and g_b raises an error."""
        with pytest.raises(ValueError, match="Specify either a or g values, not both"):
            models.BagModel(a_s=1.5, g_b=1)

    def test_g_a(self) -> None:
        """Test that giving both g_s and a_b raises an error."""
        with pytest.raises(ValueError, match="Specify either a or g values, not both"):
            models.BagModel(g_s=100, a_b=1)

    def test_a_g_all(self) -> None:
        """Test that giving both the a and g parameters for both phases raises an error."""
        with pytest.raises(ValueError, match="Specify either a or g values, not both"):
            models.BagModel(a_s=1.5, a_b=1, g_s=120, g_b=100)


class TestConstCSLikeBag(BagBaseCase[models.ConstCSModel], unittest.TestCase):
    """Tests for the constant sound speed model with css2=csb2=1/3."""

    @classmethod
    def setUpClass(cls, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Create the bag-like ConstCSModel, which uses the bag model reference data."""
        # Use bag model reference data
        model = models.ConstCSModel.cached(**cls.PARAMS_FULL)
        super().setUpClass(model)

    def test_constants(self) -> None:
        """Test the constants of the bag-like ConstCSModel."""
        assert self.model.mu_s == pytest.approx(4, abs=5e-8)
        assert self.model.mu_b == pytest.approx(4, abs=5e-8)
        assert self.model.T_ref == 1

    # @unittest.expectedFailure
    def test_critical_temp(self) -> None:
        """Disable the critical temperature test of the base class for the bag-like ConstCSModel."""
        # super().test_critical_temp()

    def test_wn_full(self) -> None:
        r"""Test the numerically solved $w_n$ against the reference data."""
        data = self.model.wn(self.alpha_n, analytical=False)
        self.assert_json(data, "w_n", allow_save=False)


class TestConstCSThermoLikeBag(BagBaseCase[models.FullModel], unittest.TestCase):
    """Tests for the ThermoModel-based constant sound speed model with css2=csb2=1/3."""

    thermo: tp.ClassVar[models.ConstCSThermoModel]

    @classmethod
    def setUpClass(cls, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Create the bag-like FullModel based on ConstCSThermoModel, which uses the bag model reference data."""
        cls.thermo = models.ConstCSThermoModel.cached(**cls.PARAMS_FULL)
        model = models.FullModel.cached(thermo=cls.thermo, name="bag")
        super().setUpClass(model)

    def test_constants(self) -> None:
        """Test the constants of the bag-like ConstCSThermoModel."""
        assert self.thermo.mu_s == pytest.approx(4, abs=5e-8)
        assert self.thermo.mu_b == pytest.approx(4, abs=5e-8)
        assert self.model.T_ref == 1

    def test_cs2_full(self) -> None:
        r"""Test the full $c_s^2$ of the ThermoModel against the reference data."""
        data = self.model.thermo.cs2_full(self.w_arr1, self.phase_arr)
        self.assert_json(data, "cs2")

    # def test_critical_temp(self):
    #     pass


class TestConstCS(ModelBaseCase[models.ConstCSModel], unittest.TestCase):
    """Tests for the constant $c_s$ model."""

    SAVE_NEW_DATA = True

    @classmethod
    def setUpClass(cls, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Create the constant sound speed model and load its reference data."""
        model = models.ConstCSModel.cached(a_s=1.2, a_b=1.1, V_s=1.3, css2=0.4**2, csb2=1/3)
        super().setUpClass(model)

    def test_wn_full(self) -> None:
        r"""Test the numerically solved $w_n$ against the reference data."""
        data = self.model.wn(self.alpha_n, analytical=False)
        self.assert_json(data, "w_n", allow_save=False)

    def test_alpha_n_max_analytical(self) -> None:
        r"""The analytical $\alpha_{n,\text{max}}$ should equal that of the numerical search of the base class.

        The models of :py:data:`CONST_CS_PARAMS` cover both signs of $p = \mu_b / \mu_s - 1$.
        """
        params: dict[str, tp.Any]
        for params in CONST_CS_PARAMS:
            with self.subTest(**params):
                model = models.ConstCSModel(**params)
                w, alpha_n_max = model.alpha_n_max_find()
                w_num, alpha_n_max_num = models.Model.alpha_n_max_find(model)
                assert (w, alpha_n_max) == (model.w_at_alpha_n_max, model.alpha_n_max)
                assert w == pytest.approx(w_num, rel=1e-12, abs=0)
                assert alpha_n_max == pytest.approx(alpha_n_max_num, rel=1e-12, abs=0)

    def test_alpha_n_min_analytical(self) -> None:
        r"""The analytical $\alpha_{n,\text{min}}$ should equal that of the numerical search of the base class.

        When the search range is extended to contain the stationary point $w_{\text{stat}}$ of $\alpha_n(w)$,
        the minimum should be at it.
        """
        params: dict[str, tp.Any]
        for params in CONST_CS_PARAMS:
            with self.subTest(**params):
                model = models.ConstCSModel(**params)
                w, alpha_n_min = model.alpha_n_min_find()
                w_num, alpha_n_min_num = models.Model.alpha_n_min_find(model)
                assert (w, alpha_n_min) == (model.w_at_alpha_n_min, model.alpha_n_min)
                assert w == pytest.approx(w_num, rel=1e-6, abs=0)
                assert alpha_n_min == pytest.approx(alpha_n_min_num, rel=1e-12, abs=0)

        model = models.ConstCSModel(css2=1/4, csb2=0.3)
        w_stat = model.alpha_n_stationary_point()
        assert model.w_crit < w_stat
        w, alpha_n_min = model.alpha_n_min_find(w_max=10 * w_stat)
        w_num, alpha_n_min_num = models.Model.alpha_n_min_find(model, w_max=10 * w_stat)
        assert w == w_stat
        assert w == pytest.approx(w_num, rel=1e-4, abs=0)
        assert alpha_n_min == pytest.approx(alpha_n_min_num, rel=1e-9, abs=0)

    def test_alpha_n_stationary_point(self) -> None:
        r"""$\frac{d \alpha_n}{d w}$ should be zero at $w_{\text{stat}}$, which should exist only for $B p < 0$."""
        # Whether the stationary point exists for each model of CONST_CS_PARAMS
        exists_all = (True, True, False, False, False)
        for params, exists in zip(CONST_CS_PARAMS, exists_all, strict=True):
            with self.subTest(**params):
                model = models.ConstCSModel(**params)
                w_stat = model.alpha_n_stationary_point()
                assert np.isfinite(w_stat) == exists
                if exists:
                    h = 1e-4 * w_stat
                    derivative = (model.alpha_n(w_stat + h) - model.alpha_n(w_stat - h)) / (2 * h)
                    # The derivative is compared to the scale of the terms of alpha_n, i.e. alpha_n / w.
                    assert abs(derivative) < 1e-6 * model.alpha_n(w_stat) / w_stat

    def test_wn_small(self) -> None:
        r"""$w_n$ should be found numerically also when it is far below 1, i.e. for large $\alpha_n$.

        For $c_{s,b}^2 \neq 1/3$, there is no analytical formula for $w_n$.
        The bracketing root finder used to reject the roots below $10^{-8}$.
        """
        model = models.ConstCSModel.cached(css2=1/4, csb2=0.3)
        for alpha_n in (1e9, 1e11, 0.5 * model.alpha_n_max):
            with self.subTest(alpha_n=alpha_n):
                wn = model.wn(alpha_n)
                assert wn < 1e-8
                assert_allclose(model.alpha_n(wn), alpha_n, rtol=1e-10)

    def test_inverse_enthalpy_ratio(self) -> None:
        r"""$\Psi(T) = w_b(T) / w_s(T)$, :ai_2023:`\ ` eq. 19."""
        temp = np.linspace(1, 3, 5)
        data = self.model.inverse_enthalpy_ratio(temp)
        ref = self.model.w(temp, Phase.BROKEN) / self.model.w(temp, Phase.SYMMETRIC)
        assert_allclose(data, ref, rtol=1e-12)
        assert self.model.inverse_enthalpy_ratio(2.) == pytest.approx(ref[2], abs=5e-8)

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
        target = models.ConstCSModel.cached(**params).alpha_n_min + 0.01
        model = models.ConstCSModel.cached(**params, alpha_n_min=target)
        assert (model.a_s, model.a_b, model.V_s, model.V_b) == (5, 1, 1, 0.1)
        assert model.alpha_n_min <= target

    def test_invalid_cs2(self) -> None:
        """Test that invalid sound speeds raise an error."""
        for css2, csb2 in ((-0.1, 1/3), (1.1, 1/3), (1/3, -0.1), (1/3, 1.1)):
            with self.subTest(css2=css2, csb2=csb2), pytest.raises(ValueError, match="have to be"):
                models.ConstCSModel(a_s=1.2, a_b=1.1, V_s=1.3, css2=css2, csb2=csb2)


class TestConstCSThermo(ModelBaseCase[models.FullModel], unittest.TestCase):
    """Tests for the ThermoModel-based constant $c_s$ model."""

    SAVE_NEW_DATA = False

    @classmethod
    def setUpClass(cls, *args: tp.Any, **kwargs: tp.Any) -> None:
        """Create the ThermoModel-based constant sound speed model and load its reference data."""
        thermo = models.ConstCSThermoModel.cached(a_s=1.2, a_b=1.1, V_s=1.3, css2=0.4**2, csb2=1/3)
        model = models.FullModel.cached(thermo=thermo, name="const_cs")
        super().setUpClass(model)

    def test_cs2_full(self) -> None:
        r"""Test the full $c_s^2$ of the ThermoModel against the reference data."""
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
        """Create the Standard Model-based FullModel and load its reference data."""
        sm = models.StandardModel.cached(V_s=1.3, g_mult_s=1.3)
        model = models.FullModel.cached(thermo=sm)
        super().setUpClass(model)

    # @unittest.expectedFailure
    # def test_alpha_n(self):
    #     super().test_alpha_n()
    #
    # @unittest.expectedFailure
    # def test_alpha_plus(self):
    #     super().test_alpha_plus()
