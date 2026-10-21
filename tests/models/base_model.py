"""Base test case for models."""

import abc
from pathlib import Path
import typing as tp
import unittest

import numpy as np
import pytest

from pttools.bubble import Bubble
from pttools.bubble.energy_budget import ubarf_approx
from pttools.bubble.phase import Phase
from pttools.math.assertions import assert_allclose
from pttools.models import Model
from pttools.type_hints import FloatArr1D
from tests.utils.const import TEST_DATA_PATH
from tests.utils.json import JsonTestCase


class ModelBaseCase[M: Model](JsonTestCase, abc.ABC):
    """Base test case for models."""

    model: M

    EXPECT_MISSING_DATA = True
    SAVE_NEW_DATA = True

    REF_DATA_PATH: tp.ClassVar[Path]
    TEST_ARR_SIZE: tp.ClassVar[int] = 10

    alpha_n: tp.ClassVar[FloatArr1D] = np.linspace(0.15, 0.5, 10)
    temp_arr: tp.ClassVar[FloatArr1D] = np.linspace(1, 100, TEST_ARR_SIZE)
    w_arr1: tp.ClassVar[FloatArr1D] = temp_arr**4
    w_arr2: tp.ClassVar[FloatArr1D] = temp_arr**3.9
    phase_arr: tp.ClassVar[FloatArr1D] = np.array([(1 + (-1)**i)/2 for i in range(TEST_ARR_SIZE)])

    @classmethod
    def setUpClass(cls, model: M) -> None:
        """Set the model to be tested and load its reference data.

        :param model: model to be tested
        """
        cls.model = model
        cls.REF_DATA_PATH = TEST_DATA_PATH / "models" / f"{model.name}.json"
        super().setUpClass()

    def test_class_is_valid(self) -> None:
        """Test that the test arrays have the same size."""
        sizes = np.array([self.w_arr1.size, self.w_arr2.size, self.phase_arr.size, self.temp_arr.size])
        if np.any(sizes != self.w_arr1.size):
            raise ValueError(f"Test arrays must have the same shape. Got: {sizes}")

    def test_alpha_n(self) -> None:
        r"""Test the transition strength $\alpha_n$ against the reference data."""
        data = self.model.alpha_n(self.w_arr1, error_on_invalid=False, log_invalid=False)
        self.assert_json(data, "alpha_n")

    def test_alpha_n_from_ubarf(self) -> None:
        r"""$\alpha_n(\bar{U}_f)$ should invert $\bar{U}_f(\alpha_n)$ within the valid range of the model."""
        if np.isnan(self.model.w_crit):
            raise unittest.SkipTest("The critical temperature has not been generated for this model.")
        model = self.model
        v_wall = 0.5
        alpha_ns = np.geomspace(max(1.1 * model.alpha_n_min, 0.05), min(0.9 * model.alpha_n_max, 1), 4)
        ubarf = np.array([ubarf_approx(v_wall=v_wall, alpha_n=alpha_n, model=model) for alpha_n in alpha_ns])
        assert_allclose(model.alpha_n_from_ubarf(v_wall=v_wall, ubarf=ubarf), alpha_ns, rtol=1e-4)
        with pytest.raises(ValueError, match="different signs"):
            model.alpha_n_from_ubarf(
                v_wall=v_wall, ubarf=0.5 * ubarf_approx(v_wall=v_wall, alpha_n=model.alpha_n_min, model=model))

    def test_alpha_n_max(self) -> None:
        r"""$\alpha_{n,\text{max}}$ should be the maximum of $\alpha_n(w)$.

        The maximum is over $w \in [w_{\text{min}}, w_{\text{crit}}]$.
        """
        if np.isnan(self.model.w_crit):
            raise unittest.SkipTest("The critical temperature has not been generated for this model.")
        model = self.model
        assert model.w_at_alpha_n_max is not None
        assert model.alpha_n_min < model.alpha_n_max
        assert model.w_min <= model.w_at_alpha_n_max <= model.w_crit
        assert_allclose(model.alpha_n(model.w_at_alpha_n_max), model.alpha_n_max, rtol=1e-12)
        w = np.geomspace(model.w_min, model.w_crit, 1000)
        assert np.nanmax(model.alpha_n(w, error_on_invalid=False, log_invalid=False)) <= model.alpha_n_max

    def test_validate_alpha_n(self) -> None:
        r"""$\alpha_n$ outside $[\alpha_{n,\text{min}}, \alpha_{n,\text{max}}]$ should be invalid also for bubbles."""
        if np.isnan(self.model.w_crit):
            raise unittest.SkipTest("The critical temperature has not been generated for this model.")
        model = self.model
        model.validate_alpha_n(model.alpha_n_max, log_invalid=False)
        for alpha_n in (0.5 * model.alpha_n_min, 2 * model.alpha_n_max):
            if alpha_n == 0:
                continue
            with pytest.raises(ValueError, match="Invalid alpha_n"):
                model.validate_alpha_n(alpha_n, log_invalid=False)
            with pytest.raises(ValueError, match="Invalid alpha_n"):
                Bubble(model, v_wall=0.5, alpha_n=alpha_n, solve=False, log_invalid=False)

    def test_wn_alpha_n_max(self) -> None:
        r"""$w_n$ should be found for $\alpha_n$ up to $\alpha_{n,\text{max}}$."""
        if np.isnan(self.model.w_crit):
            raise unittest.SkipTest("The critical temperature has not been generated for this model.")
        model = self.model
        for alpha_n in (0.5 * model.alpha_n_max, 0.999 * model.alpha_n_max):
            wn = model.wn(alpha_n)
            assert model.w_min <= wn <= model.w_crit
            assert_allclose(model.alpha_n(wn), alpha_n, rtol=1e-6)

    def test_alpha_plus(self) -> None:
        r"""Test the transition strength $\alpha_+$ against the reference data."""
        data = self.model.alpha_plus(self.w_arr1, self.w_arr2, error_on_invalid=False, log_invalid=False)
        self.assert_json(data, "alpha_plus")

    def test_critical_temp(self) -> None:
        """Test the critical temperature against the reference data."""
        data = self.model.critical_temp(guess=1)
        self.assert_json(data, "critical_temp")

    def test_criticals(self) -> None:
        """The critical temperature and enthalpy should be consistent with each other."""
        if np.isnan(self.model.T_crit):
            raise unittest.SkipTest("The critical temperature has not been generated for this model.")
        assert self.model.T_min < self.model.T_crit < self.model.T_max
        assert_allclose(self.model.critical_temp(guess=self.model.T_crit), self.model.T_crit, rtol=1e-10)
        assert_allclose(self.model.w(self.model.T_crit, Phase.SYMMETRIC), self.model.w_crit, rtol=1e-10)

    def test_cs2(self) -> None:
        r"""Test the speed of sound squared $c_s^2$ against the reference data."""
        data = self.model.cs2(self.w_arr1, self.phase_arr)
        self.assert_json(data, "cs2", rtol=3.4e-4)

    def test_e(self) -> None:
        """Test the energy density as a function of the enthalpy against the reference data."""
        data = self.model.e(self.w_arr1, self.phase_arr)
        self.assert_json(data, "e")

    def test_p(self) -> None:
        """Test the pressure as a function of the enthalpy against the reference data."""
        data = self.model.p(self.w_arr1, self.phase_arr)
        self.assert_json(data, "p", rtol=2.3e-4)

    def test_s(self) -> None:
        """Test the entropy density as a function of the enthalpy against the reference data."""
        data = self.model.s(self.w_arr1, self.phase_arr)
        self.assert_json(data, "s")

    def test_theta(self) -> None:
        r"""Test the trace anomaly $\theta$ against the reference data."""
        data = self.model.theta(self.w_arr1, self.phase_arr)
        self.assert_json(data, "theta", atol=1.2e-8)

    def test_e_temp(self) -> None:
        """Test the energy density as a function of the temperature against the reference data."""
        data = self.model.e_temp(self.temp_arr, self.phase_arr)
        self.assert_json(data, "e_temp")

    def test_export(self) -> None:
        """Test that the model can be exported."""
        self.model.export()
        # assert "name" in data
        # assert "label" in data

    def test_p_temp(self) -> None:
        """Test the pressure as a function of the temperature against the reference data."""
        data = self.model.p_temp(self.temp_arr, self.phase_arr)
        self.assert_json(data, "p_temp", rtol=5e-5)

    def test_s_temp(self) -> None:
        """Test the entropy density as a function of the temperature against the reference data."""
        data = self.model.s_temp(self.temp_arr, self.phase_arr)
        self.assert_json(data, "s_temp")

    def test_temp(self) -> None:
        """Test the temperature as a function of the enthalpy against the reference data."""
        data = self.model.temp(self.w_arr1, self.phase_arr)
        self.assert_json(data, "temp")

    def test_w(self) -> None:
        """Test the enthalpy as a function of the temperature against the reference data."""
        data = self.model.w(self.temp_arr, self.phase_arr)
        self.assert_json(data, "w")

    def test_w_n(self) -> None:
        r"""Test the nucleation enthalpy $w_n$ against the reference data."""
        data = self.model.wn(self.alpha_n)
        self.assert_json(data, "w_n")
