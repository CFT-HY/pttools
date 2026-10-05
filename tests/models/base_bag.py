"""Bag model tests."""

import abc
import typing as tp

import numpy as np
import pytest

from pttools.bubble.phase import Phase
from pttools.models import Model
from pttools.utils.assertions import assert_allclose
from tests.models.base_model import ModelBaseCase


class BagBaseCase[M: Model](ModelBaseCase[M], abc.ABC):
    """Test that a model corresponds to the bag model."""

    #: This test should use the bag model reference data instead of creating its own
    SAVE_NEW_DATA = False

    PARAMS: tp.ClassVar[dict[str, tp.Any]] = {
        "a_s": 1.2,
        "a_b": 1.1,
        "V_s": 1.3
    }
    PARAMS_FULL: tp.ClassVar[dict[str, tp.Any]] = {
        **PARAMS,
        "css2": 1/3,
        "csb2": 1/3,
        "name": "bag"
    }

    def test_alphas_same(self) -> None:
        r"""Test that $\alpha_n$ and $\alpha_+$ coincide for a detonation in the bag model.

        "The two definitions of the transition strength coincide
        only in the case of detonations within the bag model."
        See :notes:` \` p. 40
        """
        wn = 70
        alpha_n = self.model.alpha_n(wn=wn)
        alpha_plus = self.model.alpha_plus(wp=wn, wm=20)
        assert alpha_n == pytest.approx(alpha_plus, abs=5e-8)

    def test_cs2_like_bag(self) -> None:
        """Test that cs2 = 1/3."""
        assert_allclose(self.model.cs2(self.w_arr1, self.phase_arr), 1 / 3 * np.ones_like(self.w_arr1), atol=3.4e-4)

    def test_theta_constant(self) -> None:
        """The theta of the bag model is a constant."""
        theta_s = self.model.theta(self.w_arr1, Phase.SYMMETRIC)
        theta_b = self.model.theta(self.w_arr1, Phase.BROKEN)
        V_s = self.model.V_s
        V_b = self.model.V_b
        # FullModel has implicit V
        if hasattr(self.model, "thermo"):
            thermo = self.model.thermo
            if hasattr(thermo, "V_s"):
                V_s = thermo.V_s
                V_b = thermo.V_b

        assert_allclose(
            np.concatenate((theta_s, theta_b)),
            np.concatenate((
                np.ones_like(self.w_arr1) * V_s,
                np.ones_like(self.w_arr1) * V_b
            )),
            atol=1.9e-8
        )
