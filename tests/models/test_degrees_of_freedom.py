r"""Tests for the effective degrees of freedom $g_e$, $g_p$ and $g_s$ of the models.

The degrees of freedom are defined by
$$e = \frac{\pi^2}{30} g_e T^4, \quad p = \frac{\pi^2}{90} g_p T^4, \quad s = \frac{2\pi^2}{45} g_s T^3,$$
:borsanyi_2016:`\ ` eq. S12, where the potential $V$ of the model, if any, is included in $g_e$ and $g_p$.
For the bag model, $g_s$ equals $g = \frac{90}{\pi^2} a$, :maki_msc:`\ ` eq. 2.110,
and so do $g_e$ and $g_p$ when $V = 0$.
"""

import typing as tp
import unittest

import numpy as np

from pttools import models
from pttools.bubble.phase import Phase

PHASES: tuple[Phase, Phase] = (Phase.SYMMETRIC, Phase.BROKEN)


class AnalyticDegreesOfFreedomTest(unittest.TestCase):
    """Tests for the degrees of freedom of the analytic models."""

    temp: np.ndarray = np.linspace(0.5, 2, 7)

    bag_v: models.BagModel
    const_cs: models.ConstCSModel

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.bag_v = models.BagModel(g_s=123, g_b=100, V_s=0.5, V_b=0.1)
        cls.const_cs = models.ConstCSModel(css2=1/3 - 0.02, csb2=1/3 - 0.03, a_s=1.5, a_b=1, V_s=1, V_b=0)

    def g_bag(self, model: models.BagModel, phase: Phase) -> float:
        """Degrees of freedom of the bag model in the given phase."""
        return model.g_from_a(model.a_b if phase == Phase.BROKEN else model.a_s)

    def test_bag_g_potential(self) -> None:
        r"""Test that the potential is included in $g_e$ and $g_p$ but not in $g_s$ for the bag model.

        $$g_e = g + \frac{30 V}{\pi^2 T^4}, \quad g_p = g - \frac{90 V}{\pi^2 T^4}, \quad g_s = g$$
        """
        for phase in PHASES:
            with self.subTest(phase=phase):
                g = self.g_bag(self.bag_v, phase)
                V = self.bag_v.V_b if phase == Phase.BROKEN else self.bag_v.V_s
                np.testing.assert_allclose(
                    self.bag_v.ge_temp(self.temp, phase), g + 30 * V / (np.pi**2 * self.temp**4), rtol=1e-14)
                np.testing.assert_allclose(
                    self.bag_v.gp_temp(self.temp, phase), g - 90 * V / (np.pi**2 * self.temp**4), rtol=1e-14)
                np.testing.assert_allclose(self.bag_v.gs_temp(self.temp, phase), g, rtol=1e-14)

    def test_definitions(self) -> None:
        r"""Test the definitions of $g_e$, $g_p$ and $g_s$ for the constant sound speed model."""
        model = self.const_cs
        temp = self.temp
        for phase in PHASES:
            with self.subTest(phase=phase):
                np.testing.assert_allclose(
                    np.pi**2 / 30 * model.ge_temp(temp, phase) * temp**4, model.e_temp(temp, phase), rtol=1e-14)
                np.testing.assert_allclose(
                    np.pi**2 / 90 * model.gp_temp(temp, phase) * temp**4, model.p_temp(temp, phase), rtol=1e-14)
                np.testing.assert_allclose(
                    2 * np.pi**2 / 45 * model.gs_temp(temp, phase) * temp**3, model.s_temp(temp, phase), rtol=1e-14)

    def test_g_of_w(self) -> None:
        r"""Test that $g_e(w)$, $g_p(w)$ and $g_s(w)$ correspond to $g_e(T)$, $g_p(T)$ and $g_s(T)$."""
        for model in (self.bag_v, self.const_cs):
            for phase in PHASES:
                w = model.w(self.temp, phase)
                for name in ("ge", "gp", "gs"):
                    with self.subTest(model=model.name, phase=phase, name=name):
                        np.testing.assert_allclose(
                            getattr(model, name)(w, phase),
                            getattr(model, f"{name}_temp")(self.temp, phase),
                            rtol=1e-12
                        )


class FullModelDegreesOfFreedomTest(unittest.TestCase):
    """Tests for the degrees of freedom of the full model."""

    model: models.FullModel

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.model = models.FullModel(thermo=models.StandardModel(V_s=5e12, g_mult_s=1 + 1e-9, silence_temp=True))

    def test_g_of_w(self) -> None:
        r"""Test that $g_e(w)$, $g_p(w)$ and $g_s(w)$ correspond to those of the thermodynamics model."""
        temp = np.logspace(1, 5, 9)
        thermo = self.model.thermo
        for phase in PHASES:
            w = self.model.w(temp, phase)
            temp_w = self.model.temp(w, phase)
            for name in ("ge", "gp", "gs"):
                with self.subTest(phase=phase, name=name):
                    np.testing.assert_allclose(
                        getattr(self.model, name)(w, phase), getattr(thermo, name)(temp_w, phase), rtol=1e-14)

    def test_g_differ(self) -> None:
        r"""Test that $g_e$, $g_p$ and $g_s$ are different for the Standard Model, so that mixing them is detected."""
        w = self.model.w(np.array([200.]), Phase.BROKEN)
        ge = self.model.ge(w, Phase.BROKEN)
        gp = self.model.gp(w, Phase.BROKEN)
        gs = self.model.gs(w, Phase.BROKEN)
        self.assertGreater(np.abs(ge - gs).item(), 1e-2)
        self.assertGreater(np.abs(ge - gp).item(), 1e-2)


class StandardModelPotentialTest(unittest.TestCase):
    """Tests for the Standard Model with a potential."""

    thermo: models.StandardModel

    @classmethod
    @tp.override
    def setUpClass(cls) -> None:
        cls.thermo = models.StandardModel(V_s=2e8, V_b=5e7, gen_cs2=False)

    def test_ge_potential(self) -> None:
        r"""Test that $g_e$ includes the potential of the phase, $\frac{30 V(\phi)}{\pi^2 T^4}$."""
        temp = np.logspace(1.5, 3, 7)
        thermo_no_v = models.StandardModel(gen_cs2=False)
        for phase in PHASES:
            with self.subTest(phase=phase):
                V = self.thermo.V(phase)
                np.testing.assert_allclose(
                    self.thermo.ge(temp, phase),
                    thermo_no_v.ge(temp, phase) + 30 * V / (np.pi**2 * temp**4),
                    rtol=1e-14
                )

    def test_dge_dT(self) -> None:
        r"""Test $\frac{dg_e}{dT}$ against a numerical derivative of $g_e$ in both phases."""
        temp = np.logspace(1.5, 3, 7)
        h = temp * 1e-6
        for phase in PHASES:
            with self.subTest(phase=phase):
                numerical = (self.thermo.ge(temp + h, phase) - self.thermo.ge(temp - h, phase)) / (2 * h)
                np.testing.assert_allclose(self.thermo.dge_dT(temp, phase), numerical, rtol=1e-6)
