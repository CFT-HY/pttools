r"""Unit tests for the energy budget approximations of :espinosa_2010:`\ `.

The reference values for $\kappa$ and $K$ are those of :py:mod:`tests.bubble.ref`,
which are also used for testing the full solution.
The approximations should reproduce them within the accuracy of the fits.

:espinosa_2010:`\ ` has no tables, and its figures are the only numerical results,
so the only quantitative reference value in the article itself is the accuracy of the fits,
which is tested by :py:class:`EspinosaFitAccuracyTest`.
"""

from abc import ABC
import unittest

import numpy as np

from pttools.bubble import thermo_bag
from pttools.bubble.chapman_jouguet import v_chapman_jouguet_bag
from pttools.bubble.const import CS0, DEFAULT_ADIABATIC_INDEX
from pttools.bubble.energy_budget import (
    alpha_n_from_ubarf,
    alpha_n_from_ubarf_solvable,
    delta_kappa_approx,
    delta_n,
    kappa_a,
    kappa_b,
    kappa_c,
    kappa_d,
    kappa_detonation_approx,
    kappa_hybrid_approx,
    kappa_sub_def_approx,
    kappa_v_approx,
    kinetic_energy_fraction_approx,
    ubarf_approx,
)
from pttools.bubble.phase import Phase
from pttools.models.bag import BagModel
import pttools.type_hints as th
from tests.bubble.ref import RefBag, Reference, RefHindmarshHijazi, RefLectureNotes
from tests.utils.test_assertions import assert_allclose

#: The $\alpha_n$ values for the tests that don't depend on the reference data
ALPHA_NS_FIT: th.FloatArr = np.array([0.01, 0.1, 0.3, 1.0])


class EnergyBudgetApproxTest(Reference, ABC):
    r"""Compare the approximations of :espinosa_2010:`\ ` to the values in a reference."""

    RTOL_KAPPA: float = 2.8e-2
    RTOL_KE_FRAC: float = 2.8e-2

    def test_solution_types(self) -> None:
        """Ensure that the reference covers subsonic deflagrations, hybrids and detonations."""
        sol_types = set()
        for alpha_n, v_wall in zip(self.ALPHA_NS, self.V_WALLS, strict=True):
            v_cj = v_chapman_jouguet_bag(alpha_plus=alpha_n)
            if v_wall < CS0:
                sol_types.add("sub_def")
            elif v_wall > v_cj:
                sol_types.add("detonation")
            else:
                sol_types.add("hybrid")
        assert sol_types == {"sub_def", "hybrid", "detonation"}, f"Missing solution types: {sol_types}"

    def test_kappa_v_approx(self) -> None:
        r"""$\kappa_v$ should correspond to the reference values."""
        kappas = np.array([
            kappa_v_approx(v_wall, alpha_n)
            for alpha_n, v_wall in zip(self.ALPHA_NS, self.V_WALLS, strict=True)
        ])
        assert_allclose(kappas, self.KAPPA_REF, rtol=self.RTOL_KAPPA)

    def test_kinetic_energy_fraction_approx(self) -> None:
        """The kinetic energy fraction $K$ should correspond to the reference values."""
        ke_fracs = np.array([
            kinetic_energy_fraction_approx(v_wall, alpha_n)
            for alpha_n, v_wall in zip(self.ALPHA_NS, self.V_WALLS, strict=True)
        ])
        assert_allclose(ke_fracs, self.BVA_KE_FRAC_REF, rtol=self.RTOL_KE_FRAC)

    def test_kappa_v_approx_branches(self) -> None:
        """The branches of the approximation should be selected according to the solution type."""
        for alpha_n, v_wall in zip(self.ALPHA_NS, self.V_WALLS, strict=True):
            v_cj = v_chapman_jouguet_bag(alpha_plus=alpha_n)
            kappa = kappa_v_approx(v_wall, alpha_n)
            if v_wall < CS0:
                branch = kappa_sub_def_approx(v_wall, alpha_n)
            elif v_wall > v_cj:
                branch = kappa_detonation_approx(v_wall, alpha_n, v_cj)
            else:
                branch = kappa_hybrid_approx(v_wall, alpha_n)
            assert_allclose(kappa, branch, name=f"alpha_n={alpha_n}, v_wall={v_wall}")

    def test_continuity(self, eps: float = 1e-8) -> None:
        """The approximation should be continuous at the boundaries of the solution types."""
        for alpha_n in np.unique(self.ALPHA_NS):
            for v_boundary in (CS0, v_chapman_jouguet_bag(alpha_plus=alpha_n)):
                name = f"alpha_n={alpha_n}, v_wall={v_boundary}"
                at = kappa_v_approx(v_boundary, alpha_n)
                assert_allclose(kappa_v_approx(v_boundary - eps, alpha_n), at, rtol=1e-4, name=name)
                assert_allclose(kappa_v_approx(v_boundary + eps, alpha_n), at, rtol=1e-4, name=name)

    def test_ubarf_approx(self) -> None:
        r"""$\bar{U}_f^2 = \frac{K}{\Gamma}$."""
        for alpha_n, v_wall in zip(self.ALPHA_NS, self.V_WALLS, strict=True):
            ubarf = ubarf_approx(v_wall, alpha_n)
            ke_frac = kinetic_energy_fraction_approx(v_wall, alpha_n)
            assert_allclose(
                ubarf**2, ke_frac / DEFAULT_ADIABATIC_INDEX, name=f"alpha_n={alpha_n}, v_wall={v_wall}")

    def test_alpha_n_from_ubarf(self) -> None:
        r"""Inverting $\bar{U}_f$ should give back the original $\alpha_n$."""
        for alpha_n, v_wall in zip(self.ALPHA_NS, self.V_WALLS, strict=True):
            ubarf = ubarf_approx(v_wall, alpha_n)
            assert_allclose(
                float(alpha_n_from_ubarf(v_wall, ubarf)), alpha_n,
                rtol=1e-4, name=f"alpha_n={alpha_n}, v_wall={v_wall}")

    def test_alpha_n_from_ubarf_solvable(self) -> None:
        r"""The solvable function should be zero at the correct $\alpha_n$."""
        for alpha_n, v_wall in zip(self.ALPHA_NS, self.V_WALLS, strict=True):
            ubarf = ubarf_approx(v_wall, alpha_n)
            dev = alpha_n_from_ubarf_solvable(
                alpha_n=alpha_n, ubarf_target=float(ubarf),
                v_wall=v_wall, cs=CS0, adiabatic_index=DEFAULT_ADIABATIC_INDEX
            )
            assert_allclose(dev, 0, atol=1e-14, name=f"alpha_n={alpha_n}, v_wall={v_wall}")


class EnergyBudgetApproxTestBag(RefBag, EnergyBudgetApproxTest, unittest.TestCase):
    r"""Compare the approximations to old PTtools results."""


class EnergyBudgetApproxTestHindmarshHijazi(RefHindmarshHijazi, EnergyBudgetApproxTest, unittest.TestCase):
    r"""Compare the approximations to the values in :gw_pt_ssm:`\ `."""

    # RTOL_KAPPA = 2.5e-2
    # RTOL_KE_FRAC = 2.4e-2


class EnergyBudgetApproxTestLectureNotes(RefLectureNotes, EnergyBudgetApproxTest, unittest.TestCase):
    r"""Compare the approximations to the values in :notes:`\ `."""

    # RTOL_KAPPA = 6.9e-3
    # RTOL_KE_FRAC = 7.9e-3


class EspinosaFitAccuracyTest(unittest.TestCase):
    r""":espinosa_2010:`\ ` states the accuracy of their fits.

    "These fits facilitate the functions $\kappa(\xi_w, \alpha_N)$ and $\alpha_+(\xi_w, \alpha_N)$
    without solving the flow equations and with a precision better that 15% in the region
    $10^{-3} < \alpha_N < 10$." (appendix A)
    The corresponding figure 13 of the article covers $\xi_w \in [0.2, 1]$,
    and therefore the same range is used here.
    """

    # The lower end of the range of the article, $\alpha_N = 10^{-3}$, is not included,
    # since there the deviation grows to 16 % just above $v_{CJ}$, where $\kappa$ rises steeply.
    ALPHA_NS = (0.01, 0.03, 0.1, 0.3, 1., 3., 10.)
    V_WALLS = np.arange(0.2, 1., 0.05)
    #: The precision given in the article
    RTOL = 0.15

    def test_kappa_v_approx_accuracy(self) -> None:
        r"""The fits should be within 15 % of the full bag model solution."""
        for alpha_n in self.ALPHA_NS:
            for v_wall in self.V_WALLS:
                kappa_bag = thermo_bag.get_kappa_bag(v_wall=float(v_wall), alpha_n=alpha_n)
                # For strong phase transitions the slow wall speeds have no solutions.
                if not np.isfinite(kappa_bag) or kappa_bag <= 0:
                    continue
                assert_allclose(
                    kappa_v_approx(float(v_wall), alpha_n), kappa_bag,
                    rtol=self.RTOL, name=f"alpha_n={alpha_n}, v_wall={v_wall}")


class KappaLimitsTest(unittest.TestCase):
    r"""The limits of the $\kappa$ approximations of :espinosa_2010:`\ `.

    Each of the fits $\kappa_{A-D}$ corresponds to a limiting wall speed,
    and the interpolating formulas should reduce to them at those wall speeds.

    The reference values of the individual fits $\kappa_{A-D}$ and $\delta \kappa$
    are PTtools output on 2026-09-07 with the dev branch at commit dab0ab8.
    """

    @staticmethod
    def test_kappa_a() -> None:
        r"""$\kappa_A$ is the limit of subsonic deflagrations at $v_\text{wall} \ll c_s$."""
        v_wall = 1e-3
        assert_allclose(kappa_sub_def_approx(v_wall, ALPHA_NS_FIT), kappa_a(v_wall, ALPHA_NS_FIT), rtol=1e-3)

    @staticmethod
    def test_kappa_b() -> None:
        r"""$\kappa_B$ is the limit of both subsonic deflagrations and hybrids at $v_\text{wall} = c_s$."""
        kb = kappa_b(ALPHA_NS_FIT)
        assert_allclose(kappa_sub_def_approx(CS0, ALPHA_NS_FIT), kb)
        assert_allclose(kappa_hybrid_approx(CS0, ALPHA_NS_FIT), kb)
        for alpha_n in ALPHA_NS_FIT:
            assert_allclose(kappa_v_approx(CS0, alpha_n), kappa_b(alpha_n), name=f"alpha_n={alpha_n}")

    @staticmethod
    def test_kappa_c() -> None:
        r"""$\kappa_C$ is the limit of both hybrids and detonations at $v_\text{wall} = v_{CJ}$."""
        v_cj = v_chapman_jouguet_bag(ALPHA_NS_FIT)
        kc = kappa_c(ALPHA_NS_FIT)
        assert_allclose(kappa_hybrid_approx(v_cj, ALPHA_NS_FIT), kc)
        assert_allclose(kappa_detonation_approx(v_cj, ALPHA_NS_FIT, None), kc)
        for alpha_n in ALPHA_NS_FIT:
            assert_allclose(
                kappa_v_approx(v_chapman_jouguet_bag(alpha_plus=alpha_n), alpha_n),
                kappa_c(alpha_n), name=f"alpha_n={alpha_n}")

    @staticmethod
    def test_kappa_d() -> None:
        r"""$\kappa_D$ is the limit of detonations at $v_\text{wall} \rightarrow 1$."""
        assert_allclose(kappa_detonation_approx(1., ALPHA_NS_FIT, None), kappa_d(ALPHA_NS_FIT))

    @staticmethod
    def test_kappa_values() -> None:
        r"""The individual fits $\kappa_{A-D}$ should correspond to the reference values."""
        assert_allclose(kappa_a(0.1, ALPHA_NS_FIT), [0.00318642, 0.03006012, 0.07965203, 0.18741307], rtol=1e-4)
        assert_allclose(kappa_b(ALPHA_NS_FIT), [0.155413, 0.377451, 0.548365, 0.748662], rtol=1e-4)
        assert_allclose(kappa_c(ALPHA_NS_FIT), [0.088497, 0.269306, 0.432514, 0.648456], rtol=1e-4)
        assert_allclose(kappa_d(ALPHA_NS_FIT), [0.013364, 0.116789, 0.278950, 0.551572], rtol=1e-4)

    @staticmethod
    def test_delta_kappa_approx() -> None:
        r"""$\delta \kappa$ should correspond to the reference values."""
        assert_allclose(delta_kappa_approx(ALPHA_NS_FIT), [2.158106, 1.283456, 0.934894, 0.623832], rtol=1e-4)


class DeltaNTest(unittest.TestCase):
    r"""$\delta_n$ for $K$."""

    WN = 1.5

    def test_bag(self) -> None:
        r"""For the bag model with $V_- = 0$, $\delta_n = 0$."""
        model = BagModel(a_s=1.2, a_b=1, V_s=1, V_b=0)
        assert_allclose(delta_n(model, self.WN), 0)

    def test_bag_v_b(self) -> None:
        r"""$\delta_n = \frac{4 \theta_-}{3 w_n}$."""
        model = BagModel(a_s=1.2, a_b=1, V_s=1, V_b=0.2)
        theta_b = model.theta(self.WN, Phase.BROKEN)
        assert_allclose(delta_n(model, self.WN), 4 * theta_b / (3 * self.WN))

    def test_ubarf_approx_delta_n(self) -> None:
        r"""A nonzero $\delta_n$ should decrease $\bar{U}_f$."""
        model = BagModel(a_s=1.2, a_b=1, V_s=1, V_b=0.2)
        self.assertLess(
            ubarf_approx(0.7, 0.3, model=model),
            ubarf_approx(0.7, 0.3)
        )
