r"""Tests for the fluid shell solver based on :giese_2021:`\ `."""

import logging
import unittest
import warnings

import numpy as np

from pttools.bubble.bubble import Bubble
from pttools.bubble.solution_type import SolutionType
from pttools.models import ConstCSModel, gksvdv_models

#: Quantities at the wall and at the shock that should be the same for both solvers.
JUNCTION_QUANTITIES: tuple[str, ...] = (
    "vp", "vm", "vp_tilde", "vm_tilde", "wp", "wm", "wn", "v_sh", "vm_sh", "vm_tilde_sh", "wm_sh", "alpha_plus"
)


class FluidGKSVDVTest(unittest.TestCase):
    r"""Compare the :giese_2021:`\ ` solver with the PTtools solver.

    The solvers use different methods, so their results are not exactly the same,
    but the fluid velocities and enthalpies at the wall and at the shock,
    and the efficiency factors computed from the profiles should agree.
    """

    model: ConstCSModel

    @classmethod
    def setUpClass(cls) -> None:
        cls.model = gksvdv_models()[1]

    def compare(self, v_wall: float, alpha_n: float, sol_type: SolutionType, rtol: float = 1e-3) -> None:
        pttools_bubble = Bubble(self.model, v_wall=v_wall, alpha_n=alpha_n)
        giese_bubble = Bubble(self.model, v_wall=v_wall, alpha_n=alpha_n, use_giese_solver=True)
        self.assertEqual(pttools_bubble.sol_type, sol_type)
        self.assertEqual(giese_bubble.sol_type, sol_type)
        self.assertFalse(giese_bubble.solver_failed)
        self.assertFalse(giese_bubble.invalid_junction)
        # The shock is located with different resolutions, so the tiny velocities there are compared with atol.
        for name in JUNCTION_QUANTITIES:
            with self.subTest(name=name):
                np.testing.assert_allclose(
                    getattr(giese_bubble, name), getattr(pttools_bubble, name), rtol=rtol, atol=1e-4
                )
        # The efficiency factors are integrated over the profiles, which have different resolutions.
        for name in ("kappa", "omega"):
            with self.subTest(name=name):
                np.testing.assert_allclose(getattr(giese_bubble, name), getattr(pttools_bubble, name), rtol=1e-2)
        # The velocities in the plasma frame should be at most the wall velocity.
        self.assertLessEqual(giese_bubble.vp, v_wall)
        self.assertLessEqual(giese_bubble.vm, v_wall)

    def test_sub_def(self):
        self.compare(v_wall=0.4, alpha_n=0.1, sol_type=SolutionType.SUB_DEF)

    def test_hybrid(self):
        self.compare(v_wall=0.6, alpha_n=0.1, sol_type=SolutionType.HYBRID)

    def test_deton(self):
        self.compare(v_wall=0.85, alpha_n=0.1, sol_type=SolutionType.DETON)

    def test_deton_junction(self):
        """For a detonation, the fluid in front of the wall is at rest and has the nucleation enthalpy."""
        bubble = Bubble(self.model, v_wall=0.85, alpha_n=0.1, use_giese_solver=True)
        self.assertEqual(bubble.sol_type, SolutionType.DETON)
        self.assertEqual(bubble.vp, 0)
        self.assertAlmostEqual(bubble.vp_tilde, bubble.v_wall)
        self.assertAlmostEqual(bubble.wp, bubble.wn)
        self.assertGreater(bubble.wm, bubble.wn)

    def test_failure(self):
        """A solver failure should be reported by the solver only, without further warnings from the validations."""
        model = gksvdv_models()[0]
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with self.assertNoLogs("pttools.utils.validation", level=logging.ERROR):
                bubble = Bubble(model, v_wall=0.2, alpha_n=1.0, allow_invalid=False, use_giese_solver=True)
        self.assertTrue(bubble.solver_failed)
        self.assertTrue(bubble.failed)
        self.assertEqual(bubble.sol_type, SolutionType.ERROR)
        self.assertTrue(np.isnan(bubble.alpha_plus))
        self.assertTrue(np.all(np.isnan(bubble.v)))
