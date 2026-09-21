"""Unit tests for the shock solver."""

import unittest

import numpy as np

from pttools.bubble.junction import junction_condition_deviation1, junction_condition_deviation2
from pttools.bubble.phase import Phase
from pttools.bubble.relativity import lorentz
from pttools.bubble.shock import solve_shock, v_shock, v_shock_curve
from pttools.bubble.shock_bag import v_shock_bag, wm_shock_bag
from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel
from pttools.utils.assertions import assert_allclose


class TestShock(unittest.TestCase):
    """Unit tests for the shock solver."""

    #: Reference values for the shock curve of the ConstCSModel with $c_{s,s}^2 = 1/4$, generated with PTtools
    XI_REF = np.array([
        0.5, 0.5000811888, 0.5001318325, 0.5002140666,
        0.5003475964, 0.5005644189, 0.5009164904, 0.5014881757,
        0.5024164651, 0.5039237999, 0.5063713749, 0.5103456904,
        0.5167990914, 0.5272779739, 0.5442933395, 0.5719224944,
        0.6167860735, 0.6896345095, 0.8079241055, 1.
    ])
    V_SH_REF = np.array([
        0.0000000000e+00, 2.1648599499e-04, 3.5150712101e-04,
        5.7072218116e-04, 9.2660176229e-04, 1.5042686302e-03,
        2.4417385004e-03, 3.9625803408e-03, 6.4284104024e-03,
        1.0422729405e-02, 1.6883443746e-02, 2.7308871645e-02,
        4.4069481255e-02, 7.0859680986e-02, 1.1330958534e-01,
        1.7973377015e-01, 2.8194553676e-01, 4.3616486442e-01,
        6.6465213344e-01, 1.0000000000e+00
    ])

    @classmethod
    def setUpClass(cls) -> None:
        cls.bag = BagModel(a_s=1.1, a_b=1, V_s=1)
        cls.const_cs_bag_like = ConstCSModel(css2=1/3, csb2=1/4, a_s=5, a_b=1, V_s=1, alpha_n_min=0.1)
        cls.const_cs = ConstCSModel(css2=1/4, csb2=1/4, a_s=5, a_b=1, V_s=1, alpha_n_min=0.1)

    def test_v_shock_bag(self):
        r"""The general shock solver should reproduce the bag model shock curve, :gw_pt_ssm:`\ ` eq. B.17."""
        wn = self.bag.wn(0.1)
        xi, v_sh = v_shock_curve(self.bag, wn=wn)
        self.assertTrue(np.all(np.isfinite(v_sh)))
        # The first point is at xi=cs_n, where v_shock_bag returns nan due to floating point inaccuracy.
        assert_allclose(v_sh[1:], v_shock_bag(xi[1:]), rtol=1e-7)
        self.assertEqual(v_sh[0], 0)

    def test_v_shock_const_cs_bag_like(self):
        r"""A ConstCSModel with $c_{s,s}^2 = 1/3$ should have the same shock curve as the bag model."""
        wn = self.const_cs_bag_like.wn(0.1)
        xi = np.linspace(self.const_cs_bag_like.css, 0.99, 20)
        xi_ret, v_sh = v_shock_curve(self.const_cs_bag_like, wn=wn, xi=xi)
        self.assertIs(xi_ret, xi)
        self.assertTrue(np.all(np.isfinite(v_sh)))
        assert_allclose(v_sh[1:], v_shock_bag(xi[1:]), rtol=1e-7)
        self.assertEqual(v_sh[0], 0)

    def test_v_shock_const_cs(self):
        r"""Shock curve of a ConstCSModel with $c_{s,s}^2 \neq 1/3$."""
        wn = self.const_cs.wn(0.1)
        xi, v_sh = v_shock_curve(self.const_cs, wn=wn)
        self.assertTrue(np.all(np.isfinite(v_sh)))
        assert_allclose(xi, self.XI_REF)
        assert_allclose(v_sh, self.V_SH_REF, rtol=1e-6)
        # The shock curve should start from v=0 at xi=cs_n and increase monotonically to v=1 at xi=1.
        self.assertEqual(xi[0], self.const_cs.css)
        self.assertEqual(v_sh[0], 0)
        self.assertEqual(v_sh[-1], 1)
        self.assertTrue(np.all(np.diff(v_sh) > 0))

    def test_solve_shock_junction_conditions(self):
        r"""The solution of the shock solver should satisfy the junction conditions."""
        for model in (self.bag, self.const_cs):
            with self.subTest(model=model.name):
                wn = model.wn(0.1)
                cs_n = float(np.sqrt(model.cs2(wn, Phase.SYMMETRIC)))
                for xi in np.linspace(cs_n + 0.05, 0.95, 5):
                    with self.subTest(xi=xi):
                        v2_tilde, w2 = solve_shock(model, v1_tilde=xi, w1=wn, backwards=True, csp=cs_n)
                        self.assertTrue(np.isfinite(v2_tilde))
                        self.assertTrue(np.isfinite(w2))
                        p1 = model.p(wn, Phase.SYMMETRIC)
                        p2 = model.p(w2, Phase.SYMMETRIC)
                        dev1 = junction_condition_deviation1(v1=xi, w1=wn, v2=v2_tilde, w2=w2)
                        dev2 = junction_condition_deviation2(v1=xi, w1=wn, p1=p1, v2=v2_tilde, w2=w2, p2=p2)
                        self.assertAlmostEqual(dev1 / wn, 0, places=7)
                        self.assertAlmostEqual(dev2 / wn, 0, places=7)
                        # The shock velocity in the plasma frame is given by the Lorentz transformation
                        assert_allclose(
                            v_shock(model, wn=wn, xi=xi, cs_n=cs_n),
                            lorentz(xi, v2_tilde)
                        )

    def test_solve_shock_bag_enthalpy(self):
        r"""The general shock solver should reproduce the bag model enthalpy behind the shock.

        :gw_pt_ssm:`\ ` eq. B.18.
        """
        wn = self.bag.wn(0.1)
        for xi in np.linspace(0.65, 0.95, 5):
            with self.subTest(xi=xi):
                v2_tilde, w2 = solve_shock(self.bag, v1_tilde=xi, w1=wn, backwards=True)
                assert_allclose(w2, wm_shock_bag(xi, w_n=wn), rtol=1e-7)
                assert_allclose(lorentz(xi, v2_tilde), v_shock_bag(xi), rtol=1e-7)

    def test_v_shock_below_cs(self):
        r"""No shock exists for $\xi \leq c_{s,n}$."""
        wn = self.const_cs.wn(0.1)
        cs_n = float(np.sqrt(self.const_cs.cs2(wn, Phase.SYMMETRIC)))
        self.assertEqual(v_shock(self.const_cs, wn=wn, xi=cs_n, cs_n=cs_n), 0)
        self.assertEqual(v_shock(self.const_cs, wn=wn, xi=0.9 * cs_n, cs_n=cs_n), 0)
