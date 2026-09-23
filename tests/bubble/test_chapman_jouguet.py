r"""Tests for the Chapman-Jouguet speed $v_{CJ}$."""

import unittest

import numpy as np
from scipy.optimize import brentq

from pttools.bubble.chapman_jouguet import v_chapman_jouguet, v_chapman_jouguet_bag, wm_chapman_jouguet
from pttools.bubble.const import CS0
from pttools.bubble.gksvdv.gksvdv21 import getvm
from pttools.bubble.phase import Phase
from pttools.bubble.relativity import gamma2
from pttools.models import BagModel, ConstCSModel, gksvdv_models
from pttools.models.model import Model
import pttools.type_hints as th
from pttools.utils import assert_allclose

ALPHA_NS: th.FloatArr = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.5])
#: Tolerance for accepting a root of the deviation instead of a pole
DEVIATION_ROOT_TOL = 1e-10
#: Solution type of a detonation in the code of Giese et al.
GKSVDV_DETON = 2
#: Solution type of a hybrid in the code of Giese et al.
GKSVDV_HYBRID = 1


def v_cj_junction(model: Model, alpha_n: float) -> float:
    r"""Chapman-Jouguet speed directly from the junction conditions.

    Uses $\tilde{v}_+ \tilde{v}_- = \frac{{p}_+ - {p}_-}{{e}_+ - {e}_-}$ and
    $\frac{\tilde{v}_+}{\tilde{v}_-} = \frac{{e}_- + {p}_+}{{e}_+ + {p}_-}$
    with $\tilde{v}_-^2 = c_{s,-}^2({w}_-)$ and ${w}_+ = {w}_n$,
    searching for the smallest root with $0 < \tilde{v}_- < \tilde{v}_+ < 1$.
    This does not use $\alpha_+$, and is therefore independent of the implementation in PTtools.
    """
    wp = model.wn(alpha_n)
    ep = model.e(wp, Phase.SYMMETRIC)
    pp = model.p(wp, Phase.SYMMETRIC)

    def deviation(log_wm: float) -> float:
        wm = np.exp(log_wm)
        em = model.e(wm, Phase.BROKEN)
        pm = model.p(wm, Phase.BROKEN)
        return (pp - pm) * (ep + pm) / ((ep - em) * (em + pp)) - model.cs2(wm, Phase.BROKEN)

    log_wms = np.log(wp) + np.linspace(1e-6, 3, 3001)
    devs = np.array([deviation(log_wm) for log_wm in log_wms])
    # The deviation also changes its sign at its poles, and therefore the roots have to be validated.
    for ind in np.flatnonzero(np.sign(devs[:-1]) != np.sign(devs[1:])):
        wm = np.exp(brentq(deviation, log_wms[ind], log_wms[ind + 1], xtol=1e-14))
        em = model.e(wm, Phase.BROKEN)
        pm = model.p(wm, Phase.BROKEN)
        x = (pp - pm) / (ep - em)
        y = (em + pp) / (ep + pm)
        if x > 0 and y > 1 and x * y < 1 and abs(deviation(np.log(wm))) < DEVIATION_ROOT_TOL:
            return np.sqrt(x * y)
    raise RuntimeError(f"No Chapman-Jouguet detonation was found for alpha_n={alpha_n}.")


def v_cj_gksvdv(alpha_theta_bar_n: float, csb2: float) -> float:
    r"""Chapman-Jouguet speed from the :giese_2021:`\ ` code.

    The detonation solution of :func:`pttools.bubble.gksvdv.gksvdv21.getvm` exists only for $v_w \geq v_{CJ}$.
    """
    def discriminant(vw: float) -> float:
        cc = 1 - 3 * alpha_theta_bar_n + vw**2 * (1 / csb2 + 3 * alpha_theta_bar_n)
        return cc**2 - 4 * vw**2 / csb2

    v_cj = brentq(discriminant, np.sqrt(csb2) + 1e-12, 1 - 1e-12, xtol=1e-15)
    if getvm(alpha_theta_bar_n, v_cj + 1e-9, csb2)[1] != GKSVDV_DETON \
            or getvm(alpha_theta_bar_n, v_cj - 1e-9, csb2)[1] != GKSVDV_HYBRID:
        raise RuntimeError("The Chapman-Jouguet speed is not at the detonation-hybrid boundary.")
    return v_cj


class ChapmanJouguetTest(unittest.TestCase):
    """Tests for the Chapman-Jouguet speed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bag: BagModel = BagModel(a_s=1.1, a_b=1, V_s=1)
        cls.const_cs: ConstCSModel = ConstCSModel(css2=1/3, csb2=0.3, a_s=1.5, a_b=1, V_s=1, log_info=False)
        cls.models: list[Model] = [cls.bag, cls.const_cs, *gksvdv_models()]

    def test_bag_limits(self) -> None:
        r"""$v_{CJ} \to c_s$ as $\alpha_+ \to 0$ and $v_{CJ} \to 1$ as $\alpha_+ \to \infty$."""
        self.assertAlmostEqual(v_chapman_jouguet_bag(0.), CS0)
        self.assertAlmostEqual(v_chapman_jouguet_bag(1e8), 1)

    def test_bag_junction(self) -> None:
        data = v_chapman_jouguet_bag(ALPHA_NS)
        ref = np.array([v_cj_junction(self.bag, alpha_n) for alpha_n in ALPHA_NS])
        assert_allclose(data, ref, rtol=1e-10)

    def test_analytical_junction(self) -> None:
        """The analytical Chapman-Jouguet speed should fulfill the junction conditions."""
        for model in self.models:
            alpha_ns = ALPHA_NS[model.alpha_n_min < ALPHA_NS]
            with self.subTest(model=model.label_unicode):
                data = v_chapman_jouguet(model, alpha_ns)
                ref = np.array([v_cj_junction(model, alpha_n) for alpha_n in alpha_ns])
                assert_allclose(data, ref, rtol=1e-10)

    def test_analytical_gksvdv(self) -> None:
        """The analytical Chapman-Jouguet speed should correspond to that of the Giese et al. code."""
        for model in gksvdv_models():
            alpha_ns = ALPHA_NS[model.alpha_n_min < ALPHA_NS]
            with self.subTest(model=model.label_unicode):
                data = v_chapman_jouguet(model, alpha_ns)
                alpha_theta_bar_ns = model.alpha_theta_bar_n_from_alpha_n(alpha_ns)
                ref = np.array([v_cj_gksvdv(a, model.csb2) for a in alpha_theta_bar_ns])
                assert_allclose(data, ref, rtol=1e-10)

    def test_numerical(self) -> None:
        """The numerical Chapman-Jouguet speed should correspond to the analytical one."""
        for model in self.models:
            alpha_ns = ALPHA_NS[model.alpha_n_min < ALPHA_NS]
            with self.subTest(model=model.label_unicode):
                data = v_chapman_jouguet(model, alpha_ns, analytical=False)
                ref = v_chapman_jouguet(model, alpha_ns)
                assert_allclose(data, ref, rtol=1e-10)

    def test_wm(self) -> None:
        r"""$\tilde{v}_- = c_{s,-}({w}_-)$ and the first junction condition should hold for ${w}_-$."""
        for model in self.models:
            alpha_ns = ALPHA_NS[model.alpha_n_min < ALPHA_NS]
            with self.subTest(model=model.label_unicode):
                for alpha_n in alpha_ns:
                    wp = model.wn(alpha_n)
                    wm = wm_chapman_jouguet(model, wp)
                    vp = v_chapman_jouguet(model, alpha_n)
                    vm = np.sqrt(model.cs2(wm, Phase.BROKEN))
                    assert_allclose(wm * gamma2(vm) * vm, wp * gamma2(vp) * vp, rtol=1e-10)

    def test_extra_output(self) -> None:
        """The analytical and numerical paths should give the same extra output."""
        for model in self.models:
            alpha_n = 0.2
            with self.subTest(model=model.label_unicode):
                data = v_chapman_jouguet(model, alpha_n, extra_output=True)
                ref = v_chapman_jouguet(model, alpha_n, extra_output=True, analytical=False)
                self.assertIsInstance(data, tuple)
                assert_allclose(np.array(data), np.array(ref), rtol=1e-10)

    def test_const_cs_reference(self) -> None:
        """Reference values from the analytical Chapman-Jouguet speed."""
        alpha_ns = np.array([0.1, 0.2, 0.3, 0.5])
        data = v_chapman_jouguet(self.const_cs, alpha_ns)
        ref = np.array([0.748372358460625, 0.805009807422845, 0.8385729412129682, 0.8784913826580829])
        assert_allclose(data, ref, rtol=1e-12)
