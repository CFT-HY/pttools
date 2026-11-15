r"""Unit tests for the entropy production of the bubbles.

The entropy of a self-similar bubble is produced only at the discontinuities.
Across a discontinuity at $\xi_d$, the entropy current $s u^\mu$ gives the entropy that crosses a unit area of the
discontinuity per unit time as $s \gamma (\xi_d - v)$, so the total entropy production per $t^3$ is
$$\Delta S = \frac{4\pi}{3} \sum_d \xi_d^2 \left[ s \gamma (\xi_d - v) \right]_{+}^{-},$$
where the bracket is the value behind minus the value in front of the discontinuity.
"""

import typing as tp
import unittest

import numpy as np

from pttools.bubble.bubble import Bubble
from pttools.bubble.phase import Phase
from pttools.bubble.relativity import gamma2
from pttools.bubble.solution_type import SolutionType
from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel
from pttools.models.model import Model


def template_model(alpha_theta_bar_n: float, css2: float, csb2: float, psi_n: float = 1.) -> ConstCSModel:
    r"""Constant sound speed model with $w_n = T_n = 1$, parametrised as the template model of Ai et al. (2023).

    :param alpha_theta_bar_n: $\alpha_{\bar{\theta}_n}$
    :param css2: $c_{s,s}^2$
    :param csb2: $c_{s,b}^2$
    :param psi_n: $\Psi_n$
    :return: the model
    """
    mu, nu = 1 + 1/css2, 1 + 1/csb2
    a_s = 1/mu
    return ConstCSModel(
        css2=css2, csb2=csb2, a_s=a_s, a_b=psi_n * mu * a_s / nu,
        V_s=3 * (alpha_theta_bar_n - (mu - nu) / (3*mu)) / nu, V_b=0,
        allow_invalid=True, log_info=False,
    )


def entropy_production_from_jumps(bubble: Bubble) -> float:
    r"""$\Delta S$ from the entropy fluxes across the wall and the shock, see the module docstring."""
    def flux(s: float, v: float, xi: float) -> float:
        return float(s * np.sqrt(gamma2(v)) * (xi - v))

    total = bubble.v_wall**2 * (
        flux(bubble.sm, bubble.vm, bubble.v_wall) - flux(bubble.sp, bubble.vp, bubble.v_wall))
    if bubble.sol_type != SolutionType.DETON:
        sm_sh = float(bubble.model.s(bubble.wm_sh, Phase.SYMMETRIC))
        total += bubble.v_sh**2 * (flux(sm_sh, bubble.vm_sh, bubble.v_sh) - flux(bubble.sn, 0., bubble.v_sh))
    return 4*np.pi/3 * total


class EntropyProductionTest(unittest.TestCase):
    """Test the entropy production against the entropy fluxes at the discontinuities."""

    #: Models, $\alpha_{\bar{\theta}_n}$ and wall speeds of the tested bubbles
    CASES: tuple[tuple[Model, float, tuple[float, ...]], ...] = (
        (BagModel(a_s=1.1, a_b=1., V_s=1.), 0.1, (0.4, 0.6, 0.8)),
        (template_model(0.5, css2=0.75**2, csb2=0.40**2), 0.5, (0.3, 0.6, 0.9)),
    )

    bubbles: tp.ClassVar[list[Bubble]]

    @classmethod
    def setUpClass(cls) -> None:
        """Solve the bubbles."""
        cls.bubbles = [
            Bubble(model, v_wall, alpha, theta_bar=True, allow_invalid=True, log_invalid=False)
            for model, alpha, v_walls in cls.CASES for v_wall in v_walls
        ]

    def test_matches_jumps(self) -> None:
        r"""The integrated entropy production should equal the sum of the jumps of the entropy flux.

        The tolerance is that of the integration over the profile on the default grid.
        """
        for bubble in self.bubbles:
            with self.subTest(bubble=bubble.label_unicode):
                np.testing.assert_allclose(
                    bubble.va_entropy_production, entropy_production_from_jumps(bubble), rtol=3e-2)

    def test_positive(self) -> None:
        """The entropy production should be positive, and the bubble should not be flagged."""
        for bubble in self.bubbles:
            with self.subTest(bubble=bubble.label_unicode):
                assert bubble.va_entropy_production > 0
                assert bubble.entropy_production > 0
                assert bubble.va_entropy_production_relative > 0
                assert bubble.entropy_production_relative > 0
                assert not bubble.negative_net_entropy_change

    def test_constant_cs_deflagration_not_flagged(self) -> None:
        r"""A deflagration whose entropy flux increases at the wall should not be flagged.

        For this deflagration with $c_{s,s} > c_{s,b}$, the integral of $s - s_n$ without $\gamma$ is negative,
        so it was flagged with ``negative_net_entropy_change`` when the check used that integral
        instead of the entropy production.
        """
        bubble = Bubble(
            template_model(0.5, css2=0.75**2, csb2=0.40**2), 0.3, 0.5,
            theta_bar=True, allow_invalid=True, log_invalid=False)
        assert bubble.entropy_flux_diff > 0
        assert bubble.va_entropy_production > 0
        assert not bubble.negative_net_entropy_change


if __name__ == "__main__":
    unittest.main()
