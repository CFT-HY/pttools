r"""Unit tests for the automatic determination of the solution type.

With $c_{s,b} < c_{s,s}$, there are wall speeds $v_{CJ} < v_w < c_{s,s}$, at which a detonation is not possible,
since the fluid in front of the wall would be subsonic. The solution there can be a mixed hybrid,
which PTtools does not support, so PTtools must not return a detonation there.
"""

import unittest

import pytest

from pttools.bubble.bubble import Bubble
from pttools.bubble.solution_type import SolutionType, is_subsonic_in_front
from pttools.models.bag import BagModel
from tests.bubble.test_entropy_production import template_model


class SolutionTypeTest(unittest.TestCase):
    r"""Test the solution type in the window $v_{CJ} < v_w < c_{s,s}$."""

    #: $c_{s,s} = 0.7$, $c_{s,b} = 0.4$, $\alpha_{\bar{\theta}_n} = 0.05$, with $w_n = 1$
    MODEL = template_model(0.05, css2=0.70**2, csb2=0.40**2)
    #: $\alpha_n$ of the model
    ALPHA_N = float(MODEL.alpha_n(1.))

    def test_detonation_above_cs(self) -> None:
        """Above the speed of sound of the symmetric phase, the solution is a detonation."""
        assert not is_subsonic_in_front(self.MODEL, 0.8, 1.)
        assert self.MODEL.solution_type(0.8, self.ALPHA_N, wn=1.) == SolutionType.DETON

    def test_no_detonation_below_cs(self) -> None:
        """Between the Chapman-Jouguet speed and the speed of sound of the symmetric phase, there is no detonation."""
        for v_wall in (0.55, 0.65):
            with self.subTest(v_wall=v_wall):
                assert is_subsonic_in_front(self.MODEL, v_wall, 1.)
                assert self.MODEL.solution_type(v_wall, self.ALPHA_N, wn=1.) == SolutionType.ERROR

    def test_bubble_raises_below_cs(self) -> None:
        """Solving a bubble in the window should raise an error, also if a detonation is requested."""
        for sol_type in (None, SolutionType.DETON):
            with self.subTest(sol_type=sol_type), pytest.raises(ValueError, match=r"not possible|Could not determine"):
                Bubble(self.MODEL, 0.65, self.ALPHA_N, sol_type=sol_type, allow_invalid=True, log_invalid=False)

    def test_bag_model_unchanged(self) -> None:
        r"""In the bag model, $v_{CJ} > c_s$, so the detonations are not affected."""
        model = BagModel(a_s=1.1, a_b=1., V_s=1.)
        wn = float(model.wn(0.1))
        assert not is_subsonic_in_front(model, 0.8, wn)
        assert model.solution_type(0.8, 0.1, wn=wn) == SolutionType.DETON


if __name__ == "__main__":
    unittest.main()
