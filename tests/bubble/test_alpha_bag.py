r"""Unit tests for the conversion between $\alpha_+$ and $\alpha_n$ in the bag model."""

import pytest

from pttools.bubble import (
    CS2_BAG_SCALAR_PTR,
    DEFAULT_FLUID_INTEGRATE_METHOD,
    DF_DTAU_PTR_BAG,
    Bubble,
    find_alpha_n_bag,
    find_alpha_plus_bag,
)
from pttools.bubble.solution_type import SolutionType
from pttools.models.bag import BagModel

# v_wall, alpha_+, alpha_n, solution type
# The deflagration and hybrid values of alpha_n are the existing output of find_alpha_n_bag.
# The first three cases are those of Hindmarsh & Hijazi 2019, fig. 10 (see tests.bubble.ref.RefHindmarshHijazi),
# where the corresponding values of alpha_n are 0.578, 0.151 and 0.091.
# In detonations the fluid ahead of the wall is at rest, and therefore w_+ = w_n and alpha_n = alpha_+.
CASES = [
    (0.5, 0.263, 0.577904045883447, SolutionType.SUB_DEF),
    (0.7, 0.052, 0.15048375269390826, SolutionType.HYBRID),
    (0.77, 0.091, 0.091, SolutionType.DETON),
    (0.8, 0.1, 0.1, SolutionType.DETON),
    (0.9, 0.05, 0.05, SolutionType.DETON),
]


def alpha_n_bag(v_wall: float, alpha_plus: float) -> float:
    r"""$\alpha_n$ from $\alpha_+$ with the bag model solver."""
    return find_alpha_n_bag(
        v_wall, alpha_plus,
        df_dtau_ptr=DF_DTAU_PTR_BAG, ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_ptr=CS2_BAG_SCALAR_PTR
    )


@pytest.mark.parametrize(("v_wall", "alpha_plus", "alpha_n", "sol_type"), CASES)
def test_find_alpha_n_bag(v_wall: float, alpha_plus: float, alpha_n: float, sol_type: SolutionType) -> None:
    r"""Test $\alpha_n$ against the reference values."""
    assert alpha_n_bag(v_wall, alpha_plus) == pytest.approx(alpha_n, rel=1e-7)


@pytest.mark.parametrize(("v_wall", "alpha_plus", "alpha_n", "sol_type"), CASES)
def test_find_alpha_n_bag_bubble(v_wall: float, alpha_plus: float, alpha_n: float, sol_type: SolutionType) -> None:
    r"""Test that the general solver gives the same $\alpha_+$ for the $\alpha_n$ of the bag model solver."""
    model = BagModel(a_s=1.1, a_b=1, V_s=1, alpha_n_min=0.01)
    bubble = Bubble(model, v_wall=v_wall, alpha_n=alpha_n_bag(v_wall, alpha_plus), sol_type=sol_type)
    assert bubble.alpha_plus == pytest.approx(alpha_plus, rel=1e-3)


@pytest.mark.parametrize(("v_wall", "alpha_plus", "alpha_n", "sol_type"), CASES)
def test_find_alpha_n_bag_round_trip(v_wall: float, alpha_plus: float, alpha_n: float, sol_type: SolutionType) -> None:
    r"""Test that :func:`find_alpha_plus_bag` is the inverse of :func:`find_alpha_n_bag`."""
    alpha_plus_found = find_alpha_plus_bag(
        v_wall, alpha_n_bag(v_wall, alpha_plus),
        df_dtau_ptr=DF_DTAU_PTR_BAG, ode_method=DEFAULT_FLUID_INTEGRATE_METHOD, cs2_ptr=CS2_BAG_SCALAR_PTR
    )
    assert alpha_plus_found == pytest.approx(alpha_plus, rel=1e-4)
