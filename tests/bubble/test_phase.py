"""Unit tests for the phase of the points of a bubble profile."""

import numpy as np
import pytest

from pttools.bubble import Bubble
from pttools.bubble.phase import Phase
from pttools.bubble.props import find_phase
from pttools.bubble.solution_type import SolutionType
from pttools.models.bag import BagModel
from pttools.models.const_cs import ConstCSModel
from pttools.models.model import Model

BAG: BagModel = BagModel(a_s=1.1, a_b=1, V_s=1)
CONST_CS: ConstCSModel = ConstCSModel(css2=1/3 - 0.01, csb2=1/3 - 0.011, a_s=1.5, a_b=1, V_s=1)


@pytest.mark.parametrize(("model", "v_wall", "alpha_n", "sol_type"), [
    (BAG, 0.4, 0.1, SolutionType.SUB_DEF),
    (BAG, 0.65, 0.1, SolutionType.HYBRID),
    (BAG, 0.8, 0.1, SolutionType.DETON),
    (CONST_CS, 0.4, 0.2, SolutionType.SUB_DEF),
    (CONST_CS, 0.65, 0.2, SolutionType.HYBRID),
    # Parameters of a bug report, where the temperature had a dip at the wall
    (BagModel(a_s=11.582548/0.84, a_b=11.582548, alpha_n_min=0.01), 0.6428908756714361, 0.068, SolutionType.HYBRID),
])
def test_phase_at_wall(model: Model, v_wall: float, alpha_n: float, sol_type: SolutionType) -> None:
    """Test that the phase changes from broken to symmetric exactly at the discontinuity of $w$ at the wall."""
    bubble = Bubble(model, v_wall=v_wall, alpha_n=alpha_n, sol_type=sol_type)
    assert bubble.sol_type == sol_type
    phase = bubble.phase

    # The phase is broken up to some index and symmetric after it.
    i_last_broken = np.flatnonzero(phase == Phase.BROKEN)[-1]
    assert np.all(phase[:i_last_broken + 1] == Phase.BROKEN)
    assert np.all(phase[i_last_broken + 1:] == Phase.SYMMETRIC)

    # The jump of w at the wall is between the last broken point and the first symmetric point.
    near_wall = np.flatnonzero(np.abs(bubble.xi - v_wall) < 1e-3)
    w_diff = np.abs(np.diff(bubble.w[near_wall[0]:near_wall[-1] + 2]))
    assert near_wall[0] + np.argmax(w_diff) == i_last_broken

    # The temperature is continuous within the broken phase, i.e. there is no dip at the wall.
    temp = model.temp(bubble.w, phase)
    assert temp[i_last_broken] == pytest.approx(temp[i_last_broken - 1], rel=1e-2)


def test_find_phase_hybrid() -> None:
    r"""Test that of the two points at $\xi = {v}_\text{wall}$ in a hybrid the first one is in the broken phase."""
    xi = np.array([0, 0.5, 0.6, 0.6, 0.7, 1])
    np.testing.assert_array_equal(find_phase(xi, 0.6), [1, 1, 1, 0, 0, 0])
    np.testing.assert_array_equal(find_phase(xi, 0.6, SolutionType.HYBRID), [1, 1, 1, 0, 0, 0])


def test_find_phase_single_wall_point() -> None:
    r"""Test the phase of a single point at $\xi = {v}_\text{wall}$ in detonations and deflagrations."""
    xi = np.array([0, 0.5, 0.6, 0.7, 1])
    np.testing.assert_array_equal(find_phase(xi, 0.6, SolutionType.DETON), [1, 1, 1, 0, 0])
    np.testing.assert_array_equal(find_phase(xi, 0.6, SolutionType.SUB_DEF), [1, 1, 0, 0, 0])
    np.testing.assert_array_equal(find_phase(xi, 0.6), [1, 1, 0, 0, 0])


def test_find_phase_no_wall_point() -> None:
    r"""Test the phase when there is no point at $\xi = {v}_\text{wall}$."""
    xi = np.array([0, 0.5, 0.7, 1])
    np.testing.assert_array_equal(find_phase(xi, 0.6), [1, 1, 0, 0])
    np.testing.assert_array_equal(find_phase(xi, 2), [1, 1, 1, 1])
    np.testing.assert_array_equal(find_phase(xi, -1), [0, 0, 0, 0])
