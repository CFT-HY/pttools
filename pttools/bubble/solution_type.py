"""Solution types of relativistic combustion"""

import enum
import logging
import typing as tp

from pttools.bubble.phase import Phase

if tp.TYPE_CHECKING:
    from pttools.models.model import Model

logger = logging.getLogger(__name__)


@enum.unique
class SolutionType(enum.StrEnum):
    r"""There are three different types of relativistic combustion.
    For further details, please see chapter 7.2 and figure 14
    of :notes:`\ `.

    .. plot:: fig/relativistic_combustion.py
    """
    # Todo: Should the strong and weak branches of the solutions (vplus, vminus signs) be distinquished here?

    #: In a detonation the fluid outside the bubble is at rest and the wall moves at a supersonic speed.
    DETON = "Detonation"

    #: Droplets are contracting solutions where the wall speed is negative.
    DROPLET = "Droplet"

    #: This value is used to inform, that determining the type of the
    #: relativistic combustion failed.
    ERROR = "Error"

    #: In the hybrid case the wall speed is supersonic and the fluid is moving both ahead and behind the wall.
    HYBRID = "Hybrid"

    #: In a subsonic deflagration the fluid is at rest inside the bubble,
    #: and the wall moves at a subsonic speed.
    SUB_DEF = "Subsonic deflagration"

    #: This value is used, when the type of the relativistic combustion is not yet determined.
    UNKNOWN = "Unknown"


def cannot_be_detonation(v_wall: float, v_cj: float) -> float:
    r"""If $v_w < v_{CJ}$, it cannot be a detonation"""
    return v_wall < v_cj


def cannot_be_sub_def(model: "Model", v_wall: float, wn: float) -> bool:
    r"""If the wall speed $v_w > c_{sb}(w) \forall w \in [0, w_n]$,
    then the wall is certainly hypersonic in the broken phase and must have fluid movement inside the wall
    to satisfy the boundary conditions. Therefore, the solution cannot be a subsonic deflagration."""
    cs2_max, w_max = model.cs2_max(wn, Phase.BROKEN)
    return v_wall**2 > cs2_max


def is_surely_detonation(v_wall: float, v_cj: float) -> float:
    r"""If $v_w > v_{CJ}$, it is certainly a detonation"""
    return v_wall > v_cj


def is_surely_sub_def(model: "Model", v_wall: float, wn: float) -> bool:
    r"""If the wall speed $v_w < c_{sb}(w) \forall w \in [0, w_n]$,
    then the wall is certainly subsonic in the broken phase,
    and therefore the solution is certainly a subsonic deflagration."""
    cs2_min, w_min = model.cs2_min(wn, Phase.BROKEN)
    return v_wall**2 < cs2_min


def validate_solution_type(
        model: "Model",
        v_wall: float,
        alpha_n: float,
        sol_type: SolutionType,
        wn: float | None = None,
        wn_guess: float | None = None,
        wm_guess: float | None = None) -> SolutionType:
    """Ensure that the solution type is determined or can be determined automatically"""
    if sol_type is None or sol_type is SolutionType.UNKNOWN:
        sol_type = model.solution_type(
            v_wall=v_wall, alpha_n=alpha_n, wn=wn, wn_guess=wn_guess, wm_guess=wm_guess
        )
    if sol_type in [SolutionType.UNKNOWN, SolutionType.ERROR]:
        msg = \
            "Could not determine solution type automatically for " \
            f"model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. " \
            f"Got sol_type={sol_type}. Please choose it manually."
        logger.error(msg)
        raise ValueError(msg)
    return sol_type
