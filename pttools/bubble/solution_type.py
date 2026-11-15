"""Solution types of relativistic combustion."""

from __future__ import annotations

import enum
import logging
import typing as tp

from pttools.bubble.phase import Phase

if tp.TYPE_CHECKING:
    from pttools.models.model import Model

__all__ = [
    "SolutionType",
    "cannot_be_detonation",
    "cannot_be_sub_def",
    "is_subsonic_in_front",
    "is_surely_detonation",
    "is_surely_sub_def",
    "validate_solution_type",
]

logger: logging.Logger = logging.getLogger(__name__)


@enum.unique
class SolutionType(enum.StrEnum):
    r"""There are three different types of relativistic combustion.

    For further details, please see chapter 7.2 and figure 14
    of :notes:`\ `.

    .. plot:: fig/solution_types.py
    .. minigallery:: ../../examples/props/solution_types.py
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
    r"""If $v_w < v_{CJ}$, it cannot be a detonation."""
    return v_wall < v_cj


def cannot_be_sub_def(model: Model, v_wall: float, wn: float) -> bool:
    r"""Check whether the solution certainly cannot be a subsonic deflagration.

    If the wall speed $v_w > c_{sb}(w) \forall w \in [0, w_n]$,
    then the wall is certainly hypersonic in the broken phase and must have fluid movement inside the wall
    to satisfy the boundary conditions. Therefore, the solution cannot be a subsonic deflagration.
    """
    cs2_max, _w_max = model.cs2_max(wn, Phase.BROKEN)
    return v_wall**2 > cs2_max


def is_subsonic_in_front(model: Model, v_wall: float, wn: float) -> bool:
    r"""Check whether the wall is subsonic with respect to the fluid at rest in front of it, $v_w < c_{s,s}(w_n)$.

    In a detonation, the fluid in front of the wall is at rest in the state of nucleation,
    so $\tilde{v}_{+} = v_w$, which must be at least the speed of sound $c_{s,s}(w_n)$ there.
    Therefore, if this is true, the solution cannot be a detonation, even if $v_w > v_{CJ}$.
    With $c_{s,b} < c_{s,s}$, e.g. in the constant sound speed model, the solution can then be a mixed hybrid
    with rarefaction waves on both sides of the wall (Barni and Madge, in preparation),
    which PTtools does not support.
    In the bag model $v_{CJ} > c_s$, and this is always false for $v_w > v_{CJ}$.
    """
    return v_wall**2 < model.cs2(wn, Phase.SYMMETRIC)


def is_surely_detonation(v_wall: float, v_cj: float) -> float:
    r"""If $v_w > v_{CJ}$, it is certainly a detonation."""
    return v_wall > v_cj


def is_surely_sub_def(model: Model, v_wall: float, wn: float) -> bool:
    r"""Check whether the solution is certainly a subsonic deflagration.

    If the wall speed $v_w < c_{sb}(w) \forall w \in [0, w_n]$,
    then the wall is certainly subsonic in the broken phase,
    and therefore the solution is certainly a subsonic deflagration.
    """
    cs2_min, _w_min = model.cs2_min(wn, Phase.BROKEN)
    return v_wall**2 < cs2_min


def validate_solution_type(
        model: Model,
        v_wall: float,
        alpha_n: float,
        sol_type: SolutionType | None,
        wn: float | None = None,
        wn_guess: float | None = None,
        wm_guess: float | None = None) -> SolutionType:
    """Ensure that the solution type is determined or can be determined automatically."""
    if sol_type is None or sol_type is SolutionType.UNKNOWN:
        sol_type = model.solution_type(
            v_wall=v_wall, alpha_n=alpha_n, wn=wn, wn_guess=wn_guess, wm_guess=wm_guess
        )
    if sol_type == SolutionType.DETON:
        if wn is None:
            wn = model.wn(alpha_n, wn_guess)
        if is_subsonic_in_front(model, v_wall, wn):
            msg = \
                f"A detonation is not possible for model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}, " \
                f"since v_wall is below the speed of sound c_s,s(w_n)={float(model.cs2(wn, Phase.SYMMETRIC))**0.5}."
            logger.error(msg)
            raise ValueError(msg)
    if sol_type in [SolutionType.UNKNOWN, SolutionType.ERROR]:
        msg = \
            "Could not determine solution type automatically for " \
            f"model={model.name}, v_wall={v_wall}, alpha_n={alpha_n}. " \
            f"Got sol_type={sol_type}. Please choose it manually."
        logger.error(msg)
        raise ValueError(msg)
    return sol_type
