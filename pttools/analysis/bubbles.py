import typing as tp

import numpy as np

from pttools.bubble import BaseBubble, Bubble, Phase, SolutionType


def cs2_common(
        bubbles: tp.Iterable[BaseBubble],
        phase: Phase,
        w: float | None = None,
        cs2_default: float | None = None) -> float | None:
    """Get $c_s^2$ that is common for the given bubbles

    :param bubbles: bubbles
    :param phase: phase in which to evaluate $c_s^2$
    :param w: enthalpy $w$ in which to evaluate $c_s^2$
    :param cs2_default: default value if there are no bubbles, or if the values are different
    :return: $c_s^2$ that is common for the given bubbles
    """
    cs2 = None
    for bubble in bubbles:
        cs2_new = bubble.model.cs2(w=w, phase=phase)
        if cs2 is None:
            cs2 = cs2_new
        elif not np.isclose(cs2_new, cs2):
            return cs2_default
    # If there were no bubbles, return None
    return cs2_default if cs2 is None else cs2


# def v_max_behind_common(bubbles: tp.Iterable[BaseBubble]):
#     """Maximum fluid velocity $\mu$ behind the wall, common for the given bubbles"""
#     css2 = None
#     for bubble in bubbles:
#         if isinstance(bubble, Bubble) and bubble.sol_type == SolutionType.DETON:
#             css2 = bubble.model.cs2(w=bubble.wm, phase=Phase.BROKEN)
