"""Validation tools"""

import typing as tp

import numpy as np

import pttools.type_hints as th
from . import alpha

PHYSICAL_PARAMS_TYPE = tp.Union[tp.Tuple[float, float], tp.List[float]]


def check_wall_speed(v_wall: tp.Union[th.FLOAT_OR_ARR, tp.List[float]]) -> None:
    """
    Checks that v_wall values are all physical (0 < v_wall <1)
    """
    if isinstance(v_wall, float):
        if not 0.0 <= v_wall <= 1.0:
            raise ValueError(f"Unphysical parameter(s): v_wall = {v_wall}, required 0 < v_wall < 1.")
    elif isinstance(v_wall, np.ndarray):
        if np.logical_or(np.any(v_wall >= 1.0), np.any(v_wall <= 0.0)):
            raise ValueError("Unphysical parameter(s): at least one value outside 0 < v_wall < 1.")
    elif isinstance(v_wall, list):
        if any(vw >= 1.0 or vw <= 0.0 for vw in v_wall):
            raise ValueError("Unphysical parameter(s): at least one value outside 0 < v_wall < 1.")
    else:
        raise TypeError("v_wall must be float, list or array.")


def check_physical_params(params: PHYSICAL_PARAMS_TYPE) -> None:
    """
    Checks that v_wall = params[0], alpha_n = params[1] values are physical, i.e.
         0 < v_wall <1
         alpha_n < alpha_n_max(v_wall)
    """
    v_wall = params[0]
    alpha_n = params[1]
    check_wall_speed(v_wall)

    if alpha_n > alpha.alpha_n_max(v_wall):
        raise ValueError(
            f"Unphysical parameter(s): v_wall = {v_wall}, alpha_n = {alpha_n}. "
            f"Required alpha_n < {alpha.alpha_n_max(v_wall)}")
