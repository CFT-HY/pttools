"""Validation tools"""

import logging
import typing as tp

import numba
import numpy as np

import pttools.type_hints as th
from . import alpha

logger = logging.getLogger(__name__)

PHYSICAL_PARAMS_TYPE = tp.Union[tp.Tuple[float, float], tp.List[float]]


@numba.njit
def _check_wall_speed_arr(v_wall: np.ndarray):
    if np.logical_or(np.any(v_wall >= 1.0), np.any(v_wall <= 0.0)):
        raise ValueError("Unphysical parameter(s): at least one value outside 0 < v_wall < 1.")


@numba.njit
def _check_wall_speed_scalar(v_wall: float):
    if not 0.0 <= v_wall <= 1.0:
        with numba.objmode:
            logger.error("Unphysical parameter(s): v_wall = {}, required 0 < v_wall < 1.".format(v_wall))
        raise ValueError("Unphysical parameter: v_wall. See the log for details.")


@numba.generated_jit(nopython=True)
def check_wall_speed(v_wall: tp.Union[th.FLOAT_OR_ARR, tp.List[float]]):
    """
    Checks that v_wall values are all physical (0 < v_wall <1)
    """
    if isinstance(v_wall, numba.types.Float):
        return _check_wall_speed_scalar
    if isinstance(v_wall, numba.types.Array):
        if v_wall.ndim == 0:
            return _check_wall_speed_scalar
        return _check_wall_speed_arr
    if isinstance(v_wall, float):
        return _check_wall_speed_scalar(v_wall)
    elif isinstance(v_wall, np.ndarray):
        return _check_wall_speed_arr(v_wall)
    elif isinstance(v_wall, list):
        if any(vw >= 1.0 or vw <= 0.0 for vw in v_wall):
            raise ValueError("Unphysical parameter(s): at least one value outside 0 < v_wall < 1.")
    else:
        raise TypeError("v_wall must be float, list or array.")


# @numba.njit
def check_physical_params(params: PHYSICAL_PARAMS_TYPE) -> None:
    """
    Checks that v_wall = params[0], alpha_n = params[1] values are physical, i.e.
         0 < v_wall <1
         alpha_n < alpha_n_max(v_wall)
    """
    v_wall = params[0]
    alpha_n = params[1]
    check_wall_speed(v_wall)

    alpha_n_max = alpha.alpha_n_max(v_wall)
    if alpha_n > alpha_n_max:
        with numba.objmode:
            logger.error((
                    "Unphysical parameter(s): v_wall = {}, alpha_n = {}. "
                    "Required alpha_n < {}").format(
                    v_wall, alpha_n, alpha_n_max
            ))
        raise ValueError("Unphysical parameter(s). See the log for details.")
