"""Validation tools"""

import logging
import typing as tp

import numba
from numba.extending import overload
import numpy as np

import pttools.type_hints as th
from . import alpha

logger = logging.getLogger(__name__)

NucArgs = tp.Tuple[float, ...]
PhysicalParams = tp.Union[tp.Tuple[float, float], tp.Tuple[float, float, str, NucArgs]]


@numba.njit
def check_physical_params(params: PhysicalParams) -> None:
    r"""
    Checks that $v _\text{wall}$ = params[0], $\alpha_n$ = params[1] values are physical, i.e.
    $0 < v _\text{wall} < 1$,
    $\alpha_n < \alpha_{n,\max(v _\text{wall})}$
    """
    v_wall = params[0]
    alpha_n = params[1]
    check_wall_speed(v_wall)

    alpha_n_max = alpha.alpha_n_max_bag(v_wall)
    if alpha_n > alpha_n_max:
        with numba.objmode:
            logger.error((
                    "Unphysical parameter(s): v_wall = {}, alpha_n = {}. "
                    "Required alpha_n < {}").format(
                    v_wall, alpha_n, alpha_n_max
            ))
        raise ValueError("Unphysical parameter(s). See the log for details.")


def _check_wall_speed_arr(v_wall: tp.Union[th.FloatOrArr, tp.List[float]]):
    if np.logical_or(np.any(v_wall >= 1.0), np.any(v_wall <= 0.0)):
        raise ValueError("Unphysical parameter(s): at least one value outside 0 < v_wall < 1.")


def _check_wall_speed_scalar(v_wall: tp.Union[th.FloatOrArr, tp.List[float]]):
    if not 0.0 <= v_wall <= 1.0:
        with numba.objmode:
            logger.error("Unphysical parameter(s): v_wall = {}, required 0 < v_wall < 1.".format(v_wall))
        raise ValueError("Unphysical parameter: v_wall. See the log for details.")


def check_wall_speed(v_wall: tp.Union[th.FloatOrArr, tp.List[float]]):
    r"""
    Checks that $v _\text{wall}$ values are all physical: $(0 < v _\text{wall} < 1)$.
    """
    if isinstance(v_wall, float):
        return _check_wall_speed_scalar(v_wall)
    if isinstance(v_wall, np.ndarray):
        return _check_wall_speed_arr(v_wall)
    if isinstance(v_wall, list):
        return _check_wall_speed_arr(np.array(v_wall))
    raise TypeError(f"v_wall must be float, list or array. Got: {type(v_wall)}")


@overload(check_wall_speed, jit_options={"nopython": True})
def _check_wall_speed_numba(v_wall: tp.Union[th.FloatOrArr, tp.List[float]]):
    if isinstance(v_wall, numba.types.Float):
        return _check_wall_speed_scalar
    if isinstance(v_wall, numba.types.Array):
        if v_wall.ndim == 0:
            return _check_wall_speed_scalar
        return _check_wall_speed_arr
    raise TypeError(f"v_wall must be float, list or array. Got: {type(v_wall)}")


def find_most_negative_vals(vals: th.FloatOrArr, *args) \
        -> tp.List[tp.Optional[float]]:
    if vals is None or (not np.any(vals < 0)):
        return [None]*(len(args)+1)
    if np.isscalar(vals):
        return [vals, *args]

    i = np.argmin(vals)
    vals = [vals[i]]

    for arg in args:
        vals.append(arg if np.isscalar(arg) else arg[i])

    return vals
