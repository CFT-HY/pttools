"""Validation tools"""

import logging

import numba
from numba.extending import overload
import numpy as np

from pttools.bubble import alpha
from pttools.speedup import njit
from pttools.speedup.differential import DifferentialPointer
import pttools.type_hints as th

logger = logging.getLogger(__name__)

type NucArgs = tuple[float, ...]
type PhysicalParams = tuple[float, float] | tuple[float, float, str, NucArgs]


@njit
def check_physical_params(params: PhysicalParams, df_dtau_ptr: DifferentialPointer) -> None:
    r"""
    Check that $v _\text{wall}$ = params[0], $\alpha_n$ = params[1] values are physical, i.e.
    $0 < v _\text{wall} < 1$,
    $\alpha_n < \alpha_{n,\max(v _\text{wall})}$
    """
    v_wall = params[0]
    alpha_n = params[1]
    check_wall_speed(v_wall)

    alpha_n_max = alpha.alpha_n_max_bag(v_wall, df_dtau_ptr=df_dtau_ptr)
    if alpha_n > alpha_n_max:
        with numba.objmode:
            logger.error(
                    "Unphysical parameter(s): v_wall = %s, alpha_n = %s. "
                    "Required alpha_n < %s",
                    v_wall, alpha_n, alpha_n_max
            )
        raise ValueError("Unphysical parameter(s). See the log for details.")


def _check_wall_speed_arr(v_wall: th.FloatOrArr, droplet: bool = False) -> None:
    if droplet:
        if np.logical_or(np.any(v_wall <= -1.), np.any(v_wall >= 0.)):
            raise ValueError(
                f"Unphysical v_wall for a droplet: min(v_wall)={np.min(v_wall)}, max(v_wall)={np.max(v_wall)}"
            )
    elif np.logical_or(np.any(v_wall >= 1.), np.any(v_wall <= 0.)):
        raise ValueError(
            f"Unphysical v_wall for a bubble: min(v_wall)={np.min(v_wall)}, max(v_wall)={np.max(v_wall)}"
        )


def _check_wall_speed_scalar(v_wall: th.FloatOrArr, droplet: bool = False) -> None:
    if droplet:
        if not -1. <= v_wall <= 0.:
            raise ValueError(f"v_wall={v_wall} is not physical for a droplet.")
    elif not 0.0 <= v_wall <= 1.0:
        raise ValueError(f"v_wall={v_wall} is not physical for a bubble.")


def check_wall_speed(v_wall: th.FloatOrArr, droplet: bool = False) -> None:
    r"""Check that $v _\text{wall}$ values are all physical: $(0 < v _\text{wall} < 1)$"""
    if isinstance(v_wall, float):
        return _check_wall_speed_scalar(v_wall, droplet)
    if isinstance(v_wall, np.ndarray):
        return _check_wall_speed_arr(v_wall, droplet)
    if isinstance(v_wall, list):
        return _check_wall_speed_arr(np.array(v_wall), droplet)
    raise TypeError(f"v_wall must be float, list or array. Got: {type(v_wall)}")


@overload(check_wall_speed, jit_options={"nopython": True})
def _check_wall_speed_numba(v_wall: th.FloatOrArr, droplet: bool = False) -> None:
    if isinstance(v_wall, numba.types.Float):
        return _check_wall_speed_scalar
    if isinstance(v_wall, numba.types.Array):
        if v_wall.ndim == 0:
            return _check_wall_speed_scalar
        return _check_wall_speed_arr
    raise TypeError(f"v_wall must be float, list or array. Got: {type(v_wall)}")


def find_most_negative_vals(vals: th.FloatOrArr, *args) -> list[float | None]:
    """Find the most negative values in the given array"""
    if vals is None or (not np.any(vals < 0)):
        return [None]*(len(args)+1)
    if np.isscalar(vals):
        return [vals, *args]

    i = np.argmin(vals)
    vals = [vals[i]]

    for arg in args:
        vals.append(arg if np.isscalar(arg) else arg[i])

    return vals
