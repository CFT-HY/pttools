"""Utility functions for trimming fluid solutions"""

import logging
import typing as tp

import numba
import numpy as np

import pttools.type_hints as th
from . import bag
from .boundary import Phase, SolutionType
from . import check
from . import const
from . import shock

logger = logging.getLogger(__name__)


@numba.njit
def trim_fluid_wall_to_cs(
        v: np.ndarray,
        w: np.ndarray,
        xi: np.ndarray,
        t: np.ndarray,
        v_wall: th.FloatOrArr,
        sol_type: SolutionType,
        dxi_lim: float = const.DXI_SMALL,
        cs2_fun: th.CS2Fun = bag.cs2_bag) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Picks out fluid variable arrays $(v, w, \xi, t)$ which are definitely behind
    the wall for detonation and hybrid.
    Also removes negative fluid speeds and $\xi \leq c_s$, which might be left by
    an inaccurate integration.
    If the wall is within about 1e-16 of cs, rounding errors are flagged.

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param t: $t$
    :param v_wall: $v_\text{wall}$
    :param sol_type: solution type
    :param dxi_lim: not used
    :param cs2_fun: function, which gives $c_s^2$
    :return: trimmed $v, w, \xi, t$
    """
    check.check_wall_speed(v_wall)
    n_start = 0

    # TODO: should this be 0 to match with the error handling below?
    n_stop_index = -2
    # n_stop = 0
    if sol_type != SolutionType.SUB_DEF.value:
        for i in range(v.size):
            if v[i] <= 0 or xi[i] ** 2 <= cs2_fun(w[i], Phase.BROKEN.value):
                n_stop_index = i
                break

    if n_stop_index == 0:
        with numba.objmode:
            logger.warning((
                "Integation gave v < 0 or xi <= cs. "
                "sol_type: {}, v_wall: {}, xi[0] = {}, v[0] = {}. "
                "Fluid profile has only one element between vw and cs. "
                "Fix implemented by adding one extra point.").format(sol_type, v_wall, xi[0], v[0]))
        n_stop = 1
    else:
        n_stop = n_stop_index

    if (xi[0] == v_wall) and not (sol_type == SolutionType.DETON.value):
        n_start = 1
        n_stop += 1

    return v[n_start:n_stop], w[n_start:n_stop], xi[n_start:n_stop], t[n_start:n_stop]


@numba.njit
def trim_fluid_wall_to_shock(
        v: np.ndarray,
        w: np.ndarray,
        xi: np.ndarray,
        t: np.ndarray,
        sol_type: SolutionType) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Trims fluid variable arrays $(v, w, \xi)$ so last element is just ahead of shock.

    :param v: $v$
    :param w: $w$
    :param xi: $\xi$
    :param t: $t$
    :param sol_type: solution type
    :return: trimmed $v, w, \xi, t$
    """
    # TODO: should this be 0 to match with the error handling below?
    n_shock_index = -2
    # n_shock = 0
    if sol_type != SolutionType.DETON.value:
        for i in range(v.size):
            if v[i] <= shock.v_shock_bag(xi[i]):
                n_shock_index = i
                break

    if n_shock_index == 0:
        with numba.objmode:
            # F-strings are not yet supported by Numba, even in object mode.
            # https://github.com/numba/numba/issues/3250
            logger.warning((
                "v[0] < v_shock(xi[0]). "
                "sol_type: {}, xi[0] = {}, v[0] = {}, v_sh(xi[0]) = {}. "
                "Shock profile has only one element. Fix implemented by adding one extra point.").format(
                sol_type, xi[0], v[0], shock.v_shock_bag(xi[0])
            ))
        n_shock = 1
    else:
        n_shock = n_shock_index

    return v[:n_shock + 1], w[:n_shock + 1], xi[:n_shock + 1], t[:n_shock + 1]
