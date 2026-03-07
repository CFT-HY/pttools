r"""$\alpha_n$ limits for the Bag Model"""

import numba
from numba.extending import overload
import numpy as np

from pttools import speedup
from pttools.bubble import const
from pttools.bubble import fluid_bag
from pttools.bubble import check
from pttools.bubble import props
from pttools.bubble.solution_type import SolutionType
from pttools.bubble.solution_type_bag import identify_solution_type_alpha_plus_bag
from pttools.speedup import NUMBA_ENABLE_CACHE
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr


@numba.njit(nogil=True, cache=NUMBA_ENABLE_CACHE)
def alpha_n_max_bag(v_wall: th.FloatOrArr, n_xi: int = const.N_XI_DEFAULT) -> th.FloatOrArr:
    r"""
    Calculates the maximum relative trace anomaly outside the bubble, $\alpha_{n,\max}$,
    in the Bag Model for given $v_\text{wall}$, which is max $\alpha_n$ for (supersonic) deflagration.

    :param v_wall: $v_\text{wall}$
    :param n_xi: number of $\xi$ points
    :return: $\alpha_{n,\max}$, the relative trace anomaly outside the bubble
    """
    return alpha_n_max_deflagration_bag(v_wall, n_xi)


def _alpha_n_max_deflagration_bag_scalar(
        v_wall: th.FloatOrArr,
        n_xi: int = const.N_XI_DEFAULT,
        parallel: bool = True) -> th.FloatOrArr:
    check.check_wall_speed(v_wall)
    if v_wall > 0.9999:
        # Alpha_n_max diverges as v_wall -> 1, and the solver fails to find the correct solution.
        return np.nan
    sol_type = SolutionType.HYBRID.value if v_wall > const.CS0 else SolutionType.SUB_DEF.value
    ap = 1. / 3 - 1.0e-10  # Warning - this is not safe. Causes warnings for low v_wall.
    _, w, xi = fluid_bag.sound_shell_alpha_plus_bag(v_wall, ap, sol_type, n_xi)
    n_wall = props.find_v_index(xi, v_wall)
    return w[n_wall + 1] * (1. / 3)


_alpha_n_max_deflagration_bag_scalar_numba = numba.njit(_alpha_n_max_deflagration_bag_scalar)


def _alpha_n_max_deflagration_bag_arr(v_wall: th.FloatOrArr, n_xi: int = const.N_XI_DEFAULT) -> th.FloatOrArr:
    ret = np.zeros_like(v_wall)
    for i in numba.prange(v_wall.size):  # pylint: disable=not-an-iterable
        ret[i] = _alpha_n_max_deflagration_bag_scalar_numba(v_wall[i], n_xi)
    # alpha_N = (w_+/w_N)*alpha_+
    # w_ is normalized to 1 at large xi
    # Need n_wall+1, as w is an integral of v, and lags by 1 step
    return ret

_alpha_n_max_deflagration_bag_arr_parallel = numba.njit(parallel=True, nogil=True)(_alpha_n_max_deflagration_bag_arr)
_alpha_n_max_deflagration_bag_arr_single = numba.njit(nogil=True)(_alpha_n_max_deflagration_bag_arr)


def _alpha_n_max_deflagration_bag_arr_wrapper(
        v_wall: th.FloatOrArr,
        n_xi: int = const.N_XI_DEFAULT,
        parallel: bool = True) -> th.FloatOrArr:
    if parallel:
        return _alpha_n_max_deflagration_bag_arr_parallel(v_wall=v_wall, n_xi=n_xi)
    return _alpha_n_max_deflagration_bag_arr_single(v_wall=v_wall, n_xi=n_xi)


def alpha_n_max_deflagration_bag(
        v_wall: th.FloatOrArr,
        n_xi: int = const.N_XI_DEFAULT,
        parallel: bool = True) -> th.FloatOrArr:
    r"""
    Calculates the maximum phase transition strength $\alpha_{n,\max}$,
    in the Bag Model for given $v_\text{wall}$, for deflagration.
    Works also for hybrids, as they are supersonic deflagrations.

    :param v_wall: $v_\text{wall}$
    :param n_xi: number of $\xi$ points
    :return: $\alpha_{n,\max}$
    """
    if isinstance(v_wall, float):
        return _alpha_n_max_deflagration_bag_scalar(v_wall=v_wall, n_xi=n_xi)
    if isinstance(v_wall, np.ndarray):
        if not v_wall.ndim:
            return _alpha_n_max_deflagration_bag_scalar(v_wall=v_wall.item(), n_xi=n_xi)
        return _alpha_n_max_deflagration_bag_arr(v_wall=v_wall, n_xi=n_xi)
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


@overload(alpha_n_max_deflagration_bag, jit_options={"nopython": True, "nogil": True, "cache": NUMBA_ENABLE_CACHE})
def _alpha_n_max_deflagration_bag_numba(
        v_wall: th.FloatOrArr,
        n_xi: int = const.N_XI_DEFAULT,
        parallel: bool = True) -> th.FloatOrArr:
    if isinstance(v_wall, numba.types.Float):
        return _alpha_n_max_deflagration_bag_scalar
    if isinstance(v_wall, numba.types.Array):
        if not v_wall.ndim:
            return _alpha_n_max_deflagration_bag_scalar
        return _alpha_n_max_deflagration_bag_arr_wrapper
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


@numba.njit
def alpha_n_max_detonation_bag(v_wall: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Maximum allowed value of $\alpha_n$ for a detonation with wall speed $v_\text{wall}$ in the Bag Model.
    Same as :func:`alpha_plus_max_detonation`, since for a detonation $\alpha_n = \alpha_+$,
    as there is no fluid movement outside the wall.

    :param v_wall: $v_\text{wall}$
    :return: $\alpha_{n,\max,\text{detonation}}$
    """
    return alpha_plus_max_detonation_bag(v_wall)


@numba.njit
def alpha_n_max_hybrid_bag(v_wall: float, n_xi: int = const.N_XI_DEFAULT) -> float:
    r"""
    Calculates the relative trace anomaly outside the bubble, $\alpha_{n,\max}$,
    in the Bag Model for given $v_\text{wall}$, assuming hybrid fluid shell

    :param v_wall: $v_\text{wall}$
    :param n_xi: number of $\xi$ points
    :return: $\alpha_{n,\max}$
    """
    sol_type = identify_solution_type_alpha_plus_bag(v_wall=v_wall, alpha_p=1 / 3).value
    if sol_type == SolutionType.SUB_DEF:
        raise ValueError(
            f"Alpha_n_max_hybrid was called with v_wall={v_wall} < cs. Use alpha_n_max_deflagration instead."
        )

    # Might have been returned as Detonation, which takes precedence over Hybrid
    sol_type = SolutionType.HYBRID.value
    ap = 1/3 - 1e-8
    _, w, xi = fluid_bag.sound_shell_alpha_plus_bag(v_wall, ap, sol_type, n_xi)
    n_wall = props.find_v_index(xi, v_wall)

    # alpha_N = (w_+/w_N)*alpha_+
    # w_ is normalized to 1 at large xi
    return w[n_wall] * 1/3


@numba.njit
def alpha_n_min_deflagration_bag[T: FloatOrArr](v_wall: T) -> T:
    r"""
    Minimum $\alpha_n$ for a deflagration in the Bag Model. Equal to maximum $\alpha_n$ for a detonation.
    Same as :func:`alpha_n_min_hybrid`, as a hybrid is a supersonic deflagration.

    :param v_wall: $v_\text{wall}$
    :return: $\alpha_{n,\min,\text{deflagration}} = \alpha_{n,\min,\text{hybrid}} = \alpha_{n,\max,\text{detonation}}$
    """
    # This check is implemented in the inner functions
    # check.check_wall_speed(v_wall)
    return alpha_n_max_detonation_bag(v_wall)


@numba.njit
def alpha_n_min_hybrid_bag[T: FloatOrArr](v_wall: T) -> T:
    r"""
    Minimum $\alpha_n$ for a hybrid in the Bag Model. Equal to maximum $\alpha_n$ for a detonation.
    Same as :func:`alpha_n_min_deflagration`, as a hybrid is a supersonic deflagration.

    :param v_wall: $v_\text{wall}$
    :return: $\alpha_{n,\min,\text{hybrid}} = \alpha_{n,\min,\text{deflagration}} = \alpha_{n,\max,\text{detonation}}$
    """
    # This check is implemented in the inner functions
    # check.check_wall_speed(v_wall)
    return alpha_n_max_detonation_bag(v_wall)


@speedup.vectorize(nopython=True)
def alpha_plus_max_detonation_bag(v_wall: th.FloatOrArr) -> th.FloatOrArrNumba:
    r"""
    Maximum allowed value of $\alpha_+$ for a detonation with wall speed $v_\text{wall}$ in the Bag Model.
    Comes from inverting $v_w$ > $v_\text{Jouguet}$.

    $\alpha_{+,\max,\text{detonation}} = \frac{ (1 - \sqrt{3} v_\text{wall})^2 }{ 3(1 - v_\text{wall}^2 }$
    """
    check.check_wall_speed(v_wall)
    if v_wall < const.CS0:
        return 0
    a = 3 * (1 - v_wall ** 2)
    b = (1 - np.sqrt(3) * v_wall) ** 2
    return b / a


@speedup.vectorize(nopython=True)
def alpha_plus_min_hybrid(v_wall: th.FloatOrArr) -> th.FloatOrArrNumba:
    r"""
    Minimum allowed value of $\alpha_+$ for a hybrid with wall speed $v_\text{wall}$ in the Bag Model.
    Condition from coincidence of wall and shock.

    $$\alpha_{+, \min, \text{hybrid}} = \frac{ (1 - \sqrt{3} v_\text{wall})^2 }{ 9 v_\text{wall}^2 - 1}$$

    Todo: Is this specific to the bag model?

    :param v_wall: $v_\text{wall}$
    :return: $\alpha_{+, \min, \text{hybrid}}$
    """
    check.check_wall_speed(v_wall)
    if v_wall < const.CS0:
        return 0
    b = (1 - np.sqrt(3) * v_wall) ** 2
    c = 9 * v_wall ** 2 - 1
    return b / c
