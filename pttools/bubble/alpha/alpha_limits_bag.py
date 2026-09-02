r"""$\alpha_n$ limits for the Bag Model"""

import numba
from numba.extending import overload
import numpy as np

from pttools.bubble.const import ALPHA_PLUS_MAX_DEF, CS0, DEFAULT_N_XI
from pttools.bubble.fluid_bag import sound_shell_alpha_plus_bag
from pttools.bubble import check
from pttools.bubble import props
from pttools.bubble.solution_type import SolutionType
# from pttools.bubble.solution_type_bag import identify_solution_type_alpha_plus_bag
from pttools.speedup import njit, vectorize
from pttools.bubble.integrate import FluidIntegrateMethod
from pttools.speedup.differential import DifferentialPointer
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr


@njit(nogil=True)
def alpha_n_max_bag(
        v_wall: th.FloatOrArr,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        n_xi: int = DEFAULT_N_XI) -> th.FloatOrArr:
    r"""
    Calculates the maximum relative trace anomaly outside the bubble, $\alpha_{n,\max,\text{bag}}({v}_\text{wall})$.
    Bag model only.

    This limit is for subsonic deflagrations and supersonic deflagrations (hybrids),
    as increasing $\alpha_n$ for a detonation would make it a hybrid.

    :param v_wall: ${v}_\text{wall}$
    :param df_dtau_ptr: pointer to the differential equation function
    :param ode_method: differential equation solver to be used
    :param n_xi: number of $\xi$ points
    :return: $\alpha_{n,\max}$, the relative trace anomaly outside the bubble
    """
    return alpha_n_max_deflagration_bag(
        v_wall, df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, n_xi=n_xi)


def _alpha_n_max_deflagration_bag_scalar(
        v_wall: th.FloatOrArr,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        n_xi: int = DEFAULT_N_XI,
        parallel: bool = True) -> th.FloatOrArr:
    check.check_wall_speed(v_wall)
    if v_wall > 0.9999:
        # Alpha_n_max diverges as v_wall -> 1, and the solver fails to find the correct solution.
        return np.nan
    _, w, xi = sound_shell_alpha_plus_bag(
        v_wall=v_wall,
        # Warning: this is not safe. Causes warnings for low v_wall.
        alpha_plus=ALPHA_PLUS_MAX_DEF - 1e-10,
        df_dtau_ptr=df_dtau_ptr,
        ode_method=ode_method,
        sol_type=SolutionType.HYBRID.value if v_wall > CS0 else SolutionType.SUB_DEF.value,
        n_xi=n_xi
    )
    i_wall = props.find_v_index(xi, v_wall)
    # alpha_n = (w_+/w_N)*alpha_+
    # w is normalized to 1 at large xi
    # Need n_wall+1, as w is an integral of v, and lags by 1 step
    return w[i_wall + 1] * ALPHA_PLUS_MAX_DEF

_alpha_n_max_deflagration_bag_scalar_numba = njit(_alpha_n_max_deflagration_bag_scalar)


def _alpha_n_max_deflagration_bag_arr(
        v_wall: th.FloatOrArr,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        n_xi: int = DEFAULT_N_XI) -> th.FloatOrArr:
    ret = np.zeros_like(v_wall)
    for i in numba.prange(v_wall.size):  # pylint: disable=not-an-iterable
        ret[i] = _alpha_n_max_deflagration_bag_scalar_numba(
            v_wall[i], df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, n_xi=n_xi)
    return ret

_alpha_n_max_deflagration_bag_arr_parallel = njit(parallel=True, nogil=True)(_alpha_n_max_deflagration_bag_arr)
_alpha_n_max_deflagration_bag_arr_single = njit(nogil=True)(_alpha_n_max_deflagration_bag_arr)


def _alpha_n_max_deflagration_bag_arr_wrapper(
        v_wall: th.FloatOrArr,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        n_xi: int = DEFAULT_N_XI,
        parallel: bool = True) -> th.FloatOrArr:
    if parallel:
        return _alpha_n_max_deflagration_bag_arr_parallel(
            v_wall=v_wall, df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, n_xi=n_xi)
    return _alpha_n_max_deflagration_bag_arr_single(
        v_wall=v_wall, df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, n_xi=n_xi)


def alpha_n_max_deflagration_bag(
        v_wall: th.FloatOrArr,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        n_xi: int = DEFAULT_N_XI,
        parallel: bool = True) -> th.FloatOrArr:
    r"""
    Calculates the maximum phase transition strength $\alpha_{n,\max}$,
    in the Bag Model for given $v_\text{wall}$, for deflagration.
    Works also for hybrids, as they are supersonic deflagrations.

    Internally, uses :func:`sound_shell_alpha_plus_bag` and the fact that
    $$\alpha_n = \frac{w_+}{w_n} \alpha_+$$.

    :param v_wall: $v_\text{wall}$
    :param df_dtau_ptr: pointer to the differential equation function
    :param ode_method: differential equation solver to be used
    :param n_xi: number of $\xi$ points
    :return: $\alpha_{n,\max}$
    """
    if isinstance(v_wall, float):
        return _alpha_n_max_deflagration_bag_scalar(
            v_wall=v_wall, df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, n_xi=n_xi)
    if isinstance(v_wall, np.ndarray):
        if not v_wall.ndim:
            return _alpha_n_max_deflagration_bag_scalar(
                v_wall=v_wall.item(), df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, n_xi=n_xi)
        return _alpha_n_max_deflagration_bag_arr(
            v_wall=v_wall, df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, n_xi=n_xi)
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


@overload(alpha_n_max_deflagration_bag, jit_options={"nopython": True, "nogil": True})
def _alpha_n_max_deflagration_bag_numba(
        v_wall: th.FloatOrArr,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        n_xi: int = DEFAULT_N_XI,
        parallel: bool = True) -> th.FloatOrArr:
    if isinstance(v_wall, numba.types.Float):
        return _alpha_n_max_deflagration_bag_scalar
    if isinstance(v_wall, numba.types.Array):
        if not v_wall.ndim:
            return _alpha_n_max_deflagration_bag_scalar
        return _alpha_n_max_deflagration_bag_arr_wrapper
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


@njit
def alpha_n_max_detonation_bag(v_wall: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Maximum allowed value of $\alpha_n$ for a detonation with wall speed $v_\text{wall}$ in the Bag Model.
    Same as :func:`alpha_plus_max_detonation`, since for a detonation $\alpha_n = \alpha_+$,
    as there is no fluid movement outside the wall.

    :param v_wall: $v_\text{wall}$
    :return: $\alpha_{n,\max,\text{detonation}}$
    """
    return alpha_plus_max_detonation_bag(v_wall)


# @njit
# def alpha_n_max_hybrid_bag(v_wall: float, n_xi: int = DEFAULT_N_XI) -> float:
#     r"""
#     Calculates the relative trace anomaly outside the bubble, $\alpha_{n,\max}$,
#     in the Bag Model for given $v_\text{wall}$, assuming hybrid fluid shell
#
#     :param v_wall: $v_\text{wall}$
#     :param n_xi: number of $\xi$ points
#     :return: $\alpha_{n,\max}$
#     """
#     sol_type = identify_solution_type_alpha_plus_bag(v_wall=v_wall, alpha_p=ALPHA_PLUS_MAX).value
#     if sol_type == SolutionType.SUB_DEF:
#         raise ValueError(
#             f"Alpha_n_max_hybrid was called with v_wall={v_wall} < cs. Use alpha_n_max_deflagration instead."
#         )
#     _, w, xi = sound_shell_alpha_plus_bag(
#         v_wall=v_wall,
#         alpha_plus=ALPHA_PLUS_MAX - 1e-8,
#         # Might have been returned as Detonation, which takes precedence over Hybrid
#         sol_type=SolutionType.HYBRID.value,
#         n_xi=n_xi
#     )
#     n_wall = props.find_v_index(xi, v_wall)
#     # alpha_N = (w_+/w_N)*alpha_+
#     # w is normalized to 1 at large xi
#     return w[n_wall] * ALPHA_PLUS_MAX


@njit
def alpha_n_min_deflagration_bag[T: FloatOrArr](v_wall: T) -> T:
    r"""
    Minimum $\alpha_n$ for a deflagration in the Bag Model. Equal to maximum $\alpha_n$ for a detonation.
    Same as :py:func:`alpha_n_min_hybrid_bag`, as a hybrid is a supersonic deflagration.

    :param v_wall: $v_\text{wall}$
    :return: $\alpha_{n,\min,\text{deflagration}} = \alpha_{n,\min,\text{hybrid}} = \alpha_{n,\max,\text{detonation}}$
    """
    return alpha_n_max_detonation_bag(v_wall)


@njit
def alpha_n_min_hybrid_bag[T: FloatOrArr](v_wall: T) -> T:
    r"""
    Minimum $\alpha_n$ for a hybrid in the Bag Model. Equal to maximum $\alpha_n$ for a detonation.
    Same as :py:func:`alpha_n_min_deflagration_bag`, as a hybrid is a supersonic deflagration.

    :param v_wall: $v_\text{wall}$
    :return: $\alpha_{n,\min,\text{hybrid}} = \alpha_{n,\min,\text{deflagration}} = \alpha_{n,\max,\text{detonation}}$
    """
    return alpha_n_max_detonation_bag(v_wall)


@vectorize(nopython=True)
def alpha_plus_max_detonation_bag(v_wall: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Maximum allowed value of $\alpha_+$ for a detonation with wall speed $v_\text{wall}$ in the Bag Model.

    $$\alpha_{+,\max,\text{detonation}} = \frac{ (1 - \sqrt{3} v_\text{wall})^2 }{ 3(1 - v_\text{wall}^2 }$$

    This comes from inverting
    $$v_\text{wall} > v_\text{CJ}$$,
    for $v_\text{CJ}$ of :py:func:`pttools.bubble.v_chapman_jouguet_bag`.
    """
    # Todo: Is this specific to the bag model?
    check.check_wall_speed(v_wall)
    if v_wall < CS0:
        return 0
    a = (1 - np.sqrt(3) * v_wall) ** 2
    b = 3 * (1 - v_wall ** 2)
    return a / b


@vectorize(nopython=True)
def alpha_plus_min_hybrid(v_wall: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Minimum allowed value of $\alpha_+$ for a hybrid with wall speed $v_\text{wall}$ in the Bag Model.
    Condition from coincidence of wall and shock.

    $$\alpha_{+,\min,\text{hybrid}} = \frac{ (1 - \sqrt{3} v_\text{wall})^2 }{ 9 v_\text{wall}^2 - 1 }$$

    :param v_wall: $v_\text{wall}$
    :return: $\alpha_{+,\min,\text{hybrid}}$
    """
    # Todo: Is this specific to the bag model?
    check.check_wall_speed(v_wall)
    if v_wall < CS0:
        return 0
    a = (1 - np.sqrt(3) * v_wall) ** 2
    c = 9 * v_wall ** 2 - 1
    return a / c
