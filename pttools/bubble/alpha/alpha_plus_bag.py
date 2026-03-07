r"""$\alpha_+$ functions for the Bag Model"""

import numba
from numba.extending import overload
import numpy as np
from scipy.optimize import fsolve

from pttools import speedup
from pttools.bubble.alpha.alpha_limits_bag import alpha_n_max_deflagration_bag, alpha_n_max_detonation_bag
from pttools.bubble.alpha.alpha_n_bag import find_alpha_n_bag
from pttools.bubble.alpha.alpha_plus import alpha_plus_initial_guess
from pttools.bubble.cs2_bag import CS2_BAG_SCALAR_PTR
from pttools.bubble import const
from pttools.bubble.cs2 import cs2_converter
from pttools.bubble import integrate
from pttools.bubble.solution_type import SolutionType
import pttools.type_hints as th


def _find_alpha_plus_bag_scalar(
        v_wall: th.FloatOrArr,
        alpha_n_given: float,
        n_xi: int = const.N_XI_DEFAULT,
        cs2_fun_ptr: th.CS2FunScalarPtr = CS2_BAG_SCALAR_PTR,
        df_dtau_ptr: speedup.DifferentialPointer = integrate.DF_DTAU_PTR_BAG,
        xtol: float = const.FIND_ALPHA_PLUS_TOL,
        # parallel: bool = True
        ) -> th.FloatOrArrNumba:
    if alpha_n_given < alpha_n_max_detonation_bag(v_wall):
        # Must be detonation
        # sol_type = SolutionType.DETON
        return alpha_n_given
    if alpha_n_given >= alpha_n_max_deflagration_bag(v_wall):
        # Greater than the maximum possible -> fail
        return np.nan
    sol_type = SolutionType.SUB_DEF if v_wall <= const.CS0 else SolutionType.HYBRID
    ap_initial_guess = alpha_plus_initial_guess(v_wall, alpha_n_given)
    with numba.objmode(ret="float64"):
        cs2_fun = cs2_converter(cs2_fun_ptr)

        # This returns np.float64
        ret: float = fsolve(
            _find_alpha_plus_optimizer_bag,
            ap_initial_guess,
            args=(v_wall, sol_type, n_xi, alpha_n_given, cs2_fun, df_dtau_ptr),
            xtol=xtol,
            factor=0.1)[0]
    return ret


def _find_alpha_plus_bag_arr(
        v_wall: th.FloatOrArr,
        alpha_n_given: float,
        n_xi: int = const.N_XI_DEFAULT,
        cs2_fun_ptr: th.CS2FunScalarPtr = CS2_BAG_SCALAR_PTR,
        df_dtau_ptr: speedup.DifferentialPointer = integrate.DF_DTAU_PTR_BAG,
        xtol: float = const.FIND_ALPHA_PLUS_TOL) -> th.FloatOrArrNumba:
    ap = np.zeros_like(v_wall)
    for i in numba.prange(v_wall.size):  # pylint: disable=not-an-iterable
        ap[i] = _find_alpha_plus_bag_scalar(
            v_wall[i], alpha_n_given, n_xi,
            cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr
        )
    return ap


# _find_alpha_plus_bag_arr_parallel = numba.njit(parallel=True, nogil=True)(_find_alpha_plus_bag_arr)
_find_alpha_plus_bag_arr_single = numba.njit(_find_alpha_plus_bag_arr)  # nogil=True


def _find_alpha_plus_bag_arr_wrapper(
        v_wall: th.FloatOrArr,
        alpha_n_given: float,
        n_xi: int = const.N_XI_DEFAULT,
        cs2_fun_ptr: th.CS2FunScalarPtr = CS2_BAG_SCALAR_PTR,
        df_dtau_ptr: speedup.DifferentialPointer = integrate.DF_DTAU_PTR_BAG,
        xtol: float = const.FIND_ALPHA_PLUS_TOL,
        # parallel: bool = True
        ) -> th.FloatOrArrNumba:
    # if parallel:
    #     return _find_alpha_plus_bag_arr_parallel(
    #         v_wall=v_wall, alpha_n_given=alpha_n_given, n_xi=n_xi,
    #         cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr, xtol=xtol
    #     )
    return _find_alpha_plus_bag_arr_single(
        v_wall=v_wall, alpha_n_given=alpha_n_given, n_xi=n_xi,
        cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr, xtol=xtol
    )


def find_alpha_plus_bag(
        v_wall: th.FloatOrArr,
        alpha_n_given: float,
        n_xi: int = const.N_XI_DEFAULT,
        cs2_fun_ptr: th.CS2FunScalarPtr = CS2_BAG_SCALAR_PTR,
        df_dtau_ptr: speedup.DifferentialPointer = integrate.DF_DTAU_PTR_BAG,
        xtol: float = const.FIND_ALPHA_PLUS_TOL,
        # parallel: bool = True
        ) -> th.FloatOrArrNumba:
    r"""
    Calculate the at-wall strength parameter $\alpha_+$ from given $\alpha_n$ and $v_\text{wall}$ in the Bag Model.

    $$\alpha_+ = \frac{4 \Delta \theta (T_+)}{3 w_+} = \frac{4}{3} \frac{ \theta_s(T_+) - \theta_b(T_+) }{w(T_+)}$$
    (:gw_pt_ssm:`\ `, eq. 2.11)

    Uses :func:`scipy.optimize.fsolve` and therefore spends time in the Python interpreter even when jitted.
    This should be taken into account when running parallel simulations.

    :param v_wall: $v_\text{wall}$, the wall speed
    :param alpha_n_given: $\alpha_n$, the global strength parameter
    :param n_xi: number of $\xi$ points
    :return: $\alpha_+$, the at-wall strength parameter
    """
    if isinstance(v_wall, float):
        return _find_alpha_plus_bag_scalar(
            v_wall, alpha_n_given, n_xi,
            cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr, xtol=xtol  # , parallel=parallel
        )
    if isinstance(v_wall, np.ndarray):
        if not v_wall.ndim:
            return _find_alpha_plus_bag_scalar(
                v_wall.item(), alpha_n_given, n_xi,
                cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr, xtol=xtol  # , parallel=parallel
            )
        return _find_alpha_plus_bag_arr(
            v_wall, alpha_n_given, n_xi,
            cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr, xtol=xtol
        )
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


@overload(find_alpha_plus_bag, jit_options={"nopython": True})
def _find_alpha_plus_bag_numba(
        v_wall: th.FloatOrArr,
        alpha_n_given: float,
        n_xi: int = const.N_XI_DEFAULT,
        cs2_fun_ptr: th.CS2FunScalarPtr = CS2_BAG_SCALAR_PTR,
        df_dtau_ptr: speedup.DifferentialPointer = integrate.DF_DTAU_PTR_BAG,
        xtol: float = const.FIND_ALPHA_PLUS_TOL,
        # parallel: bool = True
        ) -> th.FloatOrArrNumba:
    """This cannot be compiled with nogil=True,
    since this uses :func:`scipy.optimize.fsolve`, which requires "with numba.objmode".
    """
    if isinstance(v_wall, numba.types.Float):
        return _find_alpha_plus_bag_scalar
    if isinstance(v_wall, numba.types.Array):
        if not v_wall.ndim:
            return _find_alpha_plus_bag_scalar
        return _find_alpha_plus_bag_arr_wrapper
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


@numba.njit
def _find_alpha_plus_optimizer_bag(
        alpha: th.FloatArr1D,
        v_wall: float,
        sol_type: SolutionType,
        n_xi: int,
        alpha_n_given: float,
        cs2_fun: th.CS2Fun,
        df_dtau_ptr: speedup.DifferentialPointer) -> float:
    """find_alpha_plus() is looking for the zeroes of this function: $\alpha_n = \alpha_{n,\text{given}}$."""
    return find_alpha_n_bag(
        v_wall, alpha.item(), sol_type, n_xi,
        cs2_fun=cs2_fun, df_dtau_ptr=df_dtau_ptr
    ) - alpha_n_given
