r"""$\alpha_+$ functions for the Bag Model."""

import numba
from numba.extending import overload
import numpy as np
from scipy.optimize import fsolve

from pttools import speedup
from pttools.bubble import const
from pttools.bubble.alpha.alpha_limits_bag import alpha_n_max_deflagration_bag, alpha_n_max_detonation_bag
from pttools.bubble.alpha.alpha_n_bag import find_alpha_n_bag
from pttools.bubble.alpha.alpha_plus import alpha_plus_initial_guess
from pttools.bubble.integrate import FluidIntegrateMethod
from pttools.bubble.solution_type import SolutionType
from pttools.speedup import njit
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr


def _find_alpha_plus_bag_scalar(
        v_wall: th.FloatOrArr,
        alpha_n_given: float,
        df_dtau_ptr: speedup.DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        cs2_ptr: th.CS2FunScalarPtr,
        n_xi: int = const.DEFAULT_N_XI,
        xtol: float = const.FIND_ALPHA_PLUS_TOL,
        # parallel: bool = True
        ) -> th.FloatOrArr:
    if alpha_n_given < alpha_n_max_detonation_bag(v_wall):
        # Must be detonation
        # sol_type = SolutionType.DETON
        return alpha_n_given
    if alpha_n_given >= alpha_n_max_deflagration_bag(
            v_wall, df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, cs2_ptr=cs2_ptr):
        # Greater than the maximum possible -> fail
        return np.nan
    sol_type = SolutionType.SUB_DEF if v_wall <= const.CS0 else SolutionType.HYBRID
    ap_initial_guess = alpha_plus_initial_guess(
        v_wall, alpha_n_given, df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, cs2_ptr=cs2_ptr)
    with numba.objmode(ret="float64"):
        # This returns np.float64
        # The SciPy stubs require func to return an array, but a scalar is also accepted at runtime.
        ret: float = fsolve(  # pyrefly: ignore[no-matching-overload]
            _find_alpha_plus_optimizer_bag,
            ap_initial_guess,
            args=(v_wall, sol_type, n_xi, alpha_n_given, cs2_ptr, df_dtau_ptr, ode_method),
            xtol=xtol,
            factor=0.1)[0]
    return ret


def _find_alpha_plus_bag_arr(
        v_wall: th.FloatArr,
        alpha_n_given: float,
        df_dtau_ptr: speedup.DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        cs2_ptr: th.CS2FunScalarPtr,
        n_xi: int = const.DEFAULT_N_XI,
        xtol: float = const.FIND_ALPHA_PLUS_TOL) -> th.FloatArr:
    ap = np.zeros_like(v_wall)
    for i in numba.prange(v_wall.size):
        ap[i] = _find_alpha_plus_bag_scalar(
            v_wall[i], alpha_n_given,
            df_dtau_ptr=df_dtau_ptr, ode_method=ode_method,
            cs2_ptr=cs2_ptr, n_xi=n_xi
        )
    return ap


# _find_alpha_plus_bag_arr_parallel = njit(parallel=True, nogil=True)(_find_alpha_plus_bag_arr)
_find_alpha_plus_bag_arr_single = njit(_find_alpha_plus_bag_arr)  # nogil=True


def _find_alpha_plus_bag_arr_wrapper(
        v_wall: th.FloatOrArr,
        alpha_n_given: float,
        df_dtau_ptr: speedup.DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        cs2_ptr: th.CS2FunScalarPtr,
        n_xi: int = const.DEFAULT_N_XI,
        xtol: float = const.FIND_ALPHA_PLUS_TOL,
        # parallel: bool = True
        ) -> th.FloatArr:
    # if parallel:
    #     return _find_alpha_plus_bag_arr_parallel(
    #         v_wall=v_wall, alpha_n_given=alpha_n_given, n_xi=n_xi,
    #         df_dtau_ptr=df_dtau_ptr, xtol=xtol
    #     )
    # The v_wall annotation has to be identical to that of the overload typing function,
    # but only arrays can end up here.
    return _find_alpha_plus_bag_arr_single(
        v_wall=v_wall,  # pyrefly: ignore[bad-argument-type]
        alpha_n_given=alpha_n_given,
        df_dtau_ptr=df_dtau_ptr, ode_method=ode_method,
        cs2_ptr=cs2_ptr, n_xi=n_xi, xtol=xtol
    )


def find_alpha_plus_bag[T: FloatOrArr](
        v_wall: T,
        alpha_n_given: float,
        df_dtau_ptr: speedup.DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        cs2_ptr: th.CS2FunScalarPtr,
        n_xi: int = const.DEFAULT_N_XI,
        xtol: float = const.FIND_ALPHA_PLUS_TOL,
        # parallel: bool = True
        ) -> T:
    r"""
    Calculate the at-wall strength parameter $\alpha_+$ from given $\alpha_n$ and $v_\text{wall}$ in the Bag Model.

    $$\alpha_+ = \frac{4 \Delta \theta (T_+)}{3 w_+} = \frac{4}{3} \frac{ \theta_s(T_+) - \theta_b(T_+) }{w(T_+)}$$
    (:gw_pt_ssm:`\ `, eq. 2.11)

    Uses :func:`scipy.optimize.fsolve` and therefore spends time in the Python interpreter even when jitted.
    This should be taken into account when running parallel simulations.

    :param v_wall: $v_\text{wall}$, the wall speed
    :param alpha_n_given: $\alpha_n$, the global strength parameter
    :param df_dtau_ptr: pointer to the differential equation function
    :param ode_method: differential equation solver to be used
    :param cs2_ptr: pointer to the $c_s^2$ function
    :param n_xi: number of $\xi$ points
    :return: $\alpha_+$, the at-wall strength parameter
    """
    if isinstance(v_wall, float):
        return _find_alpha_plus_bag_scalar(  # pyrefly: ignore[bad-return]
            v_wall, alpha_n_given,
            df_dtau_ptr=df_dtau_ptr, ode_method=ode_method,
            cs2_ptr=cs2_ptr, n_xi=n_xi, xtol=xtol  # , parallel=parallel
        )
    if isinstance(v_wall, np.ndarray):
        if not v_wall.ndim:
            return _find_alpha_plus_bag_scalar(  # pyrefly: ignore[bad-return]
                v_wall.item(), alpha_n_given,
                df_dtau_ptr=df_dtau_ptr, ode_method=ode_method,
                cs2_ptr=cs2_ptr, n_xi=n_xi, xtol=xtol  # , parallel=parallel
            )
        return _find_alpha_plus_bag_arr(  # pyrefly: ignore[bad-return]
            v_wall, alpha_n_given,
            df_dtau_ptr=df_dtau_ptr, ode_method=ode_method,
            cs2_ptr=cs2_ptr, n_xi=n_xi, xtol=xtol
        )
    raise TypeError(f"Unknown type for v_wall: {type(v_wall)}")


@overload(find_alpha_plus_bag, jit_options={"nopython": True})
def _find_alpha_plus_bag_numba(
        v_wall: th.FloatOrArr,
        alpha_n_given: float,
        df_dtau_ptr: speedup.DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        cs2_ptr: th.CS2FunScalarPtr,
        n_xi: int = const.DEFAULT_N_XI,
        xtol: float = const.FIND_ALPHA_PLUS_TOL,
        # parallel: bool = True
        ) -> th.NumbaFunc:
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


@njit
def _find_alpha_plus_optimizer_bag(
        alpha: th.FloatArr1D,
        v_wall: float,
        sol_type: SolutionType,
        n_xi: int,
        alpha_n_given: float,
        cs2_ptr: th.CS2FunScalarPtr,
        df_dtau_ptr: speedup.DifferentialPointer,
        ode_method: FluidIntegrateMethod) -> float:
    r"""find_alpha_plus() is looking for the zeroes of this function: $\alpha_n = \alpha_{n,\text{given}}$."""
    return find_alpha_n_bag(
        v_wall, alpha.item(),
        df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, cs2_ptr=cs2_ptr, sol_type=sol_type, n_xi=n_xi
    ) - alpha_n_given
