r"""Functions for fluid differential equations with the bag model

Now in parametric form (Jacky Lindsay and Mike Soughton MPhys project 2017-18).
RHS is Eq (33) in Espinosa et al (plus $\frac{dw}{dt}$ not written there)
"""

import logging
import typing as tp

import numba
import numpy as np

import pttools.type_hints as th
from pttools import speedup
from . import alpha
from . import approx
from . import bag
from . import boundary
from .boundary import Phase, SolutionType
from . import check
from . import const
from . import integrate
from . import props
from . import quantities
from . import shock
from . import transition
from . import trim

logger = logging.getLogger(__name__)


@numba.njit
def fluid_shell(
        v_wall: float,
        alpha_n: float,
        n_xi: int = const.N_XI_DEFAULT,
        cs2_fun: th.CS2Fun = bag.cs2_bag_scalar,
        cs2_fun_ptr: th.CS2FunScalarPtr = bag.CS2_BAG_SCALAR_PTR,
        df_dtau_ptr: speedup.DifferentialPointer = integrate.DF_DTAU_BAG_PTR) \
        -> tp.Union[tp.Tuple[float, float, float], tp.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    r"""
    Finds fluid shell $(v, w, \xi)$ from a given $v_\text{wall}, \alpha_n$, which must be scalars.

    Computes $\alpha_+$ from $\alpha_n$ and then calls :py:func:`fluid_shell_alpha_plus`.

    Bag model only!

    :param v_wall: $v_\text{wall}$
    :param alpha_n: $\alpha_n$
    :param n_xi: number of $\xi$ points
    :return: $v, w, \xi$
    """
    # check_physical_params([v_wall,alpha_n])
    sol_type = transition.identify_solution_type_bag(v_wall, alpha_n)
    if sol_type == SolutionType.ERROR:
        with numba.objmode:
            logger.error("Giving up because of identify_solution_type error")
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr
    al_p = alpha.find_alpha_plus_bag(v_wall, alpha_n, n_xi, cs2_fun_ptr=cs2_fun_ptr, df_dtau_ptr=df_dtau_ptr)
    if np.isnan(al_p):
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr
    # SolutionType has to be passed by its value when jitting
    return fluid_shell_alpha_plus(v_wall, al_p, sol_type.value, n_xi, cs2_fun=cs2_fun, df_dtau_ptr=df_dtau_ptr)


@numba.njit
def fluid_shell_alpha_plus(
        v_wall: float,
        alpha_plus: float,
        sol_type: SolutionType = SolutionType.UNKNOWN,
        n_xi: int = const.N_XI_DEFAULT,
        w_n: float = 1.,
        cs2_fun: th.CS2Fun = bag.cs2_bag,
        df_dtau_ptr: speedup.DifferentialPointer = integrate.DF_DTAU_BAG_PTR,
        sol_type_fun: callable = None) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""
    Finds the fluid shell profile (v, w, xi) from a given $v_\text{wall}, \alpha_+$ (at-wall strength parameter).
    When $v=0$ (behind and ahead of shell), this uses only two points.

    Bag model only!

    :param v_wall: $v_\text{wall}$
    :param alpha_plus: $\alpha_+$
    :param sol_type: specify wall type if more than one permitted.
    :param n_xi: increase resolution
    :param w_n: specify enthalpy outside fluid shell
    :param cs2_fun: sound speed squared as a function of enthalpy, default
    :param df_dtau_ptr: pointer to the differential equation function
    :return: $v, w, \xi$
    """
    # These didn't work, and therefore this function gets cs2_fun as a function instead of a pointer
    # cs2_fun = bag.CS2ScalarCType(cs2_fun_ptr)
    # cs2_fun = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)(cs2_fun)

    check.check_wall_speed(v_wall)

    if sol_type == SolutionType.UNKNOWN.value:
        sol_type = transition.identify_solution_type_alpha_plus(v_wall, alpha_plus).value
    # The identification above may set sol_type to error
    if sol_type == SolutionType.ERROR.value:
        with numba.objmode:
            logger.error("Giving up because of identify_solution_type error")
        nan_arr = np.array([np.nan])
        return nan_arr, nan_arr, nan_arr

    # Solve boundary conditions at wall
    # See the function docstring for the abbreviations
    vfp_w, vfm_w, vfp_p, vfm_p = boundary.fluid_speeds_at_wall(v_wall, alpha_plus, sol_type)
    wp = 1.0  # Nominal value - will be rescaled later
    wm = wp / boundary.enthalpy_ratio(vfm_w, vfp_w)  # enthalpy just behind wall

    dxi = 1. / n_xi
    # dxi = 10*eps

    # Set up parts outside shell where v=0. Need 2 points only.
    # Forwards integration, from v_wall to xi=1
    xif = np.linspace(v_wall + dxi, 1.0, 2)
    vf = np.zeros_like(xif)
    wf = np.ones_like(xif) * wp

    # Backwards integration, from cs or 0 to v_wall
    # TODO: set a value for the phase
    xib = np.linspace(min(cs2_fun(w_n, 0) ** 0.5, v_wall) - dxi, 0.0, 2)
    vb = np.zeros_like(xib)
    wb = np.ones_like(xib) * wm

    # TODO: remove the dependence on sol_type
    # Instead:
    # - check if the value of alpha_plus allows only one type of solution
    # - If yes, compute using that
    # - Otherwise compute both and then see which takes to the correct direction

    # Integrate forward and find shock.
    if not sol_type == SolutionType.DETON.value:
        # First go
        v, w, xi, t = integrate.fluid_integrate_param(
            v0=vfp_p, w0=wp, xi0=v_wall,
            phase=Phase.SYMMETRIC.value, t_end=-const.T_END_DEFAULT, n_xi=const.N_XI_DEFAULT, df_dtau_ptr=df_dtau_ptr)
        v, w, xi, t = trim.trim_fluid_wall_to_shock(v, w, xi, t, sol_type)
        # Now refine so that there are ~N points between wall and shock.  A bit excessive for thin
        # shocks perhaps, but better safe than sorry. Then improve final point with shock_zoom...
        t_end_refine = t[-1]
        v, w, xi, t = integrate.fluid_integrate_param(
            v0=vfp_p, w0=wp, xi0=v_wall,
            phase=Phase.SYMMETRIC.value, t_end=t_end_refine, n_xi=n_xi, df_dtau_ptr=df_dtau_ptr)
        v, w, xi, t = trim.trim_fluid_wall_to_shock(v, w, xi, t, sol_type)
        v, w, xi = shock.shock_zoom_last_element(v, w, xi)
        # Now complete to xi = 1
        vf = np.concatenate((v, vf))
        # enthalpy
        vfp_s = xi[-1]  # Fluid velocity just ahead of shock in shock frame = shock speed
        vfm_s = 1 / (3 * vfp_s)  # Fluid velocity just behind shock in shock frame
        wf = np.ones_like(xif) * w[-1] * boundary.enthalpy_ratio(vfm_s, vfp_s)
        wf = np.concatenate((w, wf))
        # xi
        xif[0] = xi[-1]
        xif = np.concatenate((xi, xif))

    # Integrate backward to sound speed.
    if not sol_type == SolutionType.SUB_DEF.value:
        # First go
        v, w, xi, t = integrate.fluid_integrate_param(
            v0=vfm_p, w0=wm, xi0=v_wall,
            phase=Phase.BROKEN.value, t_end=-const.T_END_DEFAULT, n_xi=const.N_XI_DEFAULT, df_dtau_ptr=df_dtau_ptr)
        v, w, xi, t = trim.trim_fluid_wall_to_cs(v, w, xi, t, v_wall, sol_type)
        #    # Now refine so that there are ~N points between wall and point closest to cs
        #    # For walls just faster than sound, will give very (too?) fine a resolution.
        #        t_end_refine = t[-1]
        #        v,w,xi,t = fluid_integrate_param(vfm_p, wm, v_wall, t_end_refine, n_xi, cs2_fun)
        #        v, w, xi, t = trim_fluid_wall_to_cs(v, w, xi, t, v_wall, sol_type)

        # Now complete to xi = 0
        vb = np.concatenate((v, vb))
        wb = np.ones_like(xib) * w[-1]
        wb = np.concatenate((w, wb))
        # Can afford to bring this point all the way to cs2.
        # TODO: set a value for the phase
        xib[0] = cs2_fun(w[-1], 0) ** 0.5
        xib = np.concatenate((xi, xib))

    # Now put halves together in right order
    # Need to fix this according to python version
    #    v  = np.concatenate((np.flip(vb,0),vf))
    #    w  = np.concatenate((np.flip(wb,0),wf))
    #    w  = w*(w_n/w[-1])
    #    xi = np.concatenate((np.flip(xib,0),xif))
    v = np.concatenate((np.flipud(vb), vf))
    w = np.concatenate((np.flipud(wb), wf))
    # This fixes the scaling of the results.
    # The original scaling does not matter for computing the problem, but the endpoint w[-1] has to match w_n.
    w = w * (w_n / w[-1])
    # The memory layout of the resulting xi array may cause problems with old Numba versions.
    xi = np.concatenate((np.flipud(xib), xif))
    # Using .copy() results in a contiguous memory layout, alleviating the issue above.
    return v, w, xi.copy()


def fluid_shell_params(
        v_wall: float,
        alpha_n: float,
        Np: int = const.N_XI_DEFAULT,
        low_v_approx: bool = False,
        high_v_approx: bool = False):
    if low_v_approx and high_v_approx:
        raise ValueError("Both low and high v approximations can't be enabled at the same time.")

    # TODO: use greek symbols for kappa and omega
    check.check_physical_params((v_wall, alpha_n))

    sol_type = transition.identify_solution_type_bag(v_wall, alpha_n)

    if sol_type is SolutionType.ERROR:
        raise RuntimeError(f"No solution for v_wall = {v_wall}, alpha_n = {alpha_n}")

    v, w, xi = fluid_shell(v_wall, alpha_n, Np)

    # vmax = max(v)

    xi_even = np.linspace(1 / Np, 1 - 1 / Np, Np)
    v_sh = shock.v_shock_bag(xi_even)
    w_sh = shock.wm_shock_bag(xi_even)

    n_wall = props.find_v_index(xi, v_wall)
    n_cs = int(np.floor(const.CS0 * Np))
    n_sh = xi.size - 2

    r = w[n_wall] / w[n_wall - 1]
    alpha_plus = alpha_n * w[-1] / w[n_wall]

    ubarf2 = quantities.ubarf_squared(v, w, xi, v_wall)
    # Kinetic energy fraction of total (Bag equation of state)
    ke_frac = ubarf2 / (0.75 * (1 + alpha_n))
    # Efficiency of turning Higgs potential into kinetic energy
    kappa = ubarf2 / (0.75 * alpha_n)
    # and efficiency of turning Higgs potential into thermal energy
    dw = 0.75 * quantities.mean_enthalpy_change(v, w, xi, v_wall) / (0.75 * alpha_n * w[-1])

    if high_v_approx:
        v_approx = approx.v_approx_high_alpha(xi[n_wall:n_sh], v_wall, v[n_wall])
        w_approx = approx.w_approx_high_alpha(xi[n_wall:n_sh], v_wall, v[n_wall], w[n_wall])
    elif low_v_approx:
        v_approx = approx.v_approx_low_alpha(xi, v_wall, alpha_plus)
        w_approx = approx.w_approx_low_alpha(xi, v_wall, alpha_plus)
    else:
        v_approx = None
        w_approx = None

    return {
        # Arrays
        "v": v,
        "w": w,
        "xi": xi,
        "xi_even": xi_even,
        "v_sh": v_sh,
        "w_sh": w_sh,
        "v_approx": v_approx,
        "w_approx": w_approx,
        # Scalars
        "n_wall": n_wall,
        "n_cs": n_cs,
        "n_sh": n_sh,
        "r": r,
        "alpha_plus": alpha_plus,
        "ubarf2": ubarf2,
        "ke_frac": ke_frac,
        "kappa": kappa,
        "dw": dw,
        "sol_type": sol_type
    }
