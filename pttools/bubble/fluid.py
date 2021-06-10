"""Functions for fluid differential equations

Now in parametric form (Jacky Lindsay and Mike Soughton MPhys project 2017-18)
# RHS is Eq (33) in Espinosa et al (plus dw/dt not written there)
"""

import sys
import typing as tp

import numpy as np
import scipy.integrate as spi

import pttools.type_hints as th
from . import alpha
from . import bag
from . import boundary
from . import check
from . import const
from . import props
from . import transition


def df_dtau(y: np.ndarray, t: float, cs2_fun: bag.CS2_FUN_TYPE = bag.cs2_bag) -> tp.Tuple[float, float, float]:
    """
     Differentials of fluid variables (v, w, xi) in parametric form, suitable for odeint
    """
    v = y[0]
    w = y[1]
    xi = y[2]
    cs2 = cs2_fun(w)
    xiXv = xi * v
    xi_v = xi - v
    v2 = v * v

    dxi_dt = xi * ((xi_v) ** 2 - cs2 * (1 - xiXv) ** 2)  # dxi/dt
    dv_dt = 2 * v * cs2 * (1 - v2) * (1 - xiXv)  # dv/dt
    dw_dt = (w / (1 - v2)) * (xi_v / (1 - xiXv)) * (1 / cs2 + 1) * dv_dt

    return dv_dt, dw_dt, dxi_dt


def fluid_integrate_param(
        v0: th.FLOAT_OR_ARR,
        w0: th.FLOAT_OR_ARR,
        xi0: th.FLOAT_OR_ARR,
        t_end: float = const.T_END_DEFAULT,
        n_xi: int = const.N_XI_DEFAULT,
        cs2_fun: bag.CS2_FUN_TYPE = bag.cs2_bag) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
     Integrates parametric fluid equations in df_dtau from an initial condition.
     Positive t_end integrates along curves from (v,w) = (0,cs0) to (1,1).
     Negative t_end integrates towards (0,cs0).
     Returns: v, w, xi, t
    """
    t = np.linspace(0., t_end, n_xi)
    if isinstance(xi0, np.ndarray):
        soln = spi.odeint(df_dtau, (v0[0], w0[0], xi0[0]), t, args=(cs2_fun,))
    else:
        soln = spi.odeint(df_dtau, (v0, w0, xi0), t, args=(cs2_fun,))
    v = soln[:, 0]
    w = soln[:, 1]
    xi = soln[:, 2]

    return v, w, xi, t


# Main function for integrating fluid equations and deriving v, w
# for complete range 0 < xi < 1

def fluid_shell(
        v_wall: th.FLOAT_OR_ARR,
        alpha_n: th.FLOAT_OR_ARR,
        n_xi: int = const.N_XI_DEFAULT) \
        -> tp.Union[tp.Tuple[float, float, float], tp.Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
     Finds fluid shell (v, w, xi) from a given v_wall, alpha_n, which must be scalars.
     Option to change xi resolution n_xi
    """
    #    check_physical_params([v_wall,alpha_n])
    wall_type = transition.identify_wall_type(v_wall, alpha_n)
    if wall_type == 'Error':
        sys.stderr.write('fluid_shell: giving up because of identify_wall_type error')
        return np.nan, np.nan, np.nan
    else:
        al_p = alpha.find_alpha_plus(v_wall, alpha_n, n_xi)
        if not np.isnan(al_p):
            return fluid_shell_alpha_plus(v_wall, al_p, wall_type, n_xi)
        else:
            return np.nan, np.nan, np.nan


def fluid_shell_alpha_plus(
        v_wall: th.FLOAT_OR_ARR,
        alpha_plus: th.FLOAT_OR_ARR,
        wall_type: str = "Calculate",
        n_xi: int = const.N_XI_DEFAULT,
        w_n: float = 1,
        cs2_fun: bag.CS2_FUN_TYPE = bag.cs2_bag) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
     Finds fluid shell (v, w, xi) from a given v_wall, alpha_plus (at-wall strength parameter).
     Where v=0 (behind and ahead of shell) uses only two points.
     v_wall and alpha_plus must be scalars, and are converted from 1-element arrays if needed.
     Options:
         wall_type (string) - specify wall type if more than one permitted.
         n_xi (int) - increase resolution
         w_n - specify enthalpy outside fluid shell
         cs2_fun - sound speed squared as a function of enthalpy, default
    """
    check.check_wall_speed(v_wall)
    dxi = 1. / n_xi
    #    dxi = 10*eps

    if isinstance(alpha_plus, np.ndarray):
        al_p = np.asscalar(alpha_plus)
    else:
        al_p = alpha_plus
    if isinstance(v_wall, np.ndarray):
        v_w = np.asscalar(v_wall)
    else:
        v_w = v_wall

    if wall_type == "Calculate":
        wall_type = transition.identify_wall_type_alpha_plus(v_w, al_p)

    if wall_type == 'Error':
        sys.stderr.write('fluid_shell_alpha_plus: giving up because of identify_wall_type error')
        return np.nan, np.nan, np.nan

    # Solve boundary conditions at wall
    vfp_w, vfm_w, vfp_p, vfm_p = boundary.fluid_speeds_at_wall(v_w, al_p, wall_type)
    wp = 1.0  # Nominal value - will be rescaled later
    wm = wp / boundary.enthalpy_ratio(vfm_w, vfp_w)  # enthalpy just behind wall

    # Set up parts outside shell where v=0. Need 2 points only.
    xif = np.linspace(v_wall + dxi, 1.0, 2)
    vf = np.zeros_like(xif)
    wf = np.ones_like(xif) * wp

    xib = np.linspace(min(cs2_fun(w_n) ** 0.5, v_w) - dxi, 0.0, 2)
    vb = np.zeros_like(xib)
    wb = np.ones_like(xib) * wm

    # Integrate forward and find shock.
    if not wall_type == 'Detonation':
        # First go
        v, w, xi, t = fluid_integrate_param(vfp_p, wp, v_w, -const.T_END_DEFAULT, const.N_XI_DEFAULT, cs2_fun)
        v, w, xi, t = trim_fluid_wall_to_shock(v, w, xi, t, wall_type)
        # Now refine so that there are ~N points between wall and shock.  A bit excessive for thin
        # shocks perhaps, but better safe than sorry. Then improve final point with shock_zoom...
        t_end_refine = t[-1]
        v, w, xi, t = fluid_integrate_param(vfp_p, wp, v_w, t_end_refine, n_xi, cs2_fun)
        v, w, xi, t = trim_fluid_wall_to_shock(v, w, xi, t, wall_type)
        v, w, xi = props.shock_zoom_last_element(v, w, xi)
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
    if not wall_type == 'Deflagration':
        # First go
        v, w, xi, t = fluid_integrate_param(vfm_p, wm, v_w, -const.T_END_DEFAULT, const.N_XI_DEFAULT, cs2_fun)
        v, w, xi, t = trim_fluid_wall_to_cs(v, w, xi, t, v_wall, wall_type)
        #    # Now refine so that there are ~N points between wall and point closest to cs
        #    # For walls just faster than sound, will give very (too?) fine a resolution.
        #        t_end_refine = t[-1]
        #        v,w,xi,t = fluid_integrate_param(vfm_p, wm, v_w, t_end_refine, n_xi, cs2_fun)
        #        v, w, xi, t = trim_fluid_wall_to_cs(v, w, xi, t, v_wall, wall_type)

        # Now complete to xi = 0
        vb = np.concatenate((v, vb))
        wb = np.ones_like(xib) * w[-1]
        wb = np.concatenate((w, wb))
        # Can afford to bring this point all the way to cs2.
        xib[0] = cs2_fun(w[-1]) ** 0.5
        xib = np.concatenate((xi, xib))

    # Now put halves together in right order
    # Need to fix this according to python version
    #    v  = np.concatenate((np.flip(vb,0),vf))
    #    w  = np.concatenate((np.flip(wb,0),wf))
    #    w  = w*(w_n/w[-1])
    #    xi = np.concatenate((np.flip(xib,0),xif))
    v = np.concatenate((np.flipud(vb), vf))
    w = np.concatenate((np.flipud(wb), wf))
    w = w * (w_n / w[-1])
    xi = np.concatenate((np.flipud(xib), xif))

    return v, w, xi


def trim_fluid_wall_to_cs(
        v: np.ndarray,
        w: np.ndarray,
        xi: np.ndarray,
        t: np.ndarray,
        v_wall: th.FLOAT_OR_ARR, wall_type: str,
        dxi_lim: float = const.dxi_small,
        cs2_fun: bag.CS2_FUN_TYPE = bag.cs2_bag) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
     Picks out fluid variable arrays (v, w, xi, t) which are definitely behind
     the wall for deflagration and hybrid.
     Also removes negative fluid speeds and xi <= sound_speed, which might be left by
     an inaccurate integration.
     If the wall is within about 1e-16 of cs, rounding errors are flagged.
    """
    check.check_wall_speed(v_wall)
    n_start = 0

    n_stop_index = -2
    n_stop = 0
    if not wall_type == 'Deflagration':
        it = np.nditer([v, w, xi], flags=['c_index'])
        for vv, ww, x in it:
            if vv <= 0 or x ** 2 <= cs2_fun(ww):
                n_stop_index = it.index
                break

    if n_stop_index == 0:
        sys.stderr.write('trim_fluid_wall_to_cs: warning: integation gave v < 0 or xi <= cs\n')
        sys.stderr.write('     wall_type: {}, v_wall: {}, xi[0] = {}, v[] = {}\n'.format(
            wall_type, v_wall, xi[0], v[0]))
        sys.stderr.write(
            '     Fluid profile has only one element between vw and cs. Fix implemented by adding one extra point.\n')
        n_stop = 1
    else:
        n_stop = n_stop_index

    if (xi[0] == v_wall) and not (wall_type == "Detonation"):
        n_start = 1
        n_stop += 1

    return v[n_start:n_stop], w[n_start:n_stop], xi[n_start:n_stop], t[n_start:n_stop]


def trim_fluid_wall_to_shock(
        v: np.ndarray,
        w: np.ndarray,
        xi: np.ndarray,
        t: np.ndarray,
        wall_type: str) -> tp.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
     Trims fluid variable arrays (v, w, xi) so last element is just ahead of shock
    """
    n_shock_index = -2
    n_shock = 0
    if not wall_type == 'Detonation':
        it = np.nditer([v, xi], flags=['c_index'])
        for vv, x in it:
            if vv <= props.v_shock(x):
                n_shock_index = it.index
                break

    if n_shock_index == 0:
        sys.stderr.write('trim_fluid_wall_to_shock: warning: v[0] < v_shock(xi[0]\n')
        sys.stderr.write('     wall_type: {}, xi[0] = {}, v[0] = {}, v_sh(xi[0]) = {}\n'.format(
            wall_type, xi[0], v[0], props.v_shock(xi[0])))
        sys.stderr.write('     Shock profile has only one element. Fix implemented by adding one extra point.\n')
        n_shock = 1
    else:
        n_shock = n_shock_index

    #    print("n_shock",n_shock,v[n_shock],v_shock(xi[n_shock]))
    return v[:n_shock + 1], w[:n_shock + 1], xi[:n_shock + 1], t[:n_shock + 1]
