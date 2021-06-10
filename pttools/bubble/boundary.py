import sys

import numpy as np

from . import const
from . import relativity


def v_plus(vm, ap, wall_type):
    """
     Wall frame fluid speed v_plus ahead of the wall, as a function of
     vm = v_minus - fluid speed v_plus behind the wall
     ap = alpha_plus - strength parameter at wall
     wall_type - Detonation, Deflagration, Hybrid
    """
    X = vm + 1. / (3 * vm)
    if wall_type == 'Detonation':
        b = 1.
    else:
        b = -1.
    return_value = (0.5 / (1 + ap)) * (X + b * np.sqrt(X ** 2 + 4. * ap ** 2 + (8. / 3.) * ap - (4. / 3.)))

    if isinstance(return_value, np.ndarray):
        return_value[np.where(isinstance(return_value, complex))] = np.nan
    else:
        if isinstance(return_value, complex):
            return_value = np.nan

    return return_value


def v_minus(vp, ap, wall_type='Detonation'):
    """
     Wall frame fluid speed v_minus behind the wall, as a function of
     vp = v_plus - fluid speed v_plus behind the wall
     ap = alpha_plus - strength parameter at wall
     wall_type - Detonation, Deflagration, Hybrid
    """
    vp2 = vp ** 2
    Y = vp2 + 1. / 3.
    Z = (Y - ap * (1. - vp2))
    X = (4. / 3.) * vp2

    if wall_type == 'Detonation':
        b = +1.
    else:
        b = -1.

    return_value = (0.5 / vp) * (Z + b * np.sqrt(Z ** 2 - X))

    if isinstance(return_value, np.ndarray):
        return_value[np.where(isinstance(return_value, complex))] = np.nan
    else:
        if isinstance(return_value, complex):
            return_value = np.nan

    return return_value


def fluid_speeds_at_wall(v_wall, alpha_p, wall_type):
    """
     Solves fluid speed boundary conditions at the wall, returning
         vfp_w, vfm_w, vfp_p, vfm_p
     Fluid speed vf? just behind (?=m) and just ahead (?=p) of wall,
     in wall (_w) and plasma (_p) frames.
    """
    if v_wall <= 1:
        # print( "max_speed_deflag(alpha_p)= ", max_speed_deflag(alpha_p))
        #     if v_wall < max_speed_deflag(alpha_p) and v_wall <= cs and alpha_p <= 1/3.:
        if wall_type == 'Deflagration':
            vfm_w = v_wall  # Fluid velocity just behind the wall in wall frame (v-)
            vfm_p = relativity.lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_plus(v_wall, alpha_p, wall_type)  # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = relativity.lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        elif wall_type == 'Hybrid':
            vfm_w = const.cs0  # Fluid velocity just behind the wall in plasma frame (hybrid)
            vfm_p = relativity.lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_plus(const.cs0, alpha_p, wall_type)  # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = relativity.lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        elif wall_type == 'Detonation':
            vfm_w = v_minus(v_wall, alpha_p)  # Fluid velocity just behind the wall in wall frame (v-)
            vfm_p = relativity.lorentz(v_wall, vfm_w)  # Fluid velocity just behind the wall in plasma frame
            vfp_w = v_wall  # Fluid velocity just ahead of the wall in wall frame (v+)
            vfp_p = relativity.lorentz(v_wall, vfp_w)  # Fluid velocity just ahead of the wall in plasma frame
        else:
            sys.stderr.write("fluid_speeds_at_wall: error: wall_type wrong or unset")
            sys.exit(1)
    else:
        sys.stderr.write("fluid_speeds_at_wall: error: v_wall > 1")

    return vfp_w, vfm_w, vfp_p, vfm_p


def enthalpy_ratio(v_m, v_p):
    """
     Ratio of enthalpies behind (w_- ) and ahead (w_+) of a shock or
     transition front, w_-/w_+. Uses conservation of momentum in moving frame.
    """
    return relativity.gamma2(v_m) * v_m / (relativity.gamma2(v_p) * v_p)
