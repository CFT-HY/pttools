import sys

import numpy as np

from . import alpha


def check_wall_speed(v_wall):
    """
    Checks that v_wall values are all physical (0 < v_wall <1)
    """
    if isinstance(v_wall, float):
        if v_wall >= 1.0 or v_wall <= 0.0:
            sys.exit('check_wall_speed: error: unphysical parameter(s)\n\
                     v_wall = {}, require 0 < v_wall < 1'.format(v_wall))
    elif isinstance(v_wall, np.ndarray):
        if np.logical_or(np.any(v_wall >= 1.0), np.any(v_wall <= 0.0)):
            sys.exit('check_wall_speed: error: unphysical parameter(s)\n\
                     at least one value outside 0 < v_wall < 1')
    elif isinstance(v_wall, list):
        for vw in v_wall:
            if vw >= 1.0 or vw <= 0.0:
                sys.exit('check_wall_speed: error: unphysical parameter(s)\n\
                         at least one value outside 0 < v_wall < 1')
    else:
        sys.exit('check_wall_speed: error: v_wall must be float, list or array.\n ')

    return None


def check_physical_params(params):
    """
    Checks that v_wall = params[0], alpha_n = params[1] values are physical, i.e.
         0 < v_wall <1
         alpha_n < alpha_n_max(v_wall)
    """
    v_wall = params[0]
    alpha_n = params[1]
    check_wall_speed(v_wall)

    if alpha_n > alpha.alpha_n_max(v_wall):
        sys.exit('check_alpha_n: error: unphysical parameter(s)\n\
                     v_wall, alpha_n = {}, {}\n\
                     require alpha_n < {}\n'.format(v_wall, alpha_n, alpha.alpha_n_max(v_wall)))
    return None
