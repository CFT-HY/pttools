"""Useful quantities for deciding type of transition"""

import sys

import numpy as np

import pttools.type_hints as th
from . import alpha
from . import boundary
from . import const


def min_speed_deton(alpha: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
     Minimum speed for a detonation (Jouguet speed).
     Equivalent to v_plus(cs0,alpha).
     Note that alpha_plus = alpha_n for detonation.
    """
    return (const.cs0/(1 + alpha))*(1 + np.sqrt(alpha*(2. + 3.*alpha)))


def max_speed_deflag(alpha_p: th.FLOAT_OR_ARR) -> th.FLOAT_OR_ARR:
    """
     Maximum speed for a deflagration: speed where wall and shock are coincident.
     May be greater than 1, meaning that hybrids exist for all wall speeds above cs.
     alpha_plus < 1/3, but alpha_n unbounded above.
    """
    return 1/(3*boundary.v_plus(const.cs0, alpha_p, 'Deflagration'))


def identify_wall_type(v_wall: float, alpha_n: float, exit_on_error: bool = False) -> str:
    """
     Determines wall type from wall speed and global strength parameter.
     wall_type = [ 'Detonation' | 'Deflagration' | 'Hybrid' ]
    """
    # v_wall = wall velocity, alpha_n = relative trace anomaly at nucleation temp outside shell
    wall_type = 'Error' # Default
    if alpha_n < alpha.alpha_n_max_detonation(v_wall):
        # Must be detonation
        wall_type = 'Detonation'
    else:
        if alpha_n < alpha.alpha_n_max_deflagration(v_wall):
            if v_wall <= const.cs0:
                wall_type = "Deflagration"
            else:
                wall_type = "Hybrid"

    if (wall_type == 'Error') & exit_on_error:
        sys.stderr.write('identify_wall_type: \
                         error: no solution for v_wall = {}, alpha_n = {}\n'.format(v_wall,alpha_n))
        sys.exit(1)

    return wall_type


def identify_wall_type_alpha_plus(v_wall: float, alpha_p: float) -> str:
    """
     Determines wall type from wall speed and at-wall strength parameter.
     wall_type = [ 'Detonation' | 'Deflagration' | 'Hybrid' ]
    """
    if v_wall <= const.cs0:
        wall_type = 'Deflagration'
    else:
        if alpha_p < alpha.alpha_plus_max_detonation(v_wall):
            wall_type = 'Detonation'
            if alpha_p > alpha.alpha_plus_min_hybrid(v_wall) and alpha_p < 1/3.:
                sys.stderr.write('identify_wall_type_alpha_plus: warning:\n')
                sys.stderr.write('      Hybrid and Detonation both possible for v_wall = {}, alpha_plus = {}\n'.format(v_wall,alpha_p))
                sys.stderr.write('      Choosing detonation.\n')
        else:
            wall_type = 'Hybrid'


    if alpha_p > (1/3.) and not wall_type == 'Detonation':
        sys.stderr.write('identify_wall_type_alpha_plus: error:\n')
        sys.stderr.write('      no solution for v_wall = {}, alpha_plus = {}\n'.format(v_wall,alpha_p))
        wall_type = 'Error'

    return wall_type
