import logging

import numpy as np

from pttools.bubble.bubble import Bubble
from pttools.bubble.boundary import SolutionType
from pttools.bubble import relativity

logger = logging.getLogger(__name__)


def compute_entropy_region(bubble: Bubble, start_ind: int, stop_ind: int, reverse: bool = False):
    """Compute the entropy profile of a bubble section by integrating its differential equation"""

    v = bubble.v[start_ind:stop_ind]
    xi = bubble.xi[start_ind:stop_ind]
    if reverse:
        v = v[::-1]
        xi = xi[::-1]
        s0 = bubble.s[stop_ind-1]
        s1 = bubble.s[stop_ind-2]
        s_end = bubble.s[start_ind]
        v_extra = bubble.v[start_ind-1:stop_ind][::-1]
        xi_extra = bubble.xi[start_ind-1:stop_ind][::-1]
    else:
        s0 = bubble.s[start_ind]
        s1 = bubble.s[start_ind+1]
        s_end = bubble.s[stop_ind-1]
        v_extra = bubble.v[start_ind:stop_ind+1]
        xi_extra = bubble.xi[start_ind:stop_ind+1]

    s = np.zeros_like(v)
    s[0] = s0
    s[1] = s1
    s_prev = s1
    # Cut arrays to match the size of the diff arrays: start at i=1
    v_cut = v[1:]
    xi_cut = xi[1:]
    # Differences are for the midpoints of the intervals: start at i=0.5
    v_diff_mid = np.diff(v_extra)
    xi_diff_mid = np.diff(xi_extra)
    # dv/dxi
    v_xi_diff_mid = v_diff_mid / xi_diff_mid
    # Take the midpoints of the midpoints -> start at i=1
    xi_diff = (xi_diff_mid[1:] + xi_diff_mid[:-1]) / 2
    v_xi_diff = (v_xi_diff_mid[1:] + v_xi_diff_mid[:-1]) / 2
    # cut = starts at i=1, v_xi diff starts at i=1
    s_diff_rel = 1 / (xi_cut - v_cut) * (
        2 * v_cut / xi_cut + (1 - relativity.gamma2(v_cut) * v_cut * (xi_cut - v_cut)) * v_xi_diff
    )

    for i in range(0, s.size-2):
        # Derivative at point i+1, using s_prev of point i+1
        s_diff = s_prev * s_diff_rel[i]
        # New point i+2
        s_prev = s_prev + s_diff * xi_diff[i]
        s[i+2] = s_prev

    abs_error = s[-1] - s_end
    rel_error = abs_error / s_end
    logger.info(
        "Entropy integration: %s, reverse=%s, absolute error: %s, relative error: %s",
        bubble.sol_type, reverse, abs_error, rel_error
    )

    if reverse:
        s = s[::-1]

    return s


def compute_entropy(bubble: Bubble):
    s = bubble.s.copy()

    start_ind: int
    stop_ind: int
    reverse: bool = False
    if bubble.sol_type == SolutionType.DETON:
        start_ind = np.argmax(bubble.v > 0)
        stop_ind = np.nonzero(bubble.xi < bubble.v_wall)[0][-1]
        reverse = True
    elif bubble.sol_type == SolutionType.SUB_DEF:
        start_ind = np.argmax(bubble.xi > bubble.v_wall)
        stop_ind = np.nonzero(bubble.v > 0)[0][-1]
    elif bubble.sol_type == SolutionType.HYBRID:
        # Behind the wall
        start_ind = np.argmax(bubble.v > 0)
        stop_ind = np.nonzero(bubble.xi < bubble.v_wall)[0][-1]
        s[start_ind:stop_ind] = compute_entropy_region(bubble, start_ind, stop_ind, reverse=True)
        # Ahead of the wall
        start_ind = np.argmax(bubble.xi > bubble.v_wall)
        stop_ind = np.nonzero(bubble.xi < bubble.v_sh)[0][-1]
    else:
        raise ValueError("Invalid solution type")

    s[start_ind:stop_ind] = compute_entropy_region(bubble, start_ind, stop_ind, reverse=reverse)

    return s
