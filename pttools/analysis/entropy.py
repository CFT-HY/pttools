import numpy as np

from pttools.bubble.bubble import Bubble
from pttools.bubble.boundary import SolutionType
from pttools.bubble import relativity


def compute_entropy(bubble: Bubble):
    s_ref = bubble.model.s(bubble.w, bubble.phase)

    reverse = False
    if bubble.sol_type == SolutionType.DETON:
        start_ind = np.argmax(bubble.v > 0)
        stop_ind = np.nonzero(bubble.xi < bubble.v_wall)[0][-1]
        reverse = True
    elif bubble.sol_type == SolutionType.SUB_DEF:
        start_ind = np.argmax(bubble.xi > bubble.v_wall)
        stop_ind = np.nonzero(bubble.v > 0)[0][-1]
    elif bubble.sol_type == SolutionType.HYBRID:
        start_ind = np.argmax(bubble.xi > bubble.v_wall)
        stop_ind = np.nonzero(bubble.xi < bubble.v_sh)[0][-1]
    else:
        raise ValueError("Invalid solution type")

    if stop_ind < start_ind:
        raise RuntimeError(f"Invalid start and stop indices: start={start_ind}, stop={stop_ind}")

    v = bubble.v[start_ind:stop_ind]
    xi = bubble.xi[start_ind:stop_ind]
    if reverse:
        v = v[::-1]
        xi = xi[::-1]
        s0 = bubble.model.s(bubble.w[stop_ind], bubble.phase[stop_ind])
    else:
        s0 = bubble.model.s(bubble.w[start_ind], bubble.phase[stop_ind])

    s = np.zeros_like(v)
    s[0] = s0
    s_prev = s0
    v_cut = v[1:]
    xi_cut = xi[1:]
    # Todo: there are probably bugs here
    s_diff_rel = v_cut / xi_cut + (1 - relativity.gamma2(v_cut) * v_cut * (xi_cut - v_cut)) * np.diff(v) / np.diff(xi)

    for i in range(1, s.size-1):
        s_diff = s_prev * s_diff_rel[i]
        s[i] = s_prev - s_diff
        s_prev = s[i]

    if reverse:
        s = s[::-1]

    s_full = s_ref.copy()
    s_full[start_ind:stop_ind] = s
    return s_full, s_ref
