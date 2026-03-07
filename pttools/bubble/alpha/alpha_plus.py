r"""$\alpha_+$ functions"""

import numba

from pttools.bubble.alpha.alpha_limits_bag import \
    alpha_n_min_hybrid_bag, alpha_n_max_deflagration_bag, alpha_plus_min_hybrid
import pttools.type_hints as th


@numba.njit
def alpha_plus_initial_guess(v_wall: th.FloatOrArr, alpha_n_given: float) -> th.FloatOrArr:
    r"""
    Initial guess for root-finding of $\alpha_+$ from $\alpha_n$.
    Linear approx between $\alpha_{n,\min}$ and $\alpha_{n,\max}$.
    Doesn't do obvious checks like Detonation - needs improving?

    :param v_wall: $v_\text{wall}$, wall speed
    :param alpha_n_given: $\alpha_{n, \text{given}}$
    :return: initial guess for $\alpha_+$
    """
    if alpha_n_given < 0.05:
        return alpha_n_given

    alpha_plus_min = alpha_plus_min_hybrid(v_wall)
    alpha_plus_max = 1/3

    alpha_n_min = alpha_n_min_hybrid_bag(v_wall)
    alpha_n_max = alpha_n_max_deflagration_bag(v_wall)

    slope = (alpha_plus_max - alpha_plus_min) / (alpha_n_max - alpha_n_min)
    return alpha_plus_min + slope * (alpha_n_given - alpha_n_min)
