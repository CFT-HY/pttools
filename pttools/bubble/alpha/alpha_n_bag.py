r"""$\alpha_n$ functions for the Bag Model"""

import numba

from pttools import speedup
from pttools.bubble import bag
from pttools.bubble import boundary
from pttools.bubble import const
from pttools.bubble import fluid_bag
from pttools.bubble import check
from pttools.bubble import integrate
from pttools.bubble import props
from pttools.bubble import transition
from pttools.speedup import NUMBA_ENABLE_CACHE
import pttools.type_hints as th


@numba.njit(nogil=True, cache=NUMBA_ENABLE_CACHE)
def find_alpha_n_bag(
        v_wall: th.FloatOrArr,
        alpha_p: float,
        sol_type: boundary.SolutionType = boundary.SolutionType.UNKNOWN,
        n_xi: int = const.N_XI_DEFAULT,
        cs2_fun: th.CS2Fun = bag.cs2_bag_scalar,
        df_dtau_ptr: speedup.DifferentialPointer = integrate.DF_DTAU_PTR_BAG) -> float:
    r"""
    Calculates the transition strength parameter at the nucleation temperature,
    $\alpha_n$, from $\alpha_+$, for given $v_\text{wall}$ in the Bag Model.

    $$\alpha_n = \frac{4 \Delta \theta (T_n)}{3 w(T_n)} = \frac{4}{3} \frac{ \theta_s(T_n) - \theta_b(T_n) }{w(T_n)}$$

    :param v_wall: $v_\text{wall}$, wall speed
    :param alpha_p: $\alpha_+$, the at-wall strength parameter.
    :param sol_type: type of the bubble (detonation, deflagration etc.)
    :param n_xi: number of $\xi$ values to investigate
    :param cs2_fun: $c_s^2$ function
    :param df_dtau_ptr: pointer to the differential equations
    :return: $\alpha_n$, global strength parameter
    """
    check.check_wall_speed(v_wall)
    if sol_type == boundary.SolutionType.UNKNOWN.value:
        sol_type = transition.identify_solution_type_alpha_plus(v_wall, alpha_p).value
    _, w, xi = fluid_bag.sound_shell_alpha_plus_bag(
        v_wall, alpha_p, sol_type, n_xi,
        cs2_fun=cs2_fun, df_dtau_ptr=df_dtau_ptr
    )
    n_wall = props.find_v_index(xi, v_wall)
    return alpha_p * w[n_wall] / w[-1]
