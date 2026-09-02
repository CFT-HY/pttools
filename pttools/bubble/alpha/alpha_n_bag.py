r"""$\alpha_n$ functions for the Bag Model"""

from pttools import speedup
from pttools.bubble.cs2_bag import cs2_bag_scalar
from pttools.bubble import const
from pttools.bubble import fluid_bag
from pttools.bubble.integrate import FluidIntegrateMethod
from pttools.bubble import check
from pttools.bubble import props
from pttools.bubble.solution_type import SolutionType
from pttools.bubble.solution_type_bag import identify_solution_type_alpha_plus_bag
from pttools.speedup import njit
import pttools.type_hints as th


@njit(nogil=True)
def find_alpha_n_bag(
        v_wall: th.FloatOrArr,
        alpha_p: float,
        df_dtau_ptr: speedup.DifferentialPointer,
        ode_method: FluidIntegrateMethod,
        sol_type: SolutionType = SolutionType.UNKNOWN,
        n_xi: int = const.DEFAULT_N_XI,
        cs2_fun: th.CS2Fun = cs2_bag_scalar) -> float:
    r"""
    Calculates the transition strength parameter at the nucleation temperature,
    $\alpha_n$, from $\alpha_+$, for given $v_\text{wall}$ in the Bag Model.

    $$\alpha_n = \frac{4 \Delta \theta (T_n)}{3 w(T_n)} = \frac{4}{3} \frac{ \theta_s(T_n) - \theta_b(T_n) }{w(T_n)}$$

    :param v_wall: $v_\text{wall}$, wall speed
    :param alpha_p: $\alpha_+$, the at-wall strength parameter.
    :param df_dtau_ptr: pointer to the differential equations
    :param ode_method: differential equation solver to be used
    :param sol_type: type of the bubble (detonation, deflagration etc.)
    :param n_xi: number of $\xi$ values to investigate
    :param cs2_fun: $c_s^2$ function
    :return: $\alpha_n$, global strength parameter
    """
    check.check_wall_speed(v_wall)
    if sol_type == SolutionType.UNKNOWN.value:
        sol_type = identify_solution_type_alpha_plus_bag(v_wall, alpha_p).value
    _, w, xi = fluid_bag.sound_shell_alpha_plus_bag(
        v_wall, alpha_p,
        df_dtau_ptr=df_dtau_ptr, ode_method=ode_method, sol_type=sol_type, n_xi=n_xi, cs2_fun=cs2_fun
    )
    n_wall = props.find_v_index(xi, v_wall)
    return alpha_p * w[n_wall] / w[-1]
