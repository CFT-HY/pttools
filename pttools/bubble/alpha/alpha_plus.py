r"""$\alpha_+$ functions"""

from pttools.bubble.alpha.alpha_limits_bag import \
    alpha_n_min_hybrid_bag, alpha_n_max_deflagration_bag, alpha_plus_min_hybrid
from pttools.speedup import njit
from pttools.bubble.integrate import FluidIntegrateMethod
from pttools.speedup.differential import DifferentialPointer
import pttools.type_hints as th


@njit
def alpha_plus_initial_guess(
        v_wall: th.FloatOrArr,
        alpha_n_given: float,
        df_dtau_ptr: DifferentialPointer,
        ode_method: FluidIntegrateMethod) -> th.FloatOrArr:
    r"""
    Initial guess for root-finding of $\alpha_+$ from $\alpha_n$.
    Linear approx between $\alpha_{n,\min}$ and $\alpha_{n,\max}$.
    Doesn't do obvious checks like Detonation - needs improving?

    :param v_wall: $v_\text{wall}$, wall speed
    :param alpha_n_given: $\alpha_{n, \text{given}}$
    :param df_dtau_ptr: pointer to the differential equation function
    :param ode_method: differential equation solver to be used
    :return: initial guess for $\alpha_+$
    """
    if alpha_n_given < 0.05:
        return alpha_n_given

    alpha_plus_min = alpha_plus_min_hybrid(v_wall)
    alpha_plus_max = 1/3

    alpha_n_min = alpha_n_min_hybrid_bag(v_wall)
    alpha_n_max = alpha_n_max_deflagration_bag(v_wall, df_dtau_ptr=df_dtau_ptr, ode_method=ode_method)

    slope = (alpha_plus_max - alpha_plus_min) / (alpha_n_max - alpha_n_min)
    return alpha_plus_min + slope * (alpha_n_given - alpha_n_min)
