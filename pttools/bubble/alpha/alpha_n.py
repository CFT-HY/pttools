r"""$\alpha_n$ functions"""

from pttools.bubble import props
from pttools.speedup import njit
import pttools.type_hints as th


@njit
def find_alpha_n_from_w_xi(w: th.FloatArr1D, xi: th.FloatArr1D, v_wall: float, alpha_p: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    Calculates the transition strength parameter with
    $$\alpha_n = \frac{w_+}{w_n} \alpha_p$$.

    Model-independent.

    :param w: $w$ array of a bubble
    :param xi: $xi$ array of a bubble
    :param v_wall: $v_\text{wall}$
    :param alpha_p: $\alpha_+$
    :return: $\alpha_n$
    """
    n_wall = props.find_v_index(xi, v_wall)
    wn = w[-1]
    return w[n_wall] / wn * alpha_p
