"""
Approximations for the kinetic energy fraction $K$

@author: chloeg
"""

import numpy as np

from pttools.bubble.chapman_jouguet import v_chapman_jouguet_bag
import pttools.type_hints as th


# Todo: the approximations come from Espinoza et al. 2010

def delta_k(alpha_n: th.FloatOrArr) -> th.FloatOrArr:
    return -0.9 * np.log(np.sqrt(alpha_n) / (1 + np.sqrt(alpha_n)))  # natural log


def k_a(xi_w: th.FloatOrArr, alpha_n: th.FloatOrArr) -> th.FloatOrArr:
    """
    Small wall speeds xi_w << cs
    """
    return xi_w**(6/5) * 6.9 * alpha_n/(1.36 - 0.037 * np.sqrt(alpha_n) + alpha_n)


def k_b(alpha_n: th.FloatOrArr) -> th.FloatOrArrNumba:
    """
    transition from subsonic to supersonic deflagrations
    xi_w = cs
    """
    return alpha_n ** (2/5) / (0.017 + (0.997 + alpha_n)**(2/5))


def k_c(alpha_n: th.FloatOrArr) -> th.FloatOrArr:
    """
    For Jouget detonations xi_w = xi_j
    """
    return np.sqrt(alpha_n) / (0.135 + (np.sqrt(0.98 + alpha_n)))


def k_d(alpha_n: th.FloatOrArr) -> th.FloatOrArr:
    r"""
    $$\kappa \approx \frac{\alpha_n}{0.73 + 0.083 \sqrt{\alpha_n} + \alpha_n}$$
    :notes:`\ ` eq. 7.44
    $\xi_w$ => 1 v. large wall speed
    """
    return alpha_n / (0.73 + 0.083 * np.sqrt(alpha_n) + alpha_n)


def calc_ke_frac(xi_w: float, alpha_n: float):
    cs = 1/np.sqrt(3)

    xi_j_val = v_chapman_jouguet_bag(alpha_n)

    if xi_w < cs:
        k_a_val = k_a(xi_w, alpha_n)
        k_b_val = k_b(alpha_n)
        k = cs**(11/5) * k_a_val * k_b_val / ((cs**(11/5) - xi_w**(11/5)) * k_b_val + xi_w * cs**(6/5) * k_a_val)
    elif xi_w == cs:
        k = k_b(alpha_n)
    elif cs < xi_w < xi_j_val:
        delta_k_val = delta_k(alpha_n)
        k_b_val = k_b(alpha_n)
        k_c_val = k_c(alpha_n)
        k = k_b_val + (xi_w - cs) * delta_k_val + ((xi_w - cs)/(xi_j_val - cs))**3 * (k_c_val - k_b -(xi_j_val - cs) * delta_k_val)
    elif xi_w == xi_j_val:
        k = k_c(alpha_n)
    elif xi_w > xi_j_val:
        k_c_val = k_c(alpha_n)
        k_d_val = k_d(alpha_n)
        k = ((xi_j_val - 1)**3 * xi_j_val**(5/2) * xi_w ** (-5/2) * k_c_val * k_d_val) / (((xi_j_val - 1)**3 - (xi_w-1)**3) * xi_j_val**(5/2) * k_c_val + (xi_w - 1)**3 * k_d_val)
    elif xi_w > 0.85:
        k = k_d(alpha_n)
    else:
        raise ValueError(f"Invalid xi_w={xi_w}")

    # :notes:`\ ` eq. 7.43 with $\delta_n = 0$ as is the case for the bag model
    return k * alpha_n / (1 + alpha_n)
