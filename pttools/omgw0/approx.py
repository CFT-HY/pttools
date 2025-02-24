import pttools.type_hints as th

import math
import numpy as np

from . import const


def omgw_approx(
        f: th.FloatOrArr,
        alpha: th.FloatOrArr,
        kappa_v: th.FloatOrArr,
        r_star: th.FloatOrArr,
        temp: th.FloatOrArr = const.T_default,
        g_star: th.FloatOrArr = const.G_STAR_DEFAULT,
        f0_peak: th.FloatOrArr = None) -> th.FloatOrArr:
    # return 2.65e-6 * H_star * beta * ((kappa_v * alpha) / (1 + alpha)) * (100 / g_star)**(1/3) * v_w * S_sw(f)
    if f0_peak is None:
        f0_peak = f0_peak_approx(temp, r_star, g_star)
    return \
        2.65e-6 * (8*np.pi)**(-1/3) * r_star * ((kappa_v * alpha) / (1 + alpha)) * (100 / g_star)**(1/3) * \
        S_sw_approx(f, f0_peak)


def S_sw_approx(f: th.FloatOrArr, f_peak: th.FloatOrArr) -> th.FloatOrArr:
    f_relative = f / f_peak
    return f_relative**3 * (7 / (4 + 3 * f_relative**2)) ** (7/2)


def f_peak_approx(v_wall: th.FloatOrArr, beta: th.FloatOrArr) -> th.FloatOrArr:
    return 2 * beta / (math.sqrt(3) * v_wall)


def f0_peak_approx(temp: th.FloatOrArr, r_star: th.FloatOrArr, g_star: th.FloatOrArr) -> th.FloatOrArr:
    # return 1.9e-5 * beta * temp / (v_wall * H_star * 100) * (g_star / 100) ** (1/6)
    return 1.9e-5 * (8 * np.pi)**(1/3) / r_star * temp / 100 * (g_star / 100) ** (1/6)


def R_star(v_wall: th.FloatOrArr, beta: th.FloatOrArr) -> th.FloatOrArr:
    return (8 * np.pi)**(1/3) * v_wall / beta
