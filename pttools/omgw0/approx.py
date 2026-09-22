r"""Approximations for $\Omega_{\text{gw},0}$."""

import math

import numpy as np

from pttools.omgw0 import const
from pttools.type_hints import FloatOrArr


def omgw_approx[T: FloatOrArr](
        f: T,
        alpha: T | float,
        kappa_v: T | float,
        r_star: T | float,
        temp: T | float = const.DEFAULT_T_STAR,
        g_star: T | float = const.DEFAULT_G_STAR,
        f0_peak: T | float | None = None) -> T:
    r""":caprini_2016:`\ ` eq. 13."""
    # Todo: this function is missing a factor of h^2
    # return 2.65e-6 * H_star * beta * ((kappa_v * alpha) / (1 + alpha)) * (100 / g_star)**(1/3) * v_w * S_sw(f)
    if f0_peak is None:
        f0_peak = f0_peak_approx(temp, r_star, g_star)
    return \
        2.65e-6 * (8*np.pi)**(-1/3) * r_star * ((kappa_v * alpha) / (1 + alpha)) * (100 / g_star)**(1/3) * \
        S_sw_approx(f, f0_peak)


def S_sw_approx[T: FloatOrArr](f: T, f_peak: T | float) -> T:
    r""":caprini_2016:`\ ` eq. 14."""
    f_relative = f / f_peak
    return f_relative**3 * (7 / (4 + 3 * f_relative**2)) ** (7/2)  # pyrefly: ignore[bad-return]


def f_peak_approx[T: FloatOrArr](v_wall: T, beta: T | float) -> T:
    return 2 * beta / (math.sqrt(3) * v_wall)  # pyrefly: ignore[bad-return]


def f0_peak_approx[T: FloatOrArr](temp: T, r_star: T | float, g_star: T | float) -> T:
    r""":caprini_2016:`\ ` eq. 15."""
    # return 1.9e-5 * beta * temp / (v_wall * H_star * 100) * (g_star / 100) ** (1/6)
    return 1.9e-5 * (8 * np.pi)**(1/3) / r_star * temp / 100 * (g_star / 100) ** (1/6)
