"""
Approximate reverse
===================

Find thermodynamic parameters for a given peak of the gravitational wave spectrum.
This is very much work in progress and not yet functional.
"""

import typing as tp

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from pttools.omgw0.approx import f0_peak_approx, omgw_approx


def solvable(params: np.ndarray, f_peak_target: float, omega_peak_target: float, temp: float, g_star: float) -> float:
    alpha, kappa_v, r_star = params
    f0_peak = f0_peak_approx(temp, r_star=r_star, g_star=g_star)
    omega_peak = omgw_approx(f=f0_peak, alpha=alpha, kappa_v=kappa_v, r_star=r_star, temp=temp, g_star=g_star, f0_peak=f0_peak)
    f_diff = np.log10(f0_peak) - np.log10(f_peak_target)
    omega_diff = np.log10(omega_peak) - np.log10(omega_peak_target)
    diff = f_diff**2 + omega_diff**2
    print(f"alpha={alpha}, kappa_v={kappa_v}, r_star={r_star}, f_diff={f_diff}, omega_diff={omega_diff}, diff={diff}")
    return diff


def solver(
        f_peak_target: float,
        omega_peak_target: float,
        v_wall_guess: float,
        alpha_n_guess: float,
        r_star_guess: float,
        temp: float,
        g_star: float) -> tp.Tuple[tp.Optional[float], tp.Optional[float], tp.Optional[float], float]:
    # The limits for v_wall and alpha_n come from the limits of the suppression data.
    x0 = np.array([v_wall_guess, alpha_n_guess, r_star_guess])
    sol: OptimizeResult = minimize(
        solvable,
        x0=x0,
        args=(f_peak_target, omega_peak_target, temp, g_star),
        bounds=((1e-3, 0.1), (1e-4, 0.999), (1e-4, 0.1))
    )
    err = solvable(sol.x, f_peak_target, omega_peak_target, temp, g_star)
    if not sol.success:
        return None, None, None, err
    return *sol.x, err


def main():
    print("Starting solver")
    temp = 100  # GeV
    g_star = 100
    ret = solver(
        f_peak_target=6e-3, omega_peak_target=3e-11,
        # f_peak_target = 1e-3, omega_peak_target=3e-11,
        # f_peak_target = 6e-3, omega_peak_target=1e-11,
        v_wall_guess=0.5, alpha_n_guess=0.05, r_star_guess=0.1,
        temp=temp, g_star=g_star
    )
    print("Results:")
    alpha  = ret[0]
    kappa_v = ret[1]
    r_star = ret[2]
    f0_peak = f0_peak_approx(temp, r_star=r_star, g_star=g_star)
    omega_peak = omgw_approx(f=f0_peak, alpha=alpha, kappa_v=kappa_v, r_star=r_star, temp=temp, g_star=g_star, f0_peak=f0_peak)
    print(f"v_wall={ret[0]}, alpha_n={ret[1]}, r_star={ret[2]}, f0_peak={f0_peak}, omega_peak={omega_peak}, diff={ret[3]}")


if __name__ == "__main__":
    main()
