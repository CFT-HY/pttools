"""
Reverse
=======

Find thermodynamic parameters for a given peak of the gravitational wave spectrum.
This is very much work in progress and not yet functional.
"""

import typing as tp

import numpy as np
from scipy.optimize import minimize, OptimizeResult

from pttools.bubble import Bubble
from pttools.models import BagModel, Model
from pttools.omgw0 import Spectrum, SuppressionMethod


def solvable(params: np.ndarray, model: Model, f_peak_target: float, omega_peak_target: float) -> tp.Tuple[float, float]:
    v_wall, alpha_n, r_star = params
    bubble = Bubble(model, v_wall=v_wall, alpha_n=alpha_n)
    spectrum = Spectrum(bubble, r_star=r_star)
    f_peak, omega_peak = spectrum.omgw0_peak(suppression=SuppressionMethod.EXT_CONSTANT)
    f_diff = np.log10(f_peak) - np.log10(f_peak_target)
    omega_diff = np.log10(omega_peak) - np.log10(omega_peak_target)
    diff = f_diff**2 + omega_diff**2
    print(f"v_wall={v_wall}, alpha_n={alpha_n}, r_star={r_star}, f_diff={f_diff}, omega_diff={omega_diff}, diff={diff}")
    return diff


def solver(
        model: Model,
        f_peak_target: float,
        omega_peak_target: float,
        v_wall_guess: float,
        alpha_n_guess: float,
        r_star_guess: float,
        alpha_n_max: float = 0.66):
    # The limits for v_wall and alpha_n come from the limits of the suppression data.
    x0 = np.array([v_wall_guess, alpha_n_guess, r_star_guess])
    sol: OptimizeResult = minimize(
        solvable,
        x0=x0,
        args=(model, f_peak_target, omega_peak_target),
        bounds=((0.25, 0.95), (0.06, alpha_n_max), (1e-3, 1))
    )
    err = solvable(sol.x, model, f_peak_target, omega_peak_target)
    if not sol.success:
        return None, None, None, err
    return *sol.x, err


def main():
    model = BagModel(alpha_n_min=0.01)
    print("Starting solver")
    ret = solver(
        model, f_peak_target=6e-3, omega_peak_target=3e-11,
        v_wall_guess=0.5, alpha_n_guess=0.05, r_star_guess=0.1,
        alpha_n_max=0.1
    )
    print("Results:")
    print(f"v_wall={ret[0]}, alpha_n={ret[1]}, r_star={ret[2]}, diff={ret[3]}")


if __name__ == "__main__":
    main()
