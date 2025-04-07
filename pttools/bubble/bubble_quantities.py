import numpy as np

from pttools.bubble.bubble import Bubble


def get_ke_frac(bubble: Bubble):
    if bubble.solved:
        return bubble.kinetic_energy_fraction
    return np.nan

get_ke_frac.return_type = float
get_ke_frac.fail_value = np.nan


def get_kappa(bubble: Bubble) -> float:
    if (not bubble.solved) or bubble.no_solution_found or bubble.solver_failed or bubble.numerical_error:
        return np.nan
    return bubble.kappa

get_kappa.return_type = float
get_kappa.fail_value = np.nan


def get_kappa_giese(bubble: Bubble) -> float:
    if (not bubble.solved) or bubble.no_solution_found or bubble.solver_failed or bubble.numerical_error:
        return np.nan
    return bubble.kappa_giese

get_kappa_giese.return_type = float
get_kappa_giese.fail_value = np.nan


def get_kappa_omega(bubble: Bubble):
    if bubble.no_solution_found or bubble.solver_failed:
        return np.nan, np.nan
    return bubble.kappa, bubble.omega

get_kappa_omega.fail_value = (np.nan, np.nan)
get_kappa_omega.return_type = (float, float)
