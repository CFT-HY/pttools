"""Functions for getting various quantities from Bubble objects

These are useful for parallel processing where one should do
as much of the processing in the parallel execution as possible.
"""

import typing as tp

import numpy as np

from pttools.bubble.bubble import Bubble
import pttools.type_hints as th


def get_ke_frac(bubble: Bubble) -> float:
    """Get kinetic energy fraction $K$ of a Bubble"""
    if bubble.solved:
        return bubble.kinetic_energy_fraction
    return np.nan

get_ke_frac.return_type = float
get_ke_frac.fail_value = np.nan


def get_kappa(bubble: Bubble) -> float:
    r"""Get $\kappa$ of a Bubble"""
    if (not bubble.solved) or bubble.no_solution_found or bubble.solver_failed or bubble.numerical_error:
        return np.nan
    return bubble.kappa

get_kappa.return_type = float
get_kappa.fail_value = np.nan


def get_kappa_for_v_walls(params: np.ndarray[tuple[int], tp.Any], v_walls: th.FloatArr1D) -> th.FloatArr1D:
    r"""Get $\kappa({v}_\text{wall})$ for the given parameters"""
    # Todo: replace the uses of this with a solution that uses get_kappa() instead.
    model, alpha_n = params
    kappas = np.full_like(v_walls, np.nan)
    for i, v_wall in enumerate(v_walls):
        try:
            kappas[i] = Bubble(model, v_wall=v_wall, alpha_n=alpha_n).kappa
        except (IndexError, ValueError, RuntimeError):
            continue
    return kappas


def get_kappa_giese(bubble: Bubble) -> float:
    r"""Get $\kappa$ of a Bubble as defined by :giese_2021:`\ `"""
    if (not bubble.solved) or bubble.no_solution_found or bubble.solver_failed or bubble.numerical_error:
        return np.nan
    return bubble.kappa_giese

get_kappa_giese.return_type = float
get_kappa_giese.fail_value = np.nan


def get_kappa_omega(bubble: Bubble) -> tuple[float, float]:
    r"""Get both $\kappa$ and $\omega$ of a Bubble"""
    if bubble.no_solution_found or bubble.solver_failed:
        return np.nan, np.nan
    return bubble.kappa, bubble.omega

get_kappa_omega.fail_value = (np.nan, np.nan)
get_kappa_omega.return_type = (float, float)
