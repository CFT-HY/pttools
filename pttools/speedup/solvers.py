"""Equation solvers that improve upon the ones available in SciPy."""

import logging
import typing as tp

import numpy as np
from scipy.optimize import fsolve

import pttools.type_hints as th

logger = logging.getLogger(__name__)


def fsolve_vary(
        func: tp.Callable,
        x0: th.FloatArr,
        args: tuple = (),
        abs_variations: float | th.FloatArr1D = 1e-3,
        rel_variations: float | th.FloatArr1D = 0.01,
        log_status: bool = True,
        **kwargs) -> th.FSolveOutput:
    """SciPy fsolve, but if it fails, it tries to vary the initial guess to find a solution."""
    if "full_output" in kwargs:
        raise ValueError("Cannot specify full_output, as it has to be True.")

    # Solve directly
    sol: th.FSolveOutput = fsolve(func, x0=x0, args=args, full_output=True, **kwargs)
    if sol[2] == 1:
        return sol

    # Vary the initial guess
    for i in range(x0.shape[0]):
        rel_var = rel_variations[i] if isinstance(rel_variations, np.ndarray) else rel_variations
        abs_var = abs_variations[i] if isinstance(abs_variations, np.ndarray) else abs_variations
        for sign in (1, -1):
            x0_var = x0.copy()
            x0_var[i] *= 1 + sign * rel_var
            x0_var[i] += sign * abs_var

            sol2: th.FSolveOutput = fsolve(
                func, x0=x0_var, args=args, full_output=True, **kwargs)
            if sol2[2] == 1:
                if log_status:
                    logger.debug("Solution was found by varying the initial guess from %s to %s", x0, x0_var)
                return sol2
    if log_status:
        logger.error(
            "Solution was not found directly, nor by varying the initial guess from %s with abs=%s, rel=%s",
            x0, abs_variations, rel_variations
        )
    return sol
