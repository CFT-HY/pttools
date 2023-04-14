import logging
import typing as tp

import numpy as np
from scipy.optimize import fsolve

logger = logging.getLogger(__name__)


def fsolve_vary(
        func: callable,
        x0: np.ndarray,
        args: tp.Optional[tp.Union[tp.Iterable, tuple]] = None,
        abs_variations: tp.Union[float, np.ndarray] = 1e-3,
        rel_variations: tp.Union[float, np.ndarray] = 0.01,
        log_status: bool = True,
        **kwargs) -> tp.Tuple[np.ndarray, dict, int, str]:
    """SciPy fsolve, but if it fails, it tries to vary the initial guess to find a solution"""
    if "full_output" in kwargs:
        raise ValueError("Cannot specify full_output, as it has to be True.")

    # Solve directly
    sol = fsolve(func, x0=x0, args=args, full_output=True, **kwargs)
    if sol[2] == 1:
        return sol

    # Vary the initial guess
    scalar_rel_var = np.isscalar(rel_variations)
    scalar_abs_var = np.isscalar(abs_variations)
    for i in range(x0.shape[0]):
        for sign in (1, -1):
            x0_var = x0.copy()
            if scalar_rel_var:
                x0_var[i] *= 1 + sign*rel_variations
            else:
                x0_var[i] *= 1 + sign*rel_variations[i]

            if scalar_abs_var:
                x0_var[i] += sign * abs_variations
            else:
                x0_var[i] += sign * abs_variations[i]

            sol2 = fsolve(func, x0=x0_var, args=args, full_output=True, **kwargs)
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
