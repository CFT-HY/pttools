"""Type hints for simplifying and unifying PTtools code"""

import typing as tp

import numpy as np

FLOAT_OR_ARR = tp.Union[float, np.ndarray]
FLOAT_OR_ARR_NUMBA = tp.Union[float, np.ndarray, callable]
INT_OR_ARR = tp.Union[int, np.ndarray]
