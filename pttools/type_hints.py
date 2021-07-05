"""Type hints for simplifying and unifying PTtools code"""

import typing as tp

from numba.core.registry import CPUDispatcher
import numpy as np

NUMBA_FUNC = tp.Union[callable, CPUDispatcher]

FLOAT_OR_ARR = tp.Union[float, np.ndarray]
FLOAT_OR_ARR_NUMBA = tp.Union[float, np.ndarray, NUMBA_FUNC]
INT_OR_ARR = tp.Union[int, np.ndarray]
