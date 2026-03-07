"""Base components for fluid shell solvers"""

import numpy as np

from pttools.bubble.solution_type import SolutionType
from pttools.speedup import NAN_ARR
import pttools.type_hints as th

# The output consists of:
# v, w, xi
# vp, vm, vp_tilde, vm_tilde, v_sh, vm_sh, vm_tilde_sh, wp, wm, wm_sh
# solution_found
type GenericSolverOutput = tuple[
    th.FloatArr1D, th.FloatArr1D, th.FloatArr1D, SolutionType,
    float, float, float, float, float, float, float, float, float, float,
    float, bool, float
]
type SolverOutput = tuple[
    th.FloatArr1D, th.FloatArr1D, th.FloatArr1D,
    float, float, float, float, float, float, float, float, float, float,
    bool
]
type DeflagrationOutput = tuple[
    th.FloatArr1D, th.FloatArr1D, th.FloatArr1D,
    float, float, float, float, float, float, float, float, float, float
]
DEFLAGRATION_NAN: DeflagrationOutput = \
    NAN_ARR, NAN_ARR, NAN_ARR, \
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
