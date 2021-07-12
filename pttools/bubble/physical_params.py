"""
Experimental data structures based on numba.jitclass

TODO: when implementing these, remove the corresponding code from ssmtools/spectrum.py

Jitclasses are a highly experimental feature of Numba. Please see the following issues.
https://github.com/numba/numba/issues/365
https://github.com/numba/numba/issues/2933
https://github.com/numba/numba/issues/4814
https://github.com/numba/numba/issues/6648
"""

import enum

import numba
# import numpy as np

from pttools import speedup


@enum.unique
class NucType(str, enum.Enum):
    """Nucleation type"""
    EXPONENTIAL = "exponential"
    SIMULTANEOUS = "simultaneous"


DEFAULT_NUC_TYPE = NucType.EXPONENTIAL


@numba.experimental.jitclass([
    ("a", numba.float64)
])
class NucArgs:
    def __init__(self, a: float):
        self.a = a


# At present, Numba implements structs with Numpy records
# nuc_args_dt = np.dtype(["a", np.float64])
# np.record()


@numba.experimental.jitclass([
    ("v_wall", numba.float64),
    ("alpha", numba.float64),
    ("nuc_type", numba.optional(numba.types.string)),
    ("nuc_args", NotImplemented if speedup.NUMBA_DISABLE_JIT else numba.optional(NucArgs.class_type.instance_type))
])
class PhysicalParams:
    def __init__(self, v_wall: float, alpha: float, nuc_type: NucType = None, nuc_args: NucArgs = None):
        self.v_wall = v_wall
        self.alpha = alpha
        self.nuc_type = nuc_type
        self.nuc_args = nuc_args
