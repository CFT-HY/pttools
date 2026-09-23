"""
Experimental data structures based on numba.jitclass.

When implementing these, remove the corresponding code from ssm/spectrum.py

Jitclasses are a highly experimental feature of Numba. Please see the following issues.
https://github.com/numba/numba/issues/365
https://github.com/numba/numba/issues/2933
https://github.com/numba/numba/issues/4814
https://github.com/numba/numba/issues/6648

This has been replaced with the object-oriented Bubble interface and will probably be removed in the future.
"""

import numba

# import numpy as np
from pttools import speedup


@speedup.jitclass([
    ("a", numba.float64)
])
class NucArgs:
    """Nucleation arguments."""

    def __init__(self, a: float):
        self.a: float = a


# At present, Numba implements structs with Numpy records
# nuc_args_dt = np.dtype(["a", np.float64])
# np.record()


@speedup.jitclass([
    ("v_wall", numba.float64),
    ("alpha", numba.float64),
    ("nuc_type", numba.optional(numba.types.string)),
    # The class_type attribute is added by the jitclass decorator.
    ("nuc_args", NotImplemented if speedup.NUMBA_DISABLE_JIT else numba.optional(
        NucArgs.class_type.instance_type))  # pyrefly: ignore[missing-attribute]
])
class PhysicalParams:
    """Physical parameters for a bubble."""

    def __init__(
            self,
            v_wall: float,
            alpha: float,
            # The jitclass field is a string, and therefore NucType has to be given by its value when jitting.
            nuc_type: str | None = None,
            nuc_args: NucArgs | None = None):
        self.v_wall: float = v_wall
        self.alpha: float = alpha
        self.nuc_type: str | None = nuc_type
        self.nuc_args: NucArgs | None = nuc_args
