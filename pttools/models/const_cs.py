r"""Constant sound speed model, aka. $\mu, \nu$ model"""

import pttools.type_hints as th
from pttools.bubble.boundary import Phase
from pttools.bubble import const
from pttools.bubble import props
from .model import Model

import numba
import numpy as np

import logging

logger = logging.getLogger(__name__)


class ConstCSModel(Model):
    r"""$\mu, \nu$-model

    .. plot:: fig/const_cs_model.py

    """
    def __init__(self, a_s: float, a_b: float, css2: float, csb2: float, eps: float):
        super().__init__()
        self.a_s = a_s
        self.a_b = a_b
        self.css2 = css2
        self.csb = np.sqrt(csb2)
        self.csb2 = csb2
        self.eps = eps

        self.mu = 1 + 1/css2
        self.nu = 1 + 1/csb2

        self.cs2 = self.gen_cs2()

    def cs2(self, v: float, w: float, xi: float):
        raise NotImplementedError("Not yet loaded!")

    def gen_cs2(self):
        # These become compile-time constants
        css2 = self.css2
        csb2 = self.csb2

        @numba.njit
        def cs2(v: float, w: float, xi: float) -> float:
            if v > props.v_shock(xi):
                # Ahead of the wall
                return css2
            if v < props.v_max_behind(xi, csb2):
                # Behind the wall
                return csb2
            # Unphysical, so let's use the bag model
            return const.CS0_2
        return cs2

    def p(self, T: th.FloatOrArr, phase: Phase) -> th.FloatOrArr:
        if phase == Phase.SYMMETRIC:
            return 1/3 * self.a_s * T**self.mu - self.eps
        if phase == Phase.BROKEN:
            return 1/3 * self.a_b * T**self.nu
        raise ValueError(f"Unknown phase: {phase}")

    def e(self, T: th.FloatOrArr, phase: Phase) -> th.FloatOrArr:
        if phase == Phase.SYMMETRIC:
            # There is a typo in :giese_2021:. The 4 there should be a mu.
            return 1/3 * self.a_s * (self.mu - 1) * T**self.mu + self.eps
        if phase == Phase.BROKEN:
            return 1/3 * self.a_b * (self.nu - 1) * T**self.nu
        raise ValueError(f"Unknown phase: {phase}")
