r"""Constant sound speed model, aka. $\mu, \nu$ model"""

import pttools.type_hints as th
from .model import Model
from .thermo import ThermoModel

import numba
import numpy as np

import logging

logger = logging.getLogger(__name__)


class ConstCSThermoModel(ThermoModel):
    # TODO: Should I override ThermoModel with const_cs analytic expressions here?
    pass


class ConstCSModel(Model):
    r"""$\mu, \nu$-model

    .. plot:: fig/const_cs_model.py

    """
    def __init__(
            self,
            a_s: float, a_b: float,
            css2: float, csb2: float,
            eps_s: float = 0, eps_b: float = 0,
            temp0: float = 1):
        if css2 > 1/3:
            raise ValueError(
                "C_{s,s}^2 has to be <= 1/3 for the solution to be physical. This is because g_eff is monotonic.")
        super().__init__(thermo=None)

        self.temp0 = temp0
        self.a_s = a_s
        self.a_b = a_b
        self.css2 = css2
        self.csb = np.sqrt(csb2)
        self.csb2 = csb2
        self.eps_s = eps_s
        self.eps_b = eps_b

        self.mu = 1 + 1/css2
        self.nu = 1 + 1/csb2

        self.cs2 = self.gen_cs2()

    def gen_cs2(self):
        # These become compile-time constants
        css2 = self.css2
        csb2 = self.csb2

        @numba.njit
        def cs2(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            # TODO: How about the w-dependence? Probably there shouldn't be any, as the speed of sound
            # is constant in ech phase. But how about when the phase is not known?
            # Mathematical operations should be faster than conditional logic in compiled functions.
            return phase*csb2 + (1 - phase)*css2
        return cs2

    # TODO: make these take in enthalpy instead

    def p(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        # TODO: as e
        pass

    # def p_temp(self, temp: th.FloatOrArr, phase: Phase) -> th.FloatOrArr:
    #     if phase == Phase.SYMMETRIC:
    #         return 1/3 * self.a_s * temp**self.mu - self.eps
    #     if phase == Phase.BROKEN:
    #         return 1/3 * self.a_b * temp**self.nu
    #     raise ValueError(f"Unknown phase: {phase}")

    def e(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        # TODO: make this take in w instead
        # There is a typo in :giese_2021:. The 4 there should be a mu.
        e_s = 1/3 * self.a_s * (self.mu - 1) * temp**self.mu + self.eps_s
        e_b = 1/3 * self.a_b * (self.nu - 1) * temp**self.nu + self.eps_b
        return e_b * phase + e_s * (1 - phase)

    # TODO: add factors of 1/3

    def temp(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        temp_s = self.temp0 * (w/(self.mu*self.a_s*self.temp0**4))**(1/self.mu)
        temp_b = self.temp0 * (w/(self.nu*self.a_b*self.temp0**4))**(1/self.nu)
        return temp_b * phase + temp_s * (1 - phase)

    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        w_s = self.mu * self.a_s * (temp/self.temp0)**self.mu * self.temp0**4
        w_b = self.nu * self.a_b * (temp/self.temp0)**self.nu * self.temp0**4
        return w_b * phase + w_s * (1 - phase)
