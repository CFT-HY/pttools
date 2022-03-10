r"""Constant sound speed model, aka. $\mu, \nu$ model"""

import pttools.type_hints as th
from .model import Model

import numba
import numpy as np

import logging

logger = logging.getLogger(__name__)


# class ConstCSThermoModel(ThermoModel):
#     """The constant sound speed model simplifies the thermodynamics a lot, and therefore needs its own ThermoModel"""


class ConstCSModel(Model):
    r"""$\mu, \nu$-model

    .. plot:: fig/const_cs_xi_v.py

    .. plot:: fig/const_cs_p.py

    .. plot:: fig/const_cs_s.py

    """
    def __init__(
            self,
            a_s: float, a_b: float,
            css2: float, csb2: float,
            V_s: float = 0, V_b: float = 0,
            temp0: float = 1):
        if css2 > 1/3:
            raise ValueError(
                "C_{s,s}^2 has to be <= 1/3 for the solution to be physical. This is because g_eff is monotonic.")
        self.css2 = css2
        self.csb2 = csb2
        super().__init__(thermo=None)

        self.temp0 = temp0
        self.a_s = a_s
        self.a_b = a_b
        self.csb = np.sqrt(csb2)
        self.eps_s = V_s
        self.eps_b = V_b

        self.mu = 1 + 1/css2
        self.nu = 1 + 1/csb2

        self.cs2 = self.gen_cs2()

    def critical_temp_opt(self, temp: float):
        const = (self.V_b - self.V_s)**self.temp0**4
        return self.a_s * (temp/self.temp0)**self.mu - self.a_b * (temp/self.temp0)**self.nu + const

    def gen_cs2(self):
        # These become compile-time constants
        css2 = self.css2
        csb2 = self.csb2

        @numba.njit
        def cs2(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            # Mathematical operations should be faster than conditional logic in compiled functions.
            return phase*csb2 + (1 - phase)*css2
        return cs2

    def e(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return self.e_temp(self.temp(w, phase), phase)

    def e_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Energy density $e(T,\phi)$

        There is a typo in :giese_2021:. The 4 there should be a mu."""
        e_s = 1/3 * self.a_s * (self.mu - 1) * temp**self.mu + self.eps_s
        e_b = 1/3 * self.a_b * (self.nu - 1) * temp**self.nu + self.eps_b
        return e_b * phase + e_s * (1 - phase)

    def p(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return self.p_temp(self.temp(w, phase), phase)

    def p_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        p_s = 1/3 * self.a_s * temp**self.mu - self.V_s
        p_b = 1/3 * self.a_b * temp**self.nu - self.V_b
        return p_b * phase + p_s * (1 - phase)

    def s_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Entropy density $s=\frac{dp}{dT}(T,\phi)$"""
        s_s = 1/3 * self.mu * self.a_s * (temp/self.temp0)**(self.mu-1) * self.temp0**3
        s_b = 1/3 * self.nu * self.a_b * (temp/self.temp0)**(self.nu-1) * self.temp0**3
        return s_b * phase + s_s * (1 - phase)

    def temp(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        temp_s = self.temp0 * (3*w / (self.mu*self.a_s*self.temp0**4))**(1/self.mu)
        temp_b = self.temp0 * (3*w / (self.nu*self.a_b*self.temp0**4))**(1/self.nu)
        return temp_b * phase + temp_s * (1 - phase)

    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        w_s = 1/3 * self.mu * self.a_s * (temp/self.temp0)**self.mu * self.temp0**4
        w_b = 1/3 * self.nu * self.a_b * (temp/self.temp0)**self.nu * self.temp0**4
        return w_b * phase + w_s * (1 - phase)
