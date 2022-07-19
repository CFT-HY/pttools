r"""Constant sound speed model, aka. $\mu, \nu$ model"""

import logging

import numba
import numpy as np

import pttools.type_hints as th
from pttools.models.analytic import AnalyticModel

logger = logging.getLogger(__name__)


class ConstCSModel(AnalyticModel):
    r"""$\mu, \nu$-model

    .. plot:: fig/const_cs_xi_v.py

    .. plot:: fig/const_cs_p.py

    .. plot:: fig/const_cs_s.py

    """
    DEFAULT_NAME = "const_cs"

    def __init__(
            self,
            a_s: float, a_b: float,
            css2: float, csb2: float,
            V_s: float = 0, V_b: float = 0,
            t_min: float = 0,
            t_ref: float = 1,
            name: str = None):
        # Ensure that these descriptions correspond to those in the base class
        r"""
        :param a_s: prefactor of $p$ in the symmetric phase. The convention is as in :notes:`\ ` eq. 7.33.
        :param a_b: prefactor of $p$ in the broken phase. The convention is as in :notes:`\ ` eq. 7.33.
        :param css2: $c_{s,s}^2, speed of sound squared in the symmetric phase
        :param csb2: $c_{s,b}^2, speed of sound squared in the broken phase
        :param V_s: $V_s \equiv \epsilon_s$, the potential term of $p$ in the symmetric phase
        :param V_b: $V_b \equiv \epsilon_b$, the potential term of $p$ in the broken phase
        :param temp0: reference temperature, usually 1 * unit of choice, e,g. 1 GeV
        :param name: custom name for the model
        """
        if css2 > 1/3:
            raise ValueError(
                "C_{s,s}^2 has to be <= 1/3 for the solution to be physical. This is because g_eff is monotonic.")
        self.css2 = css2
        self.csb2 = csb2
        super().__init__(a_s=a_s, a_b=a_b, V_s=V_s, V_b=V_b, name=name)

        self.a_s = a_s
        self.a_b = a_b
        self.csb = np.sqrt(csb2)
        self.V_s = V_s
        self.V_b = V_b

        self.mu = 1 + 1/css2
        self.nu = 1 + 1/csb2

        self.cs2 = self.gen_cs2()

    def critical_temp_opt(self, temp: float):
        const = (self.V_b - self.V_s)**self.t_ref**4
        return self.a_s * (temp/self.t_ref)**self.mu - self.a_b * (temp/self.t_ref)**self.nu + const

    def gen_cs2(self):
        # These become compile-time constants
        css2 = self.css2
        csb2 = self.csb2

        @numba.njit
        def cs2(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            # Mathematical operations should be faster than conditional logic in compiled functions.
            return phase*csb2 + (1 - phase)*css2
        return cs2

    def e_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Energy density $e(T,\phi)$
        $$e_s = a_s (\mu - 1) T^\mu + V_s$$
        $$e_b = a_b (\nu - 1) T^\nu + V_b$$
        :giese_2021:`\ `, eq. 15.
        In the article there is a typo: the 4 there should be a $\mu$.
        """
        self.validate_temp(temp)
        e_s = self.a_s * (self.mu - 1) * temp**self.mu + self.V_s
        e_b = self.a_b * (self.nu - 1) * temp**self.nu + self.V_b
        return e_b * phase + e_s * (1 - phase)

    def p_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pressure $p(T,\phi)$
        $$p_s = a_s T^\mu - V_s$$
        $$p_b = a_b T^\nu - V_b$$
        :giese_2021:`\ `, eq. 15.
        """
        self.validate_temp(temp)
        p_s = self.a_s * temp**self.mu - self.V_s
        p_b = self.a_b * temp**self.nu - self.V_b
        return p_b * phase + p_s * (1 - phase)

    def s_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Entropy density $s=\frac{dp}{dT}$
        $$s_s = \mu a_s \left( \frac{T}{T_0} \right)^{\mu-1} T_0^3$$
        $$s_b = \nu a-b \left( \frac{T}{T_0} \right)^{\nu-1} T_0^3$$
        Derived from :giese_2021:`\ `, eq. 15.
        """
        self.validate_temp(temp)
        s_s = self.mu * self.a_s * (temp/self.t_ref)**(self.mu-1) * self.t_ref**3
        s_b = self.nu * self.a_b * (temp/self.t_ref)**(self.nu-1) * self.t_ref**3
        return s_b * phase + s_s * (1 - phase)

    def temp(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Temperature $T(w,\phi)$. Inverted from the equation of $w(T,\phi)$.
        $$T_s = T_0 \left( \frac{3w}{\mu a_s T_0^4} \right)^\frac{1}{\mu}$$
        $$T_b = T_0 \left( \frac{3w}{\nu a_s T_0^4} \right)^\frac{1}{\nu}$$
        """
        temp_s = self.t_ref * (3*w / (self.mu*self.a_s*self.t_ref**4))**(1/self.mu)
        temp_b = self.t_ref * (3*w / (self.nu*self.a_b*self.t_ref**4))**(1/self.nu)
        return temp_b * phase + temp_s * (1 - phase)

    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Enthalpy density $w(T,\phi)$
        $$w_s = \mu a_s \left( \frac{T}{T_0} \right)^\mu T_0^4$$
        $$w_s = \nu a_s \left( \frac{T}{T_0} \right)^\nu T_0^4$$
        """
        self.validate_temp(temp)
        w_s = self.mu * self.a_s * (temp/self.t_ref)**self.mu * self.t_ref**4
        w_b = self.nu * self.a_b * (temp/self.t_ref)**self.nu * self.t_ref**4
        return w_b * phase + w_s * (1 - phase)
