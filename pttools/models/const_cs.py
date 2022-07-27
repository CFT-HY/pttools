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
    DEFAULT_LABEL = "Constant $c_s$ model"
    DEFAULT_NAME = "const_cs"

    def __init__(
            self,
            css2: float, csb2: float,
            V_s: float, V_b: float = 0,
            a_s: float = None, a_b: float = None,
            g_s: float = None, g_b: float = None,
            t_min: float = None,
            t_max: float = None,
            t_ref: float = 1,
            name: str = None,
            label: str = None):
        # Ensure that these descriptions correspond to those in the base class
        r"""
        :param a_s: prefactor of $p$ in the symmetric phase. The convention is as in :notes:`\ ` eq. 7.33.
        :param a_b: prefactor of $p$ in the broken phase. The convention is as in :notes:`\ ` eq. 7.33.
        :param css2: $c_{s,s}^2, speed of sound squared in the symmetric phase
        :param csb2: $c_{s,b}^2, speed of sound squared in the broken phase
        :param V_s: $V_s \equiv \epsilon_s$, the potential term of $p$ in the symmetric phase
        :param V_b: $V_b \equiv \epsilon_b$, the potential term of $p$ in the broken phase
        :param t_ref: reference temperature, usually 1 * unit of choice, e,g. 1 GeV
        :param name: custom name for the model
        """
        if css2 > 1/3 or csb2 > 1/3:
            raise ValueError(
                "c_{s,s}^2 and c_{s,b}^2 have to be <= 1/3 for the solution to be physical. "
                "This is because g_eff is monotonic. "
                f"Got: c_{{s,s}}^2={css2}, c_{{s,b}}^2={csb2}."
            )

        self.css2 = css2
        self.csb2 = csb2
        self.css = np.sqrt(css2)
        self.csb = np.sqrt(csb2)
        self.mu = 1 + 1 / css2
        self.nu = 1 + 1 / csb2
        self.t_ref = t_ref

        super().__init__(
            V_s=V_s, V_b=V_b,
            a_s=a_s, a_b=a_b,
            g_s=g_s, g_b=g_b,
            t_min=t_min, t_max=t_max,
            name=name, label=label
        )

        self.const_cs_wn_const: float = 1 / 3 * (
            (1 - 4 / self.mu) * self.t_ref ** (self.mu - 4)
            - (1 - 4 / self.nu) * self.t_ref ** (self.nu - 4)
        )

    def critical_temp_opt(self, temp: float) -> float:
        const = (self.V_b - self.V_s)*self.t_ref**4
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
        e_s = (self.mu - 1) * self.a_s * temp**self.mu + self.V_s
        e_b = (self.nu - 1) * self.a_b * temp**self.nu + self.V_b
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
        $$T_s = T_0 \left( \frac{w}{\mu a_s T_0^4} \right)^\frac{1}{\mu}$$
        $$T_b = T_0 \left( \frac{w}{\nu a_s T_0^4} \right)^\frac{1}{\nu}$$
        """
        temp_s = self.t_ref * (w / (self.mu*self.a_s*self.t_ref**4))**(1/self.mu)
        temp_b = self.t_ref * (w / (self.nu*self.a_b*self.t_ref**4))**(1/self.nu)
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

    def w_n(
            self,
            alpha_n: th.FloatOrArr,
            wn_guess: float = None,
            allow_negative: bool = False,
            analytical: bool = True) -> th.FloatOrArr:
        r"""Enthalpy at nucleation temperature
        $$w_n = \frac{b}{\alpha_n - a}$$
        where
        $$a = \frac{4}{3} \frac{V_s - V_b}$$
        $$b = \frac{1}{3\alpha_n} \left[
        (1-\frac{4}{\mu}T_0^{\mu-4}
        - (1-\frac{4}{\nu}T_0^{\nu-4}
        \right]$$
        This can be derived from the equations for $\theta$ and $\alpha_n$.
        """
        diff = alpha_n - self.const_cs_wn_const
        if np.any(diff < 0):
            if np.isscalar(alpha_n):
                info = f"Got: wn={alpha_n}."
            else:
                i = np.argmin(diff)
                info = f"Most problematic values: wn={alpha_n[i]}, diff={diff[i]}."
            msg = f"Got too small alpha_n for the model \"{self.name}\". {info}"
            logger.error(msg)
            if not allow_negative:
                ValueError(msg)

        if not analytical:
            super().w_n(alpha_n, wn_guess)
        return self.bag_wn_const / (alpha_n - self.const_cs_wn_const)
