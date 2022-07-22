"""Bag model"""

import logging

import numba

import pttools.type_hints as th
from pttools.models.analytic import AnalyticModel

logger = logging.getLogger(__name__)


class BagModel(AnalyticModel):
    r"""Bag equation of state.
    This is one of the simplest equations of state for a relativistic plasma.
    Each integration corresponds to a line on the figure below (fig. 9 of :gw_pt_ssm:`\ `).

    .. plot:: fig/xi_v_plane.py

    """
    DEFAULT_NAME = "bag"

    def __init__(
            self,
            V_s: float, V_b: float = 0,
            a_s: float = None, a_b: float = None,
            g_s: float = None, g_b: float = None,
            t_min: float = None, t_max: float = None,
            name: str = None):
        if V_b != 0:
            logger.warning("V_b has been specified for the bag model, even though it's usually omitted.")

        super().__init__(V_s=V_s, V_b=V_b, a_s=a_s, a_b=a_b, g_s=g_s, g_b=g_b, t_min=t_min, t_max=t_max, name=name)
        if self.a_s <= self.a_b:
            raise ValueError("The bag model must have a_s > a_b for the critical temperature to be non-negative.")

    def critical_temp(self, guess: float) -> float:
        r"""Critical temperature for the bag model

        $$T_{cr} = \sqrt[4]{\frac{3 V_s}{a_s - a_b}}$$
        :giese_2020:`\ ` p. 6
        """
        return (3 * self.V_s / (self.a_s - self.a_b))**0.25

    @staticmethod
    @numba.njit
    def cs2(w: th.FloatOrArr = None, phase: th.FloatOrArr = None):
        r"""Sound speed squared, $c_s^2=\frac{1}{3}$.
        :notes:`\ `, p. 37,
        :rel_hydro_book:`\ `, eq. 2.207
        """
        return 1/3

    def e_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Energy density as a function of temperature, :giese_2021:`\ ` eq. 15, :borsanyi_2016:`\ `, eq. S12
        The convention for $a_s$ and $a_b$ is that of :notes:`\ `, eq. 7.33.
        """
        self.validate_temp(temp)
        e_s = 3*self.a_s * temp**4
        e_b = 3*self.a_b * temp**4
        return e_b * phase + e_s * (1 - phase)

    def gen_cs2(self):
        return self.cs2

    def p_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pressure $p(T,\phi)$, :notes:`\ `, eq. 5.14, 7.1, 7.33, :giese_2021:`\ `, eq. 18
        $$p_s = a_s T^4$$
        $$p_b = a_b T^4$$
        The convention for $a_s$ and $a_b$ is that of :notes:`\ ` eq. 7.33.
        """
        self.validate_temp(temp)
        p_s = self.a_s * temp**4 - self.V_s
        p_b = self.a_b * temp**4 - self.V_b
        return p_b * phase + p_s * (1 - phase)

    def s_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Entropy density $s=\frac{dp}{dT}$
        $$s_s = 4 a_s T^3$$
        $$s_b = 4 a_b T^3$$
        Derived from :notes:`\ ` eq. 7.33.
        """
        self.validate_temp(temp)
        s_s = 4*self.a_s*temp**3
        s_b = 4*self.a_b*temp**3
        return s_b * phase + s_s * (1 - phase)

    def temp(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Temperature $T(w,\phi)$. Inverted from
        $$T(w) = \sqrt[4]{\frac{w}{4a(\phi)}}$$

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        :return: temperature $T(w,\phi)$
        """
        # return (w / (4*(self.a_b*phase + self.a_s*(1-phase))))**(1/4)
        # Defined in the same way as for ConstCSModel
        temp_s = (w / (4*self.a_s))**0.25
        temp_b = (w / (4*self.a_b))**0.25
        return temp_b * phase + temp_s * (1 - phase)

    @staticmethod
    def v_shock(xi: th.FloatOrArr) -> th.FloatOrArr:
        r"""Velocity at the shock, :gw_pt_ssm:`\ ` eq. B.17
        $$v_\text{sh}(\xi) = \frac{3\xi^22 - 1}{2\xi}$$
        """
        return (3*xi**2 - 1)/(2*xi)

    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Enthalpy $w(T)$
        $$w(T) = 4a(\phi)T^4$$

        :param temp: temperature $T$
        :param phase: phase $\phi$
        """
        self.validate_temp(temp)
        return 4 * (self.a_b * phase + self.a_s * (1-phase))**temp**4

    @staticmethod
    def w_shock(xi: th.FloatOrArr, w_n: th.FloatOrArr) -> th.FloatOrArr:
        r"""Enthalpy at the shock, :gw_pt_ssm:`\ ` eq. B.18
        $$w_\text{sh}(\xi) = w_n \frac{9\xi^2 - 1}{3(1-\xi^2)}$$
        """
        return w_n * (9*xi**2 - 1)/(2*(1-xi**2))
