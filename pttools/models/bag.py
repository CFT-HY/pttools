"""Bag model"""

import numba

import pttools.type_hints as th
from pttools.models.analytic import AnalyticModel


class BagModel(AnalyticModel):
    r"""Bag equation of state.
    This is one of the simplest equations of state for a relativistic plasma.
    Each integration corresponds to a line on the figure below (fig. 9 of :gw_pt_ssm:`\ `).

    .. plot:: fig/xi_v_plane.py

    """
    DEFAULT_NAME = "bag"

    def critical_temp(self, guess: float) -> float:
        # TODO
        raise NotImplementedError

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
        return (w / (4*(self.a_b*phase + self.a_s*(1-phase))))**(1/4)

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
