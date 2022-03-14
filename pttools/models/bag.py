"""Bag model"""

import numba

import pttools.type_hints as th
from pttools.bubble.boundary import Phase
from pttools.models.model import Model


class BagModel(Model):
    r"""Bag equation of state.
    Each integration corresponds to a line on the figure below (fig. 9 of :gw_pt_ssm:`\ `).

    .. plot:: fig/xi_v_plane.py

    """
    def __init__(self, a_s: float, a_b: float, V_s: float, V_b: float = 0):
        super().__init__(thermo=None, V_s=V_s, V_b=V_b)
        self.a_s = a_s
        self.a_b = a_b

    @staticmethod
    @numba.njit
    def cs2(w: th.FloatOrArr = None, phase: float = None):
        return 1/3

    def e(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        """:giese_2021: eq. 2, :borsanyi_2016: eq. S12"""
        # TODO: should the factors of 3 be included in the a-parameters?
        e_s = 3*self.a_s * self.temp(w, Phase.SYMMETRIC)**4
        e_b = 3*self.a_b * self.temp(w, Phase.BROKEN)**4
        return e_b * phase + e_s * (1 - phase)

    def gen_cs2(self):
        return self.cs2

    def p(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        """:notes: eq. 5.14, 7.1, 7.33
        $$ p_s = a_s T^4 $$
        $$ p_b = a_b T^4 $$
        """
        p_s = self.a_s * self.temp(w, Phase.SYMMETRIC)**4 - self.V_s
        p_b = self.a_b * self.temp(w, Phase.BROKEN)**4 - self.V_b
        return p_b * phase + p_s * (1 - phase)

    def temp(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Temperature $T(w,\phi)$
        $$ T(w) = \sqrt[4]{\frac{w}{4a(\phi)}} $$
        :param w: enthalpy $w$
        :param phase: phase $\phi$
        :return: temperature $T(w,\phi)$
        """
        return (w / (4*(self.a_b*phase + self.a_s*(1-phase))))**(1/4)

    @staticmethod
    def v_shock(xi: th.FloatOrArr) -> th.FloatOrArr:
        """:gw_pt_ssm: eq. B.17
        $$ v_\text{sh}(\xi) = \frac{3\xi^22 - 1}{2\xi} $$
        """
        return (3*xi**2 - 1)/(2*xi)

    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Enthalpy $w(T)$
        $$ w(T) = 4a(\phi)T^4 $$
        """
        return 4 * (self.a_b * phase + self.a_s * (1-phase))**temp**4

    @staticmethod
    def w_shock(xi: th.FloatOrArr, w_n: th.FloatOrArr) -> th.FloatOrArr:
        """:gw_pt_ssm: eq. B.18
        w_\text{sh}(\xi) = w_n \frac{9\xi^2 - 1}{3(1-\xi^2)}
        """
        return w_n * (9*xi**2 - 1)/(2*(1-xi**2))
