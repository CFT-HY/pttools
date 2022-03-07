"""Bag model"""

import pttools.type_hints as th
from pttools.bubble.boundary import Phase
from pttools.models.sm import StandardModel
from pttools.models.model import Model

import numba
import numpy as np
from scipy import interpolate


class BagModel(StandardModel, Model):
    r"""Bag equation of state.
    Each integration corresponds to a line on the figure below (fig. 9 of :gw_pt_ssm:`\ `).

    .. plot:: fig/xi_v_plane.py

    """
    def __init__(self):
        super().__init__()
        w = self.w(self.GEFF_DATA_TEMP)
        self.gs_w_spline = interpolate.splrep(w, self.GEFF_DATA_GS)
        self.grho_w_spline = interpolate.splrep(w, self.GEFF_DATA_GE)
        self.T_spline = interpolate.splrep(w, self.GEFF_DATA_TEMP)

    def a_b(self, w: th.FloatOrArr) -> th.FloatOrArr:
        return 4*np.pi**2 / 90 * self.gs_w(w)

    def a_s(self, w: th.FloatOrArr) -> th.FloatOrArr:
        return self.a_b(w)

    @staticmethod
    @numba.njit
    def cs2(w: th.FloatOrArr = None, phase: float = None):
        return 1/3

    def e(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        """:borsanyi_2016: eq. S12

        :return: $e(T)$
        """
        return np.pi ** 2 / 30 * self.ge(temp) * temp ** 4

    def e_w(self, w: th.FloatOrArr) -> th.FloatOrArr:
        """
        :return: $e(w)$
        """
        return np.pi**2 / 30 * self.grho_w(w) * self.T(w)**4

    def grho_w(self, w: th.FloatOrArr) -> th.FloatOrArr:
        """
        :return: $g_\rho(w)$
        """
        return interpolate.splev(w, self.grho_w_spline)

    def gs_w(self, w: th.FloatOrArr) -> th.FloatOrArr:
        """
        :return: $g_s(w)$
        """
        return interpolate.splev(w, self.gs_w_spline)

    def p(self, w: float, phase: float):
        """:notes: eq. 5.14, 7.1, 7.33
        $$ p_s = a_s T^4 $$
        $$ p_b = a_b T^4 $$
        """
        # TODO: could this be done with a single interpolation?
        # TODO: using a's with gs_w() is probably wrong.
        if phase == Phase.SYMMETRIC:
            return self.a_s(w) * self.T(w)**4 - self.V_s
        elif phase == Phase.BROKEN:
            return self.a_b(w) * self.T(w)**4
        raise ValueError(f"Unknown phase: {phase}")

    def T(self, w: th.FloatOrArr, phase: th.FloatOrArr = None):
        return interpolate.splev(w, self.T_spline)

    def theta(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        return self.e(temp) - 3/4 * self.w(temp)

    def theta_w(self, w: th.FloatOrArr) -> th.FloatOrArr:
        return self.e_w(w) - 3/4 * w

    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr = None) -> th.FloatOrArr:
        r"""Enthalpy density $w$

        $$ w = e + p = Ts = T \frac{dp}{dT} = \frac{4\pi^2}{90} g_{eff} T^4 $$
        For the steps please see :notes: page 23 and eq. 7.1. and :borsanyi_2016: eq. S12.

        :param temp: temperature $T$ (MeV)
        :param phase: phase $\phi$ (not used)
        :return: enthalpy density $w$
        """
        return (4*np.pi**2)/90 * self.gs(temp) * temp**4
