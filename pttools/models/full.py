"""Full thermodynamics-based model"""

import typing as tp

import numba
import numpy as np
import scipy.interpolate
import scipy.optimize

import pttools.type_hints as th
from pttools.bubble.boundary import Phase
from pttools.models.base import BaseModel
from pttools.models.thermo import ThermoModel


class FullModel(BaseModel):
    """Template for equations of state"""
    def __init__(self, thermo: ThermoModel, V_s: float = 0, V_b: float = 0):
        """
        :param thermo: model of the underlying thermodynamics.
                       Some models don't take this, but use their own approximations instead.
        :param V_s: the constant term in the expression of $p$ in the symmetric phase
        :param V_b: the constant term in the expression of $p$ in the broken phase
        """
        super().__init__(V_s=V_s, V_b=V_b)
        self.thermo = thermo

        self.temp_spline_s = scipy.interpolate.splrep(
            self.w(self.thermo.GEFF_DATA_TEMP, Phase.SYMMETRIC), self.thermo.GEFF_DATA_TEMP
        )
        self.temp_spline_b = scipy.interpolate.splrep(
            self.w(self.thermo.GEFF_DATA_TEMP, Phase.BROKEN), self.thermo.GEFF_DATA_TEMP
        )

        self.cs2 = self.gen_cs2()

    def gen_cs2(self):
        """This function generates the Numba-jitted cs2 function to be used by the fluid integrator"""
        w_s = self.w(self.thermo.GEFF_DATA_TEMP, Phase.SYMMETRIC)
        w_b = self.w(self.thermo.GEFF_DATA_TEMP, Phase.BROKEN)

        cs2_spl_s = scipy.interpolate.splrep(
            w_s,
            self.thermo.cs2_full(self.thermo.GEFF_DATA_TEMP, Phase.SYMMETRIC),
            k=1)
        cs2_spl_b = scipy.interpolate.splrep(
            w_b,
            self.thermo.cs2_full(self.thermo.GEFF_DATA_TEMP, Phase.BROKEN),
            k=1)

        @numba.njit
        def cs2(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            if phase == Phase.SYMMETRIC.value:
                return scipy.interpolate.splev(w, cs2_spl_s)
            elif phase == Phase.BROKEN.value:
                return scipy.interpolate.splev(w, cs2_spl_b)
            return scipy.interpolate.splev(w, cs2_spl_b) * phase \
                + scipy.interpolate.splev(w, cs2_spl_s) * (1 - phase)
        return cs2

    def critical_temp_opt(self, temp: float):
        """Optimizer function for critical temperature"""
        return (self.gp_temp(temp, Phase.SYMMETRIC) - self.gp_temp(temp, Phase.BROKEN))*temp**4 \
            + self.critical_temp_const

    def e_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Computes the energy density $e(T,\phi)$ by using :borsanyi_2016: eq. S12
        $$ e(T) = \frac{\pi^2}{30} g_e(T) T^4 $$
        :param temp: temperature $T$
        :param phase: phase $\phi$
        :return: $e(T)
        """
        return np.pi**2 / 30 * self.thermo.ge(temp, phase) * temp**4

    def gp(self, w: th.FloatOrArr, phase: th.FloatOrArr):
        r"""Effective degrees of freedom for pressure, $g_{\text{eff},p}(w,\phi)$"""
        temp = self.temp(w, phase)
        return self.gp_temp(temp, phase)

    def gp_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr):
        r"""Effective degrees of freedom for pressure, $g_{\text{eff},p}(T,\phi)"""
        return 4*self.thermo.gs(temp, phase) - 3*self.thermo.ge(temp, phase) + (90*self.V(phase)) / (np.pi**2 * temp**4)

    def p_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pressure $p(T,\phi)$"""
        return np.pi**2 / 90 * self.gp(temp, phase) * temp**4 - self.V(phase)

    def s_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        # TODO
        pass

    def temp(self, w: th.FloatOrArr, phase: th.FloatOrArr):
        r"""Temperature $T$"""
        if phase == Phase.SYMMETRIC.value:
            return scipy.interpolate.splev(w, self.temp_spline_s)
        elif phase == Phase.BROKEN.value:
            return scipy.interpolate.splev(w, self.temp_spline_b)
        return scipy.interpolate.splev(w, self.temp_spline_b) * phase \
            + scipy.interpolate.splev(w, self.temp_spline_s) * (1 - phase)

    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Enthalpy density $w$

        $$ w = e + p = Ts = T \frac{dp}{dT} = \frac{4\pi^2}{90} g_{eff} T^4 $$
        For the steps please see :notes: page 23 and eq. 7.1. and :borsanyi_2016: eq. S12.

        :param temp: temperature $T$ (MeV)
        :param phase: phase $\phi$ (not used)
        :return: enthalpy density $w$
        """
        return (4*np.pi**2)/90 * self.thermo.gs(temp, phase) * temp**4
