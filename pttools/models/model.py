"""Template for equations of state"""

import typing as tp

import numba
import numpy as np
import scipy.interpolate
import scipy.optimize

import pttools.type_hints as th
from pttools.bubble.boundary import Phase
from pttools.models.thermo import ThermoModel


class Model:
    """Template for equations of state"""
    def __init__(self, thermo: tp.Optional[ThermoModel], V_s: float = 0, V_b: float = 0):
        """
        :param thermo: model of the underlying thermodynamics.
            Some models don't take this, but use their own approximations instead.
        :param V_s: the constant term in the expression of $p$ in the symmetric phase
        :param V_b: the constant term in the expression of $p$ in the broken phase
        """
        self.thermo = thermo
        self.V_s = V_s
        self.V_b = V_b

        # Equal values are allowed so that the default values are accepted.
        if V_b > V_s:
            raise ValueError("The bubble does not expand if V_b >= V_s.")

        self.critical_temp_const = 90 / np.pi ** 2 * (self.V_b - self.V_s)

        if thermo is not None:
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

    def critical_temp(self, guess: float) -> float:
        """Solves for the critical temperature $T_c$, where $p_s(T_c)=p_b(T_c)$"""
        # This returns np.float64
        return scipy.optimize.fsolve(
            self.critical_temp_opt,
            guess
            # args=(const),
            # xtol=
            # factor=0.1
        )[0]

    def cs2(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        """Speed of sound squared. This must be a Numba-compiled function."""
        raise RuntimeError("The cs2(w, phase) function has not yet been loaded")

    def e(self, w: th.FloatOrArr, phase: th.FloatOrArr):
        r"""Computes the energy density $e(w,\phi)$ by first computing $w(T)$ and then using :borsanyi_2016: eq. S12
        $$ e(T) = \frac{\pi^2}{30} g_e(T) T^4 $$
        :param w: enthalpy $w$
        :param phase: phase $\phi$
        :return: $e(w)
        """
        temp = self.temp(w, phase)
        return np.pi**2 / 30 * self.thermo.ge(temp, phase) * temp**4

    def gp(self, w: th.FloatOrArr, phase: th.FloatOrArr):
        r"""Effective degrees of freedom for pressure, $g_{\text{eff},p}(w,\phi)$"""
        temp = self.temp(w, phase)
        return self.gp_temp(temp, phase)

    def gp_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr):
        r"""Effective degrees of freedom for pressure, $g_{\text{eff},p}(T,\phi)"""
        return 4*self.thermo.gs(temp, phase) - 3*self.thermo.ge(temp, phase) + (90*self.V(phase)) / (np.pi**2 * temp**4)

    def p(self, w: th.FloatOrArr, phase: th.FloatOrArr):
        r"""Pressure $p(w,\phi)$"""
        temp = self.temp(w, phase)
        return np.pi**2 / 90 * self.gp(temp, phase) * temp**4 - self.V(phase)

    def temp(self, w: th.FloatOrArr, phase: th.FloatOrArr):
        r"""Temperature $T$"""
        if phase == Phase.SYMMETRIC.value:
            return scipy.interpolate.splev(w, self.temp_spline_s)
        elif phase == Phase.BROKEN.value:
            return scipy.interpolate.splrep(w, self.temp_spline_b)
        return scipy.interpolate.splev(w, self.temp_spline_b) * phase \
            + scipy.interpolate.splev(w, self.temp_spline_s) * (1 - phase)

    def theta(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        """Trace anomaly :notes: eq. 7.24
        $$ \theta = \frac{1}{4}(e - 3p) $$
        """
        return 1/4 * (self.e(w, phase) - 3*self.p(w, phase))

    def V(self, phase: th.FloatOrArr) -> th.FloatOrArr:
        """Potential $V$"""
        return phase*self.V_b + (1 - phase)*self.V_s

    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Enthalpy density $w$

        $$ w = e + p = Ts = T \frac{dp}{dT} = \frac{4\pi^2}{90} g_{eff} T^4 $$
        For the steps please see :notes: page 23 and eq. 7.1. and :borsanyi_2016: eq. S12.

        :param temp: temperature $T$ (MeV)
        :param phase: phase $\phi$ (not used)
        :return: enthalpy density $w$
        """
        return (4*np.pi**2)/90 * self.thermo.gs(temp, phase) * temp**4
