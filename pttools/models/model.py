"""Template for equations of state"""

import numba
import numpy as np
import scipy.interpolate

import pttools.type_hints as th
from pttools.bubble.boundary import Phase
from pttools.models.thermo import ThermoModel


class Model:
    """Template for equations of state"""
    def __init__(self, thermo: ThermoModel, V_s: float = 0, V_b: float = 0):
        self.thermo = thermo
        self.V_s = V_s
        self.V_b = V_b

        # Equal values are allowed so that the default values are accepted.
        if V_b > V_s:
            raise ValueError("The bubble does not expand if V_b >= V_s.")

        self.temp_spline_s = scipy.interpolate.splrep(
            self.w(self.thermo.GEFF_DATA_TEMP, Phase.SYMMETRIC), self.thermo.GEFF_DATA_TEMP
        )
        self.temp_spline_b = scipy.interpolate.splrep(
            self.w(self.thermo.GEFF_DATA_TEMP, Phase.BROKEN), self.thermo.GEFF_DATA_TEMP
        )

        self.cs2 = self.gen_cs2()

    def gen_cs2(self):
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

    def cs2(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        """Speed of sound squared. This must be a Numba-compiled function."""
        raise RuntimeError("The cs2(w, phase) function has not yet been loaded")

    # @abc.abstractmethod
    # def p(self, w: float, phase: float):
    #     pass

    def T(self, w: th.FloatOrArr, phase: th.FloatOrArr = None):
        if phase == Phase.SYMMETRIC.value:
            return scipy.interpolate.splev(w, self.temp_spline_s)
        elif phase == Phase.BROKEN.value:
            return scipy.interpolate.splrep(w, self.temp_spline_b)
        return scipy.interpolate.splev(w, self.temp_spline_b) * phase \
            + scipy.interpolate.splev(w, self.temp_spline_s) * (1 - phase)

    def V(self, phase: th.FloatOrArr) -> th.FloatOrArr:
        """Potential"""
        return phase*self.V_b + (1 - phase)*self.V_s

    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr = None) -> th.FloatOrArr:
        r"""Enthalpy density $w$

        $$ w = e + p = Ts = T \frac{dp}{dT} = \frac{4\pi^2}{90} g_{eff} T^4 $$
        For the steps please see :notes: page 23 and eq. 7.1. and :borsanyi_2016: eq. S12.

        :param temp: temperature $T$ (MeV)
        :param phase: phase $\phi$ (not used)
        :return: enthalpy density $w$
        """
        return (4*np.pi**2)/90 * self.thermo.gs(temp, phase) * temp**4
