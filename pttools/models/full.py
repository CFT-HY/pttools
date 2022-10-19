"""Full thermodynamics-based model"""

import typing as tp

import numba
import numpy as np
import scipy.interpolate
import scipy.optimize

import pttools.type_hints as th
from pttools.bubble.boundary import Phase
from pttools.models.model import Model
# if tp.TYPE_CHECKING:
from pttools.models.thermo import ThermoModel


class FullModel(Model):
    r"""Full thermodynamics-based equation of state

    Temperature limits should be set in the ThermoModel.

    :param thermo: model of the underlying thermodynamics.
               Some models don't take this, but use their own approximations instead.
    :param V_s: the constant term in the expression of $p$ in the symmetric phase
    :param V_b: the constant term in the expression of $p$ in the broken phase
    """
    DEFAULT_LABEL = "Full model"
    DEFAULT_NAME = "full"

    def __init__(self, thermo: ThermoModel, V_s: float, V_b: float = 0, name: str = None, label: str = None):
        if label is None:
            label = f"Full model ({thermo.label})"
        super().__init__(V_s=V_s, V_b=V_b, name=name, label=label, gen_cs2=False)
        self.thermo = thermo
        # Override auto-generated limits with those from the ThermoModel
        self.t_min = thermo.t_min
        self.t_max = thermo.t_max

        self.temp_spline_s = scipy.interpolate.splrep(
            np.log10(self.w(self.thermo.GEFF_DATA_TEMP, Phase.SYMMETRIC)), self.thermo.GEFF_DATA_LOG_TEMP
        )
        self.temp_spline_b = scipy.interpolate.splrep(
            np.log10(self.w(self.thermo.GEFF_DATA_TEMP, Phase.BROKEN)), self.thermo.GEFF_DATA_LOG_TEMP
        )

        self.cs2 = self.gen_cs2()

    def gen_cs2(self):
        """This function generates the Numba-jitted cs2 function to be used by the fluid integrator"""
        cs2_spl_s = scipy.interpolate.splrep(
            np.log10(self.w(self.thermo.GEFF_DATA_TEMP, Phase.SYMMETRIC)),
            self.thermo.cs2_full(self.thermo.GEFF_DATA_TEMP, Phase.SYMMETRIC),
            k=1
        )
        cs2_spl_b = scipy.interpolate.splrep(
            np.log10(self.w(self.thermo.GEFF_DATA_TEMP, Phase.BROKEN)),
            self.thermo.cs2_full(self.thermo.GEFF_DATA_TEMP, Phase.BROKEN),
            k=1
        )

        @numba.njit
        def cs2(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            if np.all(phase == Phase.SYMMETRIC.value):
                return scipy.interpolate.splev(np.log10(w), cs2_spl_s)
            if np.all(phase == Phase.BROKEN.value):
                return scipy.interpolate.splev(np.log10(w), cs2_spl_b)
            return scipy.interpolate.splev(np.log10(w), cs2_spl_b) * phase \
                + scipy.interpolate.splev(np.log10(w), cs2_spl_s) * (1 - phase)
        return cs2

    def critical_temp_opt(self, temp: float) -> float:
        """Optimizer function for critical temperature"""
        return (self.thermo.gp(temp, Phase.SYMMETRIC) - self.thermo.gp(temp, Phase.BROKEN))*temp**4 \
            + self.critical_temp_const

    def e_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Energy density $e(T,\phi)$, using :borsanyi_2016:`\ `, eq. S12
        $$ e(T,\phi) = \frac{\pi^2}{30} g_e(T,\phi) T^4 $$
        :param temp: temperature $T$
        :param phase: phase $\phi$
        :return: $e(T,\phi)$
        """
        self.validate_temp(temp)
        return np.pi**2 / 30 * self.thermo.ge(temp, phase) * temp**4 + self.V(phase)

    def gp(self, w: th.FloatOrArr, phase: th.FloatOrArr):
        r"""Effective degrees of freedom for pressure, $g_{\text{eff},p}(w,\phi)$"""
        temp = self.temp(w, phase)
        return self.thermo.gp(temp, phase)

    def p_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pressure $p(T,\phi)$
        $$ p(T,\phi) = \frac{\pi^2}{90} g_p(T,\phi) T^4 - V(\phi) $$
        """
        self.validate_temp(temp)
        return np.pi**2 / 90 * self.thermo.gp(temp, phase) * temp**4 - self.V(phase)

    def s_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Entropy density $s(T,\phi), using :borsanyi_2016:`\ `, eq. S12$
        $$ s(T,\phi) = \frac{2\pi^2}{45} g_s(T) T^3$$
        :param temp: temperature $T$
        :param phase: phase $\phi$
        :return: $s(T,\phi)$
        """
        self.validate_temp(temp)
        return 2*np.pi**2 / 45 * self.thermo.gs(temp, phase) * temp**3

    def temp(self, w: th.FloatOrArr, phase: th.FloatOrArr):
        r"""Temperature $T$"""
        if np.all(phase == Phase.SYMMETRIC.value):
            return 10**scipy.interpolate.splev(np.log10(w), self.temp_spline_s)
        if np.all(phase == Phase.BROKEN.value):
            return 10**scipy.interpolate.splev(np.log10(w), self.temp_spline_b)
        return 10**scipy.interpolate.splev(np.log10(w), self.temp_spline_b) * phase \
            + 10**scipy.interpolate.splev(np.log10(w), self.temp_spline_s) * (1 - phase)

    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Enthalpy density $w$
        $$ w = e + p = Ts = T \frac{dp}{dT} = \frac{2\pi^2}{45} g_s T^4 $$
        For the steps please see :notes:`\ ` page 23 and eq. 7.1. and :borsanyi_2016: eq. S12.

        :param temp: temperature $T$ (MeV)
        :param phase: phase $\phi$ (not used)
        :return: enthalpy density $w$
        """
        self.validate_temp(temp)
        return temp * self.s_temp(temp, phase)
