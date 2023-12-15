"""Full thermodynamics-based model"""

import logging

import numba
import numpy as np
from scipy.interpolate import splev, splrep

import pttools.type_hints as th
from pttools.bubble.boundary import Phase
from pttools.models.model import Model
# if tp.TYPE_CHECKING:
from pttools.models.thermo import ThermoModel

logger = logging.getLogger(__name__)


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

    def __init__(
            self,
            thermo: ThermoModel,
            V_s: float = 0, V_b: float = 0,
            T_crit_guess: float = None,
            allow_invalid: bool = False,
            name: str = None,
            label_latex: str = None,
            label_unicode: str = None):
        logger.debug("Initialising FullModel.")
        if not label_latex:
            label_latex = f"Full ({thermo.label_latex})"
        if not label_unicode:
            label_unicode = f"Full ({thermo.label_unicode})"
        self.thermo = thermo

        super().__init__(
            V_s=V_s, V_b=V_b,
            T_min=thermo.t_min, T_max=thermo.t_max,
            name=name, label_latex=label_latex, label_unicode=label_unicode,
            gen_critical=False, gen_cs2=False, gen_cs2_neg=False, implicit_V=True)

        self.temp_spline_s = splrep(
            np.log10(self.w(self.thermo.GEFF_DATA_TEMP, Phase.SYMMETRIC)), self.thermo.GEFF_DATA_LOG_TEMP
        )
        self.temp_spline_b = splrep(
            np.log10(self.w(self.thermo.GEFF_DATA_TEMP, Phase.BROKEN)), self.thermo.GEFF_DATA_LOG_TEMP
        )
        self.t_crit, self.w_crit = self.criticals(T_crit_guess, allow_invalid)
        self.w_at_alpha_n_min, self.alpha_n_min = self.alpha_n_min_find()

        self.cs2 = self.gen_cs2()

    def gen_cs2(self):
        """This function generates the Numba-jitted cs2 function to be used by the fluid integrator"""
        cs2_spl_s = splrep(
            np.log10(self.w(self.thermo.GEFF_DATA_TEMP, Phase.SYMMETRIC)),
            self.thermo.cs2_full(self.thermo.GEFF_DATA_TEMP, Phase.SYMMETRIC),
            k=1
        )
        cs2_spl_b = splrep(
            np.log10(self.w(self.thermo.GEFF_DATA_TEMP, Phase.BROKEN)),
            self.thermo.cs2_full(self.thermo.GEFF_DATA_TEMP, Phase.BROKEN),
            k=1
        )

        @numba.njit
        def cs2(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            if np.all(phase == Phase.SYMMETRIC.value):
                return splev(np.log10(w), cs2_spl_s)
            if np.all(phase == Phase.BROKEN.value):
                return splev(np.log10(w), cs2_spl_b)
            return splev(np.log10(w), cs2_spl_b) * phase \
                + splev(np.log10(w), cs2_spl_s) * (1 - phase)
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

    def ge_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return self.thermo.ge(temp, phase)

    def gp_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return self.thermo.gp(temp, phase)

    def gs_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return self.thermo.gs(temp, phase)

    def p_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pressure $p(T,\phi)$
        $$ p(T,\phi) = \frac{\pi^2}{90} g_p(T,\phi) T^4$$
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
            return 10**splev(np.log10(w), self.temp_spline_s)
        if np.all(phase == Phase.BROKEN.value):
            return 10**splev(np.log10(w), self.temp_spline_b)
        return 10**splev(np.log10(w), self.temp_spline_b) * phase \
            + 10**splev(np.log10(w), self.temp_spline_s) * (1 - phase)

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
