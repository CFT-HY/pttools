"""Base class for analytical models"""

import abc
import logging
import typing as tp

import numpy as np

import pttools.type_hints as th
from pttools.bubble.boundary import SolutionType
from pttools.models.model import Model
from pttools.speedup.utils import is_nan_or_none

logger = logging.getLogger(__name__)


class AnalyticModel(Model, abc.ABC):
    r"""A generic analytical model, where the temperature dependence of $g_\text{eff}$ is implemented directly in the
    equation of state.

    You should specify either the relativistic degrees of freedom $g_\text{eff}(\phi=s)$ and $g_\text{eff}(\phi=b)$,
    or the prefactors $a_s$ and $a_b$.
    The convention for the latter is as in :notes:`\ ` eq. 7.33. for the bag model, where
    $$p_s = a_sT^4 - V_s,$$
    $$p_b = a_bT^4 - V_b.$$

    :param V_s: $V_s = \epsilon_s$, the potential term of $p$ in the symmetric phase
    :param V_b: $V_b = \epsilon_b$, the potential term of $p$ in the broken phase
    :param a_s: prefactor of $p$ in the symmetric phase
    :param a_b: prefactor of $p$ in the broken phase
    :param g_s: $g_\text{eff}(\phi=s)$, degrees of freedom for $p$ in the symmetric phase at T=T0
    :param g_b: $g_\text{eff}(\phi=b)$, degrees of freedom for $p$ in the broken phase at T=T0
    :param name: custom name for the model
    :param auto_potential: set V_s and V_b so that T_c = 1 (bag model only)
    """
    DEFAULT_V_S = 1
    DEFAULT_A_G_MULT = 1.1

    def __init__(
            self,
            V_s: float = DEFAULT_V_S, V_b: float = None,
            a_s: float = None, a_b: float = None,
            g_s: float = None, g_b: float = None,
            T_min: float = None, T_max: float = None, T_crit_guess: float = None,
            name: str = None,
            label_latex: str = None,
            label_unicode: str = None,
            gen_critical: bool = True,
            gen_cs2: bool = True,
            gen_cs2_neg: bool = True,
            allow_invalid: bool = False,
            auto_potential: bool = False,
            log_info: bool = True):
        if V_b is None:
            V_b = self.DEFAULT_V_B
        if log_info and V_b != 0:
            logger.warning(
                "Got V_b = %s != 0. This may result in inaccurate results with the GW spectrum computation, "
                "as the GW spectrum equations presume V_b = 0.", V_b)

        self.a_s: float
        self.a_b: float
        self.g_s: float
        self.g_b: float
        self.a_s, self.a_b, self.g_s, self.g_b = self.get_a_g(a_s, a_b, g_s, g_b)

        if auto_potential:
            if not ((is_nan_or_none(V_s) or V_s == 0) and (is_nan_or_none(V_b) or V_b == 0)):
                raise ValueError("Cannot set manual potentials when automatic potential is enabled.")
            V_s = self.a_s - self.a_b
            V_b = 0

        self.bag_wn_const: float = 4 / 3 * (V_s - V_b)

        super().__init__(
            V_s=V_s, V_b=V_b,
            T_min=T_min, T_max=T_max, T_crit_guess=T_crit_guess,
            name=name, label_latex=label_latex, label_unicode=label_unicode,
            gen_critical=gen_critical, gen_cs2=gen_cs2, gen_cs2_neg=gen_cs2_neg,
            allow_invalid=allow_invalid, log_info=log_info
        )
        if log_info and self.a_s <= self.a_b:
            logger.warning(
                f"The model \"{self.name}\" does not satisfy a_s > a_b. "
                "Please check that the critical temperature is non-negative. "
                f"Got: a_s={self.a_s}, a_b={self.a_b}.")

    @staticmethod
    def a_from_g(g: th.FloatOrArr) -> th.FloatOrArr:
        return np.pi**2 / 90 * g

    def alpha_n_bag(
            self,
            wn: th.FloatOrArr,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        r"""Transition strength parameter at nucleation temperature, $\alpha_n$, :notes:`\ `, eq. 7.40.
        $$\alpha_n = \frac{4}{3w_n}(V_s - V_b)$$

        :param wn: $w_n$, enthalpy of the symmetric phase at the nucleation temperature
        :param error_on_invalid: raise error for invalid values
        :param nan_on_invalid: return nan for invalid values
        :param log_invalid: log negative values
        """
        self.check_w_for_alpha(
            wn,
            error_on_invalid=error_on_invalid,
            nan_on_invalid=nan_on_invalid,
            log_invalid=log_invalid,
            name="wn", alpha_name="alpha_n"
        )
        # self.check_p(wn, allow_fail=allow_no_transition)
        return self.bag_wn_const / wn

    def alpha_plus_bag(
            self,
            wp: th.FloatOrArr,
            wm: th.FloatOrArr,
            vp_tilde: float = None,
            sol_type: SolutionType = None,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        r"""Transition strength parameter $\alpha_+$, :notes:`\ `, eq. 7.25.
        $$\alpha_+ = \frac{4}{3w_+}(V_s - V_b)$$

        :param wp: $w_+$, enthalpy ahead of the wall
        :param wm: $w_-$, enthalpy behind the wall (not used)
        :param error_on_invalid: raise error for invalid values
        :param nan_on_invalid: return nan for invalid values
        :param log_invalid: whether to log invalid values
        """
        self.check_w_for_alpha(
            wp,
            # w_min=self.w_crit,
            error_on_invalid=error_on_invalid,
            nan_on_invalid=nan_on_invalid,
            name="wp", alpha_name="alpha_plus"
        )
        alpha_plus = self.bag_wn_const / wp
        return self.check_alpha_plus(
            alpha_plus, vp_tilde=vp_tilde, sol_type=sol_type,
            error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
        )

    def export(self) -> tp.Dict[str, any]:
        return {
            **super().export(),
            "a_s": self.a_s,
            "a_b": self.a_b
        }

    @staticmethod
    def g_from_a(a: th.FloatOrArr) -> th.FloatOrArr:
        return 90 / np.pi**2 * a

    def ge_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return 30/np.pi**2 * self.e_temp(temp, phase) / temp**4

    @classmethod
    def get_a_g(cls, a_s: float, a_b: float, g_s: float, g_b: float, default_mult: float = DEFAULT_A_G_MULT):
        a_s_none = is_nan_or_none(a_s)
        a_b_none = is_nan_or_none(a_b)
        g_s_none = is_nan_or_none(g_s)
        g_b_none = is_nan_or_none(g_b)
        a_none = a_s_none and a_b_none
        g_none = g_s_none and g_b_none
        if not g_none:
            if not a_none:
                raise ValueError("Specify either a or g values, not both.")
            if g_s_none:
                g_s = default_mult * g_b
            elif g_b_none:
                g_b = g_s / default_mult

            a_s = cls.a_from_g(g_s)
            a_b = cls.a_from_g(g_b)
        else:
            if a_b_none:
                a_b = 1
            if a_s_none:
                a_s = default_mult * a_b

            g_s = cls.g_from_a(a_s)
            g_b = cls.g_from_a(a_b)

        return a_s, a_b, g_s, g_b

    @classmethod
    def alpha_n_min_find_params_a_g(
            cls,
            a_s: float, a_b: float, g_s: float, g_b: float,
            alpha_n_min_target: float, V_s_default: float, V_b: float,
            default_mult: float = DEFAULT_A_G_MULT,
            safety_factor_alpha = Model.ALPHA_N_MIN_FIND_SAFETY_FACTOR_ALPHA):
        a_s, a_b, _, _ = cls.get_a_g(a_s, a_b, g_s, g_b, default_mult=default_mult)
        return cls.alpha_n_min_find_params(
            alpha_n_min_target=alpha_n_min_target, a_s_default=a_s, a_b=a_b, V_s_default=V_s, V_b=V_b,
            safety_factor_alpha=safety_factor_alpha
        )

    def gs_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr):
        return 45/(2*np.pi**2) * self.s_temp(temp, phase) / temp**4

    def gp_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr):
        return 90/np.pi**2 * self.p_temp(temp, phase) / temp**4
