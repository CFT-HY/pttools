"""Bag model"""

import logging
import typing as tp

import numba
import numpy as np

import pttools.type_hints as th
from pttools.bubble.boundary import SolutionType
from pttools.models.analytic import AnalyticModel

logger = logging.getLogger(__name__)


@numba.njit
def cs2(w: th.FloatOrArr = None, phase: th.FloatOrArr = None) -> th.FloatOrArr:
    r"""Sound speed squared, $c_s^2=\frac{1}{3}$.
    :notes:`\ `, p. 37,
    :rel_hydro_book:`\ `, eq. 2.207
    """
    return 1/3 * np.ones_like(w) * np.ones_like(phase)


class BagModel(AnalyticModel):
    r"""Bag equation of state.
    This is one of the simplest equations of state for a relativistic plasma.
    Each integration corresponds to a line on the figure below (fig. 9 of :gw_pt_ssm:`\ `).

    .. plot:: fig/xi_v_plane.py

    """
    DEFAULT_LABEL_LATEX = "Bag model"
    DEFAULT_LABEL_UNICODE = DEFAULT_LABEL_LATEX
    DEFAULT_NAME = "bag"

    def __init__(
            self,
            V_s: float, V_b: float = 0,
            a_s: float = None, a_b: float = None,
            g_s: float = None, g_b: float = None,
            T_min: float = None, T_max: float = None,
            alpha_n_min: float = None,
            name: str = None,
            label_latex: str = None,
            label_unicode: str = None,
            allow_invalid: bool = False,
            auto_potential: bool = False):
        logger.debug("Initialising BagModel.")
        if V_b != 0:
            logger.warning("V_b has been specified for the bag model, even though it's usually omitted.")
        if alpha_n_min is not None:
            a_s, a_b, V_s, V_b = self.alpha_n_min_find_params(
                alpha_n_min_target=alpha_n_min, a_s_default=a_s, a_b=a_b, V_s_default=V_s, V_b=V_b)

        super().__init__(
            V_s=V_s, V_b=V_b,
            a_s=a_s, a_b=a_b,
            g_s=g_s, g_b=g_b,
            T_min=T_min, T_max=T_max,
            name=name, label_latex=label_latex, label_unicode=label_unicode,
            gen_cs2=False, gen_cs2_neg=False,
            allow_invalid=allow_invalid,
            auto_potential=auto_potential
        )
        if self.a_s <= self.a_b:
            raise ValueError(
                "The bag model must have a_s > a_b for the critical temperature to be non-negative. "
                f"Got: a_s={self.a_s}, a_b={self.a_b}"
            )
        # The < case already generates an error in the base class, but let's check that as well just to be sure.
        if self.V_s <= self.V_b:
            msg = f"The bubble will not expand in the Bag model, when V_s <= V_b. Got: V_s = V_b = {V_s}"
            logger.error(msg)
            if not allow_invalid:
                raise ValueError(msg)

        # These have to be after super().__init__() for a_s and a_b to be populated.
        label_prec = 3
        if self.label_latex is self.DEFAULT_LABEL_LATEX:
            self.label_latex = \
                f"Bag, $a_s={self.a_s:.{label_prec}f}, a_b={self.a_b:.{label_prec}f}, " \
                f"V_s={self.V_s:.{label_prec}f}, V_b={self.V_b:.{label_prec}f}$"
        if self.label_unicode is self.DEFAULT_LABEL_UNICODE:
            self.label_unicode = \
                f"Bag, a_s={self.a_s:.{label_prec}f}, a_b={self.a_b:.{label_prec}f}, " \
                f"V_s={self.V_s:.{label_prec}f}, V_b={self.V_b:.{label_prec}f}"

    def alpha_n(
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

    def alpha_n_min_find(self, w_min: float = None, w_max: float = None) -> tp.Tuple[float, float]:
        return self.w_crit, self.alpha_n(self.w_crit)

    @staticmethod
    def alpha_n_min_find_params(
            alpha_n_min_target: float,
            a_s_default: float = None,
            a_b: float = 1,
            V_s_default: float = None,
            V_b: float = 0,
            safety_factor_alpha: float = 0.999) -> tp.Tuple[float, float, float, float]:
        if a_s_default < 0 or a_b < 0 or V_s_default < 0 or V_b < 0:
            raise ValueError(
                f"Invalid parameters: a_s_default={a_s_default}, a_b={a_b}, V_s_default={V_s_default}, V_b={V_b}")
        a_s = a_b / (1 - 3*alpha_n_min_target * safety_factor_alpha)
        if a_s_default is not None and a_s_default < a_s:
            a_s = a_s_default
        V_s = 1 if V_s_default is None else V_s_default
        return a_s, a_b, V_s, V_b

    def alpha_plus(
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

    def alpha_theta_bar_n(
            self,
            wn: th.FloatOrArr,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        return self.alpha_n(
            wn=wn,
            error_on_invalid=error_on_invalid,
            nan_on_invalid=nan_on_invalid,
            log_invalid=log_invalid
        )

    def critical_temp(self, **kwargs) -> float:
        r"""Critical temperature for the bag model

        $$T_{cr} = \sqrt[4]{\frac{V_s - V_b}{a_s - a_b}}$$
        Note that :giese_2020:`\ ` p. 6 is using a different convention.
        """
        return ((self.V_s - self.V_b) / (self.a_s - self.a_b))**0.25

    cs2 = staticmethod(cs2)

    @staticmethod
    @numba.njit
    def cs2_neg(w: th.FloatOrArr = None, phase: th.FloatOrArr = None) -> th.FloatOrArr:
        return - 1/3 * np.ones_like(w) * np.ones_like(phase)

    @staticmethod
    @numba.njit
    def cs2_temp(temp, phase):
        return cs2(temp, phase)

    def delta_theta(
            self,
            wp: th.FloatOrArr, wm: th.FloatOrArr,
            error_on_invalid: bool = True, nan_on_invalid: bool = True, log_invalid: bool = True) -> th.FloatOrArr:
        delta_theta = (self.V_s - self.V_b) * np.ones_like(wp) * np.ones_like(wm)
        return self.check_delta_theta(
            delta_theta, wp=wp, wm=wm,
            error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
        )

    def e_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Energy density as a function of temperature, :giese_2021:`\ ` eq. 15, :borsanyi_2016:`\ `, eq. S12
        The convention for $a_s$ and $a_b$ is that of :notes:`\ `, eq. 7.33.
        """
        self.validate_temp(temp)
        e_s = 3*self.a_s * temp**4 + self.V_s
        e_b = 3*self.a_b * temp**4 + self.V_b
        return e_b * phase + e_s * (1 - phase)

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

    def params_str(self) -> str:
        return f"a_s={self.a_s}, a_b={self.a_b}, V_s={self.V_s}, V_b={self.V_b}"

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

    def theta(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        """Trace anomaly $\theta$

        For the bag model the trace anomaly $\theta$ does not depend on the enthalpy.
        """
        return (self.V_b * phase + self.V_s * (1 - phase)) * np.ones_like(w)

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
        return 4 * (self.a_b * phase + self.a_s * (1-phase))*temp**4

    def w_n(
            self,
            alpha_n: th.FloatOrArr,
            wn_guess: float = 1,
            analytical: bool = True,
            theta_bar: bool = False,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        r"""Enthalpy at nucleation temperature
        $$w_n = \frac{4}{3} \frac{V_s - V_b}{\alpha_n}$$
        This can be derived from the equations for $\theta$ and $\alpha_n$.
        """
        if theta_bar:
            super().w_n(
                alpha_n, wn_guess=wn_guess, theta_bar=theta_bar,
                error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
            )
        if analytical:
            return self.bag_wn_const / alpha_n
        return super().w_n(
            alpha_n, wn_guess,
            error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
        )

    @staticmethod
    def w_shock(xi: th.FloatOrArr, w_n: th.FloatOrArr) -> th.FloatOrArr:
        r"""Enthalpy at the shock, :gw_pt_ssm:`\ ` eq. B.18
        $$w_\text{sh}(\xi) = w_n \frac{9\xi^2 - 1}{3(1-\xi^2)}$$
        """
        return w_n * (9*xi**2 - 1)/(2*(1-xi**2))
