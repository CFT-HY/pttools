"""Bag model."""

import logging
import typing as tp

import numpy as np

from pttools.bubble.cs2_bag import CS2_BAG_SCALAR_PTR, cs2_bag_multi, cs2_bag_neg, cs2_bag_temp
from pttools.bubble.integrate import DEFAULT_FLUID_INTEGRATE_METHOD, DF_DTAU_PTR_BAG
from pttools.bubble.phase import Phase
from pttools.bubble.solution_type import SolutionType
from pttools.bubble.solution_type_bag import identify_solution_type_bag
from pttools.models.analytic import AnalyticModel
from pttools.speedup.differential import DifferentialPointer
import pttools.type_hints as th
from pttools.type_hints import FloatOrArr
from pttools.utils.docstrings import copy_docstring_dec

logger = logging.getLogger(__name__)


class BagModel(AnalyticModel):
    r"""Bag equation of state.
    This is one of the simplest equations of state for a relativistic plasma.
    Each integration corresponds to a line on the figure below (fig. 9 of :gw_pt_ssm:`\ `).

    .. plot:: fig/xi_v_plane.py

    :notes:`\ `, p. 37
    """

    DEFAULT_LABEL_LATEX = "Bag model"
    DEFAULT_LABEL_UNICODE = DEFAULT_LABEL_LATEX
    DEFAULT_NAME = "bag"
    TEMPERATURE_IS_PHYSICAL = False

    # These can be used in functions designed for ConstCSModel
    mu_s: tp.Final[float] = 4.
    mu_b: tp.Final[float] = 4.

    def __init__(
            self,
            V_s: float = AnalyticModel.DEFAULT_V_S,
            V_b: float = AnalyticModel.DEFAULT_V_B,
            a_s: float | None = None,
            a_b: float | None = None,
            g_s: float | None = None,
            g_b: float | None = None,
            T_min: float | None = None,
            T_max: float | None = None,
            alpha_n_min: float | None = None,
            name: str | None = None,
            label_latex: str | None = None,
            label_unicode: str | None = None,
            allow_invalid: bool = False,
            auto_potential: bool = False,
            log_info: bool = True):
        if log_info:
            logger.debug(
                "Initialising BagModel with V_s=%s, V_b=%s, a_s=%s, a_b=%s, "
                "g_s=%s, g_b=%s, T_min=%s, T_max=%s, alpha_n_min=%s.",
                V_s, V_b, a_s, a_b, g_s, g_b, T_min, T_max, alpha_n_min
            )
        if V_b != 0:
            logger.warning("V_b has been specified for the bag model, even though it's usually omitted.")
        if alpha_n_min is not None:
            a_s, a_b, _, _ = self.get_a_g(a_s, a_b, g_s, g_b)
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
            auto_potential=auto_potential,
            log_info=log_info
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

    @copy_docstring_dec(AnalyticModel.alpha_plus_bag)
    def alpha_n[T: FloatOrArr](
            self,
            wn: T,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> T:
        return self.alpha_n_bag(
            wn=wn,
            error_on_invalid=error_on_invalid,
            nan_on_invalid=nan_on_invalid,
            log_invalid=log_invalid
        )

    @tp.override
    def alpha_n_min_find(self, w_min: float | None = None, w_max: float | None = None) -> tuple[float, float]:
        return self.w_crit, self.alpha_n(self.w_crit)

    @classmethod
    def alpha_n_min_find_params(
            cls,
            alpha_n_min_target: float,
            a_s_default: float | None = None,
            a_b: float = 1,
            V_s_default: float | None = None,
            V_b: float | None = None,
            safety_factor_alpha: float | None = None,
            **kwargs) -> tuple[float, float, float, float]:
        if V_s_default is None:
            V_s_default = cls.DEFAULT_V_S
        if V_b is None:
            V_b = cls.DEFAULT_V_B
        if (a_s_default is not None and a_s_default < 0) or a_b < 0 or V_s_default < 0 or V_b < 0:
            raise ValueError(
                f"Invalid parameters: a_s_default={a_s_default}, a_b={a_b}, V_s_default={V_s_default}, V_b={V_b}")
        if safety_factor_alpha is None:
            safety_factor_alpha = cls.ALPHA_N_MIN_FIND_SAFETY_FACTOR_ALPHA
        a_s = a_b / (1 - 3*alpha_n_min_target * safety_factor_alpha)
        if a_s_default is not None and a_s_default < a_s:
            a_s = a_s_default
        return a_s, a_b, V_s_default, V_b

    @copy_docstring_dec(AnalyticModel.alpha_plus_bag)
    def alpha_plus[T: FloatOrArr](
            self,
            wp: T,
            wm: th.FloatOrArr,
            vp_tilde: float | None = None,
            sol_type: SolutionType | None = None,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> T:
        return self.alpha_plus_bag(
            wp=wp, wm=wm, vp_tilde=vp_tilde,
            sol_type=sol_type,
            error_on_invalid=error_on_invalid,
            nan_on_invalid=nan_on_invalid,
            log_invalid=log_invalid
        )

    def alpha_theta_bar_n[T: FloatOrArr](
            self,
            wn: T,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> T:
        return self.alpha_n(
            wn=wn,
            error_on_invalid=error_on_invalid,
            nan_on_invalid=nan_on_invalid,
            log_invalid=log_invalid
        )

    def alpha_theta_bar_n_max_lte[T: FloatOrArr](
            self,
            wn: T,
            sol_type: SolutionType,
            mu_b: th.FloatOrArr = 4.,
            Psi_n: th.FloatOrArr | None = None) -> T:
        return super().alpha_theta_bar_n_max_lte(wn=wn, sol_type=sol_type, mu_b=mu_b, Psi_n=Psi_n)

    def alpha_theta_bar_n_min_lte[T: FloatOrArr](
            self,
            wn: T,
            sol_type: SolutionType,
            mu_s: th.FloatOrArr = 4.,
            mu_b: th.FloatOrArr = 4.,
            Psi_n: th.FloatOrArr | None = None) -> T:
        return super().alpha_theta_bar_n_min_lte(wn=wn, sol_type=sol_type, mu_s=mu_s, mu_b=mu_b, Psi_n=Psi_n)

    def alpha_theta_bar_plus[T: FloatOrArr](
            self,
            wp: T,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> T:
        return self.alpha_plus(
            wp=wp,
            wm=np.nan,  # Not used
            error_on_invalid=error_on_invalid,
            nan_on_invalid=nan_on_invalid,
            log_invalid=log_invalid
        )

    @tp.override
    def critical_temp(
            self,
            guess: float | None = None,
            guess_backup: float = 2,
            t_max_backup: float = 10000,
            allow_fail: bool = False) -> float:
        r"""Critical temperature for the bag model.

        $$T_{cr} = \sqrt[4]{\frac{V_s - V_b}{a_s - a_b}}$$
        Note that :giese_2020:`\ ` p. 6 is using a different convention.

        The parameters are not used, as the critical temperature is computed analytically.
        They are present only for compatibility with :meth:`Model.critical_temp`.
        """
        return ((self.V_s - self.V_b) / (self.a_s - self.a_b))**0.25

    cs2 = staticmethod(cs2_bag_multi)
    cs2_neg = staticmethod(cs2_bag_neg)
    cs2_temp = staticmethod(cs2_bag_temp)

    @tp.override
    def cs2_max(
            self,
            w_max: float, phase: Phase,
            w_min: float = 0, allow_fail: bool = False, **kwargs) -> tuple[float, float]:
        return 1/3, np.nan

    @tp.override
    def cs2_min(
                self,
                w_max: float, phase: Phase,
                w_min: float = 0, allow_fail: bool = False, **kwargs) -> tuple[float, float]:
        return 1/3, np.nan

    def delta_theta[T: FloatOrArr](
            self,
            wp: T, wm: th.FloatOrArr,
            error_on_invalid: bool = True, nan_on_invalid: bool = True, log_invalid: bool = True) -> T:
        delta_theta = (self.V_s - self.V_b) * np.ones_like(wp) * np.ones_like(wm)
        return self.check_delta_theta(
            delta_theta, xp=wp, xm=wm, x_name="w",
            error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
        )

    def cs2_ptr(self) -> th.CS2FunScalarPtr:
        return CS2_BAG_SCALAR_PTR

    def df_dtau_ptr(self) -> DifferentialPointer:
        return DF_DTAU_PTR_BAG

    def e_temp[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        r"""Energy density as a function of temperature, :giese_2021:`\ ` eq. 15, :borsanyi_2016:`\ `, eq. S12
        The convention for $a_s$ and $a_b$ is that of :notes:`\ `, eq. 7.33.
        """
        self.validate_temp(temp)
        e_s = 3 * self.a_s * temp**4 + self.V_s
        e_b = 3 * self.a_b * temp**4 + self.V_b
        return tp.cast(T, e_b * phase + e_s * (1 - phase))

    # def nu_gdh2024(self, w: th.FloatOrArr, phase: th.FloatOrArr = Phase.BROKEN) -> th.FloatOrArr:
    #     """This is not the case when V != 0"""
    #     return np.zeros_like(w) * np.zeros_like(phase)

    # def omega[T: FloatOrArr](self, w: T, phase: th.FloatOrArr) -> T:
    #     """This is not the case when V != 0"""
    #     return 1/3 * np.ones_like(w) * np.ones_like(phase)

    def p_temp[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        r"""Pressure $p(T,\phi)$, :notes:`\ `, eq. 5.14, 7.1, 7.33, :giese_2021:`\ `, eq. 18
        $$p_s = a_s T^4$$
        $$p_b = a_b T^4$$
        The convention for $a_s$ and $a_b$ is that of :notes:`\ ` eq. 7.33.
        """
        self.validate_temp(temp)
        p_s = self.a_s * temp**4 - self.V_s
        p_b = self.a_b * temp**4 - self.V_b
        return tp.cast(T, p_b * phase + p_s * (1 - phase))

    def params_str(self) -> str:
        return f"a_s={self.a_s}, a_b={self.a_b}, V_s={self.V_s}, V_b={self.V_b}"

    def s_temp[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        r"""Entropy density $s=\frac{dp}{dT}$
        $$s_s = 4 a_s T^3$$
        $$s_b = 4 a_b T^3$$
        Derived from :notes:`\ ` eq. 7.33.
        """
        self.validate_temp(temp)
        s_s = 4 * self.a_s * temp**3
        s_b = 4 * self.a_b * temp**3
        return tp.cast(T, s_b * phase + s_s * (1 - phase))

    @tp.override
    def solution_type(
            self,
            v_wall: float,
            alpha_n: float,
            wn: float | None = None,
            wn_guess: float | None = None,
            wm_guess: float | None = None) -> SolutionType:
        return identify_solution_type_bag(
            v_wall=v_wall, alpha_n=alpha_n,
            df_dtau_ptr=self.df_dtau_ptr(), ode_method=DEFAULT_FLUID_INTEGRATE_METHOD,
            cs2_ptr=CS2_BAG_SCALAR_PTR)

    def temp[T: FloatOrArr](self, w: T, phase: th.FloatOrArr) -> T:
        r"""Temperature $T(w,\phi)$. Inverted from
        $$T(w) = \sqrt[4]{\frac{w}{4a(\phi)}}$$.

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        :return: temperature $T(w,\phi)$
        """
        # return (w / (4*(self.a_b*phase + self.a_s*(1-phase))))**(1/4)
        # Defined in the same way as for ConstCSModel
        temp_s = (w / (4 * self.a_s))**0.25
        temp_b = (w / (4 * self.a_b))**0.25
        return tp.cast(T, temp_b * phase + temp_s * (1 - phase))

    def theta[T: FloatOrArr](self, w: T, phase: th.FloatOrArr) -> T:
        r"""Trace anomaly $\theta$.

        For the bag model the trace anomaly $\theta$ does not depend on the enthalpy.
        """
        return tp.cast(T, (self.V_b * phase + self.V_s * (1 - phase)) * np.ones_like(w))

    @staticmethod
    def v_shock[T: FloatOrArr](xi: T) -> T:
        r"""Velocity at the shock, :gw_pt_ssm:`\ ` eq. B.17
        $$v_\text{sh}(\xi) = \frac{3\xi^22 - 1}{2\xi}$$.
        """
        return tp.cast(T, (3 * xi**2 - 1) / (2 * xi))

    def w[T: FloatOrArr](self, temp: T, phase: th.FloatOrArr) -> T:
        r"""Enthalpy $w(T)$
        $$w(T) = 4a(\phi)T^4$$.

        :param temp: temperature $T$
        :param phase: phase $\phi$
        """
        self.validate_temp(temp)
        return tp.cast(T, 4 * (self.a_b * phase + self.a_s * (1 - phase)) * temp**4)

    def wn[T: FloatOrArr](
            self,
            alpha_n: T,
            wn_guess: float | None = 1,
            analytical: bool = True,
            theta_bar: bool = False,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> T:
        r"""Enthalpy at nucleation temperature
        $$w_n = \frac{4}{3} \frac{V_s - V_b}{\alpha_n}$$
        This can be derived from the equations for $\theta$ and $\alpha_n$.
        """
        if theta_bar:
            return super().wn(
                alpha_n, wn_guess=wn_guess, theta_bar=theta_bar,
                error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
            )
        if analytical:
            return tp.cast(T, self.bag_wn_const / alpha_n)
        return super().wn(
            alpha_n, wn_guess,
            error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
        )

    @staticmethod
    def w_shock(xi: th.FloatOrArr, w_n: th.FloatOrArr) -> th.FloatOrArr:
        r"""Enthalpy at the shock, :gw_pt_ssm:`\ ` eq. B.18
        $$w_\text{sh}(\xi) = w_n \frac{9\xi^2 - 1}{3(1-\xi^2)}$$.
        """
        return w_n * (9 * xi**2 - 1) / (2 * (1 - xi**2))
