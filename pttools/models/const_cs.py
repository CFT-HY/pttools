r"""Constant sound speed model, aka. $\mu, \nu$ model"""

import logging
import typing as tp

import numba
import numpy as np
from scipy.optimize import minimize, minimize_scalar, OptimizeResult

import pttools.type_hints as th
from pttools.bubble.boundary import Phase, SolutionType
from pttools.bubble import transition
from pttools.models.analytic import AnalyticModel
from pttools.models.bag import BagModel

logger = logging.getLogger(__name__)


def cs2_to_mu(cs2: th.FloatOrArr) -> th.FloatOrArr:
    r"""Convert speed of sound squared $c_s^2$ to $\mu$

    $$\mu = 1 + \frac{1}{c_s^2}$$
    """
    return 1 + 1 / cs2


class ConstCSModel(AnalyticModel):
    r"""$\mu, \nu$-model"""
    DEFAULT_LABEL_LATEX = "Constant $c_s$ model"
    DEFAULT_LABEL_UNICODE = "Constant cₛ model"
    DEFAULT_NAME = "const_cs"

    def __init__(
            self,
            css2: float, csb2: float,
            V_s: float = None, V_b: float = 0,
            a_s: float = None, a_b: float = None,
            g_s: float = None, g_b: float = None,
            alpha_n_min: float = None,
            T_min: float = None,
            T_max: float = None,
            T_ref: float = 1,
            T_crit_guess: float = None,
            name: str = None,
            label_latex: str = None,
            label_unicode: str = None,
            allow_invalid: bool = False):
        # Ensure that these descriptions correspond to those in the base class
        r"""
        :param a_s: prefactor of $p$ in the symmetric phase. The convention is as in :notes:`\ ` eq. 7.33.
        :param a_b: prefactor of $p$ in the broken phase. The convention is as in :notes:`\ ` eq. 7.33.
        :param css2: $c_{s,s}^2$, speed of sound squared in the symmetric phase
        :param csb2: $c_{s,b}^2$, speed of sound squared in the broken phase
        :param V_s: $V_s \equiv \epsilon_s$, the potential term of $p$ in the symmetric phase
        :param V_b: $V_b \equiv \epsilon_b$, the potential term of $p$ in the broken phase
        :param T_ref: reference temperature, usually 1 * unit of choice, e,g. 1 GeV
        :param name: custom name for the model

        TODO: Rename mu to mu_s and nu to mu_b
        """
        logger.debug(f"Initialising ConstCSModel with css2={css2}, csb2={csb2}.")
        self.css2 = self.validate_cs2(css2, "css2")
        self.csb2 = self.validate_cs2(csb2, "csb2")

        if np.isnan(css2) or np.isnan(csb2):
            raise ValueError(
                "c_{s,s}^2 and c_{s,b}^2 have to be 0 < c_s <= 1."
                f"Got: c_{{s,s}}^2={css2}, c_{{s,b}}^2={csb2}."
            )
        if css2 > 1/3 or csb2 > 1/3:
            logger.warning(
                "c_{s,s}^2 > 1/3 or c_{s,b}^2 > 1/3. "
                "Please ensure that g_eff is monotonic in your model. "
                f"Got: c_{{s,s}}^2=%s, c_{{s,b}}^2=%s.",
                css2, csb2
            )

        self.css = np.sqrt(css2)
        self.csb = np.sqrt(csb2)
        self.mu = cs2_to_mu(css2)
        self.nu = cs2_to_mu(csb2)
        self.T_ref = T_ref
        # This seems to contain invalid assumptions and approximations.
        # self.alpha_n_min_limit_cs = (self.mu - self.nu) / (3*self.mu)
        self.const_cs_wn_const: float = 4 / 3 * (1 / self.nu - 1 / self.mu)

        if T_crit_guess is None:
            T_crit_guess = T_ref

        if alpha_n_min is not None:
            a_s, a_b, V_s, V_b = self.alpha_n_min_find_params(
                alpha_n_min_target=alpha_n_min, a_s_default=a_s, a_b=a_b, V_s_default=V_s, V_b=V_b)

        label_prec = 3
        label_latex = f"Const. $c_s, c_{{ss}}^2={self.css2:.{label_prec}f}, c_{{sb}}^2={self.csb2:.{label_prec}f}$" \
            if not label_latex else label_latex
        label_unicode = f"Const. cₛ, css2={self.css2:.{label_prec}f}, csb2={self.csb2:.{label_prec}f}" \
            if not label_unicode else label_unicode

        super().__init__(
            V_s=V_s, V_b=V_b,
            a_s=a_s, a_b=a_b,
            g_s=g_s, g_b=g_b,
            T_min=T_min, T_max=T_max, T_crit_guess=T_crit_guess,
            name=name, label_latex=label_latex, label_unicode=label_unicode,
            allow_invalid=allow_invalid
        )
        # This can only be set after self.a_s and self.a_b are set.
        # This seems to contain invalid assumptions and approximations.
        # self.alpha_n_min_limit_a = (self.a_s - self.a_b) / (3 * self.a_s)

    @staticmethod
    def validate_cs2(cs2: float, name: str = "cs2") -> float:
        if cs2 < 0 or cs2 > 1:
            return np.nan
        if cs2 > 1/3 and np.isclose(cs2, 1/3):
            logger.warning(f"{name} is slightly over 1/3. Changing it to 1/3.")
            return 1/3
        return cs2

    # def a_s_min(self, a_b: th.FloatOrArr, alpha_n: th.FloatOrArr) -> th.FloatOrArr:
    #     """Theoretical minimum for $a_s$"""
    #     return self.nu / (self.mu * (1 - 3*alpha_n)) * a_b

    def alpha_n(
            self,
            wn: th.FloatOrArr,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        r"""Transition strength parameter at nucleation temperature, $\alpha_n$, :notes:`\ `, eq. 7.40.
        $$\alpha_n = \frac{4}{3} \left( \frac{1}{\nu} - \frac{1}{\mu} + \frac{1}{w_n} (V_s - V_b) \right)$$

        :param wn: $w_n$, enthalpy of the symmetric phase at the nucleation temperature
        :param error_on_invalid: raise error for invalid values
        :param nan_on_invalid: return nan for invalid values
        :param log_invalid: whether to log invalid values
        """
        self.check_w_for_alpha(
            wn,
            error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid,
            name="wn", alpha_name="alpha_n"
        )
        # self.check_p(wn, allow_fail=allow_no_transition)

        tn = self.temp(wn, Phase.SYMMETRIC)
        ret = (1-4/self.mu)/3 - (1-4/self.nu)/3*self.w(tn, Phase.BROKEN)/wn + self.bag_wn_const/wn
        invalid = ret < 0
        if (error_on_invalid or nan_on_invalid or log_invalid) and np.any(invalid):
            if np.isscalar(ret):
                info = f"Got negative alpha_n={ret} with wn={wn}, mu={self.mu}, nu={self.nu}, t_crit={self.T_crit}."
            else:
                i = np.argmin(wn)
                info = f"Got negative alpha_n. Most problematic values: alpha_n={ret[i]}, wn={wn[i]}, mu={self.mu}, nu={self.nu}"
            if log_invalid:
                logger.error(info)
            if error_on_invalid:
                raise ValueError(info)
            if nan_on_invalid:
                if np.isscalar(ret):
                    return np.nan
                ret[invalid] = np.nan
        return ret

    # def alpha_n_error_msg(self, alpha_n: th.FloatOrArr, name: str = "alpha_n") -> str:
    #     # Additional parameter: a_limit: bool = True
    #
    #     if not np.isscalar(alpha_n):
    #         alpha_n = np.min(alpha_n)
    #     # msg = \
    #         f"Got invalid {name}={alpha_n}, since " \
    #     #     f"alpha_n_min_limit_cs={self.alpha_n_min_limit_cs} for css2={self.css2}, csb2={self.csb2} "\
    #     #     f"({'fail' if alpha_n < self.alpha_n_min_limit_cs else 'OK'})"
    #     # if a_limit:
    #     #     msg += \
    #     msg = \
    #         f"Got invalid {name}={alpha_n}, since " \
    #         f"alpha_n_min_limit_a={self.alpha_n_min_limit_a} for a_s={self.a_s}, a_b={self.a_b} "\
    #         f"({'fail' if alpha_n < self.alpha_n_min_limit_a else 'OK'})"
    #     return msg

    def alpha_n_min_find(self, w_min: float = None, w_max: float = None) -> tp.Tuple[float, float]:
        # xopt, fval = super().alpha_n_min_find(w_min=w_min, w_max=w_max)
        analytical = self.alpha_n(self.w_crit)
        # print(f"const_cs alpha_n_min: analytical={fval}, found={analytical}")
        # return xopt, fval
        return self.w_crit, analytical

    def alpha_n_min_find_params(
            self,
            alpha_n_min_target: float,
            a_s_default: float = None,
            a_b: float = 1,
            V_s_default: float = None,
            V_b: float = 0,
            safety_factor_alpha: float = 0.999,
            safety_factor_a: float = 1.001,
            safety_factor_V: float = 0.001,
            a_max: float = 1e4,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True,
            cancel_on_invalid: bool = True) -> tp.Tuple[float, float, float, float]:
        if safety_factor_a < 1 or safety_factor_V < 0:
            raise ValueError(f"Got invalid safety factors: a={safety_factor_a}, V={safety_factor_V}")

        if np.isclose(self.mu, 4) and np.isclose(self.nu, 4):
            return BagModel.alpha_n_min_find_params(
                alpha_n_min_target=alpha_n_min_target,
                a_s_default=a_s_default,
                a_b=a_b,
                V_s_default=V_s_default,
                V_b=V_b,
                safety_factor_alpha=safety_factor_alpha
            )

        # if alpha_n_min_target < self.alpha_n_min_limit_cs:
        #     msg = self.alpha_n_error_msg(alpha_n=alpha_n_min_target, name="alpha_n_min_target", a_limit=False)
        #     if cancel_on_invalid:
        #         logger.error(
        #             msg + ". Using given defaults a_s=%s, a_b=%s, V_s=%s, V_b=%s",
        #             a_s_default, a_b, V_s_default, V_b, self.css2, self.csb2
        #         )
        #         return a_s_default, a_b, V_s_default, V_b
        #     if log_invalid:
        #         logger.error(msg)
        #     if error_on_invalid:
        #         raise ValueError(msg)
        #     if nan_on_invalid:
        #         return np.nan, a_b, np.nan, V_b

        # if np.isclose(self.nu, 4):
        #     # Assuming Tc = T0 = 1
        #     tn_tc = 2 - safety_factor_a
        #     if tn_tc < 0 or tn_tc > 1:
        #         raise ValueError(f"Got invalid Tn/Tc={tn_tc}")
        #     a_s = a_b / (tn_tc**(-self.mu) - self.mu/4*(alpha_n_min_target - (1 - 4/self.mu)/3))
        #     if a_s < a_b:
        #         msg = f"Got invalid a_s={a_s} for a_b={a_b}, mu_s={self.mu}, tn_tc={tn_tc}, alpha_n_min_target={alpha_n_min_target}"
        #         if cancel_on_invalid:
        #             logger.error(
        #                 msg + ". Using given defaults a_s=%s, a_b=%s, V_s=%s, V_b=%s for css2=%s, csb2=%s.",
        #                 a_s_default, a_b, V_s_default, V_b, self.css2, self.csb2
        #             )
        #             return a_s_default, a_b, V_s_default, V_b
        #         if log_invalid:
        #             logger.error(msg)
        #         if error_on_invalid:
        #             raise ValueError(msg)
        #         if nan_on_invalid:
        #             return np.nan, a_b, np.nan, V_b
        #     V_s = a_s - a_b - V_b
        #     if V_s < V_b:
        #         msg = f"Got invalid V_s={V_s} for a_s={a_s}, a_b={a_b}, V_b={V_b}"
        #         if cancel_on_invalid:
        #             logger.error(
        #                 msg + ". Using given defaults a_s=%s, a_b=%s, V_s=%s, V_b=%s for css2=%s, csb2=%s.",
        #                 a_s_default, a_b, V_s_default, V_b, self.css2, self.csb2
        #             )
        #             return a_s_default, a_b, V_s_default, V_b
        #         if log_invalid:
        #             logger.error(msg)
        #         if error_on_invalid:
        #             raise ValueError(msg)
        #         if nan_on_invalid:
        #             return a_s, a_b, np.nan, V_b
        #     return a_s, a_b, V_s, V_b

        # This is an approximation and does not take into account all the terms
        # theor_min = 4/3*(1/self.nu - 1/self.mu)
        # if alpha_n_min_target < theor_min:
        #     alpha_n_min_target_new = theor_min / safety_factor_alpha
        #     msg = f"alpha_n_min_target = {alpha_n_min_target} is below a theoretical estimate of {theor_min}. "\
        #           f"Setting it to {alpha_n_min_target_new}"
        #     logger.warning(msg)
        #     alpha_n_min_target = alpha_n_min_target_new

        if a_s_default is None:
            a_s_default = 1.1
        if V_s_default is None:
            V_s_default = 1.
        if a_b is None:
            a_b = 1.

        # ---
        # Solve numerically
        # ---
        model = ConstCSModel(css2=self.css2, csb2=self.csb2, a_s=a_s_default, a_b=a_b, V_s=V_s_default)
        # If we are already below the target
        if model.alpha_n_min < alpha_n_min_target:
            return a_s_default, a_b, V_s_default, V_b

        V_s = V_s_default
        sol: OptimizeResult = minimize_scalar(
            self.alpha_n_min_find_params_solvable,
            bracket=((a_s_default + a_b)/2, a_s_default, 2*a_s_default),
            bounds=(a_b * safety_factor_a, a_max),
            args=(V_s, a_b, V_b, self.css2, self.csb2, safety_factor_alpha * alpha_n_min_target)
        )
        a_s = sol.x
        if sol.success:
            model2 = ConstCSModel(css2=self.css2, csb2=self.csb2, a_s=a_s, a_b=a_b, V_s=V_s, V_b=V_b)
            if model2.alpha_n_min <= alpha_n_min_target:
                return a_s, a_b, V_s, V_b

        # If no success, try with another V_s
        V_s = V_s_default / 100
        sol: OptimizeResult = minimize_scalar(
            self.alpha_n_min_find_params_solvable,
            bracket=((a_s_default + a_b)/2, a_s_default, 2*a_s_default),
            bounds=(a_b * safety_factor_a, a_max),
            args=(V_s, a_b, V_b, self.css2, self.csb2, safety_factor_alpha * alpha_n_min_target)
        )
        a_s = sol.x
        if sol.success:
            model2 = ConstCSModel(css2=self.css2, csb2=self.csb2, a_s=a_s, a_b=a_b, V_s=V_s, V_b=V_b)
            if model2.alpha_n_min <= alpha_n_min_target:
                return a_s, a_b, V_s, V_b

        # If no success with minimize_scalar, try minimize
        sol: OptimizeResult = minimize(
            self.alpha_n_min_find_params_solvable2,
            x0=np.array([a_s_default, V_s_default]),
            bounds=((a_b * safety_factor_a, None), (V_b + safety_factor_V, None)),
            args=(a_b, V_b, self.css2, self.csb2, safety_factor_alpha * alpha_n_min_target)
        )
        if sol.success:
            a_s = sol.x[0]
            V_s = sol.x[1]
        # If all solvers failed
        else:
            msg = f"Failed to find alpha_n_min. Reason: {sol.message}"
            if cancel_on_invalid:
                logger.error(
                    msg + ". Using given defaults a_s=%s, a_b=%s, V_s=%s, V_b=%s for css2=%s, csb2=%s.",
                    a_s_default, a_b, V_s_default, V_b, self.css2, self.csb2
                )
                return a_s_default, a_b, V_s_default, V_b
            if log_invalid:
                logger.error(msg)
            if error_on_invalid:
                raise RuntimeError(msg)
            if nan_on_invalid:
                return np.nan, a_b, np.nan, V_b

        # If the solver returns a useless result
        model2 = ConstCSModel(css2=self.css2, csb2=self.csb2, a_s=a_s, a_b=a_b, V_s=V_s, V_b=V_b)
        if model2.alpha_n_min > model.alpha_n_min:
            logger.error(
                "alpha_n_min solver returned greater alpha_n_min than with default values. "
                "Using defaults a_s=%s, a_b=%s, V_s=%s, V_b=%s for css2=%s, csb2=%s.",
                a_s_default, a_b, V_s_default, V_b, self.css2, self.csb2
            )
            return a_s_default, a_b, V_s_default, V_b
        return a_s, a_b, V_s, V_b

    @staticmethod
    def alpha_n_min_find_params_solvable(
            a_s: float, V_s: float,
            a_b: float, V_b: float,
            css2: float, csb2: float,
            alpha_n_target: float):
        try:
            model = ConstCSModel(css2=css2, csb2=csb2, a_s=a_s, a_b=a_b, V_s=V_s, V_b=V_b)
        except (ValueError, RuntimeError):
            return np.nan
        diff = model.alpha_n_min - alpha_n_target
        return diff**2

    @classmethod
    def alpha_n_min_find_params_solvable2(
            cls,
            args: np.ndarray,
            a_b: float, V_b: float,
            css2: float, csb2: float,
            alpha_n_target: float):
        a_s = args[0]
        V_s = args[1]
        return cls.alpha_n_min_find_params_solvable(
            a_s=a_s, V_s=V_s,
            css2=css2, csb2=csb2,
            a_b=a_b, V_b=V_b,
            alpha_n_target=alpha_n_target
        )

    def alpha_plus(
            self,
            wp: th.FloatOrArr,
            wm: th.FloatOrArr,
            vp_tilde: float = None,
            sol_type: SolutionType = None,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        r"""If $\nu=4 \Leftrightarrow c_{sb}=\frac{1}{\sqrt{3}}$, then $w_-$ does not affect the result."""
        self.check_w_for_alpha(
            wp,
            # w_min=self.w_crit,
            error_on_invalid=error_on_invalid,
            nan_on_invalid=nan_on_invalid,
            log_invalid=log_invalid,
            name="wp", alpha_name="alpha_plus"
        )
        self.check_w_for_alpha(
            wm,
            error_on_invalid=error_on_invalid,
            nan_on_invalid=nan_on_invalid,
            log_invalid=log_invalid,
            name="wm", alpha_name="alpha_plus"
        )

        alpha_plus = (1 - 4/self.mu)/3 - (1 - 4/self.nu)*wm/(3*wp) + self.bag_wn_const/wp
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
        return (1 - self.nu/self.mu)/3 + self.nu/4 * self.alpha_n_bag(
            wn=wn,
            error_on_invalid=error_on_invalid,
            nan_on_invalid=nan_on_invalid,
            log_invalid=log_invalid
        )

    def alpha_theta_bar_n_max_lte(self, wn: float, sol_type: SolutionType, Psi_n: float = None) -> float:
        r"""$\alpha_{n,\text{max}}^\text{def}$, :ai_2023:`\ `, eq. 28, 31"""
        if sol_type == SolutionType.DETON or sol_type == SolutionType.HYBRID:
            if Psi_n is None or np.isnan(Psi_n):
                Psi_n = self.Psi_n(wn)
            if np.max(np.abs(Psi_n - 1)) > 1:
                logger.warning(
                    "alpha_theta_bar_n_max_lte approximation is not valid, as |1 - Psi_n| > 1. "
                    "You have to check yourself that alpha_n is valid."
                )
            sqrt_val = (1 - Psi_n)/((self.nu - 1)*(self.nu - 2))
            if sqrt_val < 0:
                return np.nan
            return (1 - Psi_n) / 3 * (1 + self.nu/3 * np.sqrt(sqrt_val))
        return np.inf

    def alpha_theta_bar_n_min_lte(self, wn: th.FloatOrArr, sol_type: SolutionType, Psi_n: float = None) -> float:
        r"""$\alpha_{n,\text{min}}^\text{def}$, :ai_2023:`\ `, eq. 27, 30"""
        if Psi_n is None or np.isnan(Psi_n):
            Psi_n = self.Psi_n(wn)
        if sol_type == SolutionType.DETON:
            if np.abs(self.nu - 4) < 1:
                logger.warning(
                    "alpha_theta_bar_n_min_lte_det approximation is not valid, as |nu - 4| > 1. "
                    "You have to check yourself that alpha_n is valid."
                )
            return (1 - Psi_n) / (12*Psi_n) * (4 - (1 - Psi_n)*(self.nu - 4))
        if sol_type == SolutionType.SUB_DEF:
            return np.maximum((1 - Psi_n)/3, (self.mu - self.nu)/(3*self.mu))
        if sol_type == SolutionType.HYBRID:
            # Not known / no simple formula
            return 0
        raise ValueError(f"Invalid solution type: {sol_type}")

    def alpha_theta_bar_plus(
            self,
            wp: th.FloatOrArr,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        return (1 - self.nu/self.mu)/3 + self.nu/4 * self.alpha_plus_bag(
            wp=wp,
            wm=np.nan,  # Not used
            error_on_invalid=error_on_invalid,
            nan_on_invalid=nan_on_invalid,
            log_invalid=log_invalid
        )

    def critical_temp_opt(self, temp: float) -> float:
        const = (self.V_b - self.V_s) * self.T_ref ** 4
        return self.a_s * (temp / self.T_ref)**self.mu - self.a_b * (temp / self.T_ref)**self.nu + const

    def gen_cs2(self):
        # These become compile-time constants
        css2 = self.css2
        csb2 = self.csb2

        # Using the BagModel cs2 saves us from having to compile additional Numba functions
        if css2 == 1/3 and csb2 == 1/3:
            return BagModel.cs2

        @numba.njit
        def cs2(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            # Mathematical operations should be faster than conditional logic in compiled functions.
            return (phase*csb2 + (1 - phase)*css2) * np.ones_like(w)
        return cs2

    def gen_cs2_neg(self) -> th.CS2Fun:
        css2 = self.css2
        csb2 = self.csb2

        if css2 == 1/3 and csb2 == 1/3:
            return BagModel.cs2_neg

        @numba.njit
        def cs2_neg(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            return -(phase*csb2 + (1 - phase)*css2) * np.ones_like(w)
        return cs2_neg

    def cs2_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        # ConstCSModel.cs2() is independent of T and w
        return self.cs2(temp, phase)

    def delta_theta(
            self,
            wp: th.FloatOrArr, wm: th.FloatOrArr,
            error_on_invalid: bool = True, nan_on_invalid: bool = True, log_invalid: bool = True) -> th.FloatOrArr:
        ret = (1/4 - 1/self.mu)*wp/3 - (1/4 - 1/self.nu)*wm/3 + self.V_s - self.V_b
        return self.check_delta_theta(
            ret, wp=wp, wm=wm,
            error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid
        )

    def e_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Energy density $e(T,\phi)$
        $$e_s = a_s (\mu - 1) T^\mu + V_s$$
        $$e_b = a_b (\nu - 1) T^\nu + V_b$$
        :giese_2021:`\ `, eq. 15.
        In the article there is a typo: the 4 there should be a $\mu$.
        """
        self.validate_temp(temp)
        e_s = (self.mu - 1) * self.a_s * (temp / self.T_ref) ** (self.mu - 4) * temp ** 4 + self.V_s
        e_b = (self.nu - 1) * self.a_b * (temp / self.T_ref) ** (self.nu - 4) * temp ** 4 + self.V_b
        return e_b * phase + e_s * (1 - phase)

    def export(self) -> tp.Dict[str, any]:
        return {
            **super().export(),
            "css2": self.css2,
            "csb2": self.csb2,
            "mu": self.mu,
            "nu": self.nu
        }

    def inverse_enthalpy_ratio(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        return self.a_b * self.nu / (self.a_s * self.mu)

    def params_str(self) -> str:
        return \
            f"css2={self.css2:.3f}, csb2={self.csb2:.3f}, alpha_n_min={self.alpha_n_min:.3f} " \
            f"(a_s={self.a_s:.3f}, a_b={self.a_b:.3f}, V_s={self.V_s:.3f}, V_b={self.V_b:.3f})"

    def p_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pressure $p(T,\phi)$
        $$p_s = a_s T^\mu - V_s$$
        $$p_b = a_b T^\nu - V_b$$
        :giese_2021:`\ `, eq. 15.
        """
        self.validate_temp(temp)
        p_s = self.a_s * (temp / self.T_ref) ** (self.mu - 4) * temp ** 4 - self.V_s
        p_b = self.a_b * (temp / self.T_ref) ** (self.nu - 4) * temp ** 4 - self.V_b
        return p_b * phase + p_s * (1 - phase)

    def s_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Entropy density $s=\frac{dp}{dT}$
        $$s_s = \mu a_s \left( \frac{T}{T_0} \right)^{\mu-1} T_0^3$$
        $$s_b = \nu a-b \left( \frac{T}{T_0} \right)^{\nu-1} T_0^3$$
        Derived from :giese_2021:`\ `, eq. 15.
        """
        self.validate_temp(temp)
        s_s = self.mu * self.a_s * (temp / self.T_ref) ** (self.mu - 4) * temp ** 3
        s_b = self.nu * self.a_b * (temp / self.T_ref) ** (self.nu - 4) * temp ** 3
        return s_b * phase + s_s * (1 - phase)

    def solution_type(
            self,
            v_wall: float,
            alpha_n: float,
            wn: float = None,
            wn_guess: float = None,
            wm_guess: float = None) -> SolutionType:
        if v_wall**2 < self.csb2:
            return SolutionType.SUB_DEF
        # For detonations alpha_theta_bar_plus = alpha_theta_bar_n
        alpha_theta_bar_plus = self.alpha_theta_bar_n_from_alpha_n(alpha_n, wn=wn, wn_guess=wn_guess)
        # Second-order equation for vm
        b = 3 * alpha_theta_bar_plus - 1 - v_wall ** 2 * (1 / self.csb2 + 3 * alpha_theta_bar_plus)
        # If the result were vm < 0
        if b > 0:
            return SolutionType.HYBRID
        a = v_wall / self.csb2
        c = v_wall
        disc = b**2 - 4*a*c
        if disc < 0:
            return SolutionType.HYBRID
        # A detonation solution exists
        return SolutionType.DETON

    def temp(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Temperature $T(w,\phi)$. Inverted from the equation of $w(T,\phi)$.
        $$T_s = T_0 \left( \frac{w}{\mu a_s T_0^4} \right)^\frac{1}{\mu}$$
        $$T_b = T_0 \left( \frac{w}{\nu a_s T_0^4} \right)^\frac{1}{\nu}$$
        """
        # Some solvers may call this function with w < 0 when finding a solution, which causes NumPy to emit warnings.
        invalid = w < 0
        if np.isscalar(w):
            if invalid:
                w = np.nan
        else:
            if np.any(invalid):
                w = w.copy()
                w[invalid] = np.nan
        temp_s = self.T_ref * (w / (self.mu * self.a_s * self.T_ref ** 4)) ** (1 / self.mu)
        temp_b = self.T_ref * (w / (self.nu * self.a_b * self.T_ref ** 4)) ** (1 / self.nu)
        return temp_b * phase + temp_s * (1 - phase)

    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Enthalpy density $w(T,\phi)$
        $$w_s = \mu a_s \left( \frac{T}{T_0} \right)^\mu T_0^4$$
        $$w_s = \nu a_s \left( \frac{T}{T_0} \right)^\nu T_0^4$$
        """
        self.validate_temp(temp)
        w_s = self.mu * self.a_s * (temp / self.T_ref) ** (self.mu - 4) * temp ** 4
        w_b = self.nu * self.a_b * (temp / self.T_ref) ** (self.nu - 4) * temp ** 4
        return w_b * phase + w_s * (1 - phase)

    def w_n(
            self,
            alpha_n: th.FloatOrArr,
            wn_guess: float = 1,
            analytical: bool = True,
            theta_bar: bool = False,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        r"""Enthalpy at nucleation temperature"""
        if theta_bar:
            return super().w_n(
                alpha_n=alpha_n,
                wn_guess=wn_guess,
                theta_bar=theta_bar,
                error_on_invalid=error_on_invalid,
                nan_on_invalid=nan_on_invalid,
                log_invalid=log_invalid
            )
        # invalid_alpha_n = alpha_n < self.alpha_n_min_limit_a or alpha_n < self.alpha_n_min_limit_cs
        # if np.any(invalid_alpha_n):
        #     msg = self.alpha_n_error_msg(alpha_n=alpha_n)
        #     if log_invalid:
        #         logger.error(msg)
        #     if error_on_invalid:
        #         raise ValueError(msg)
        #     if nan_on_invalid:
        #         if np.isscalar(alpha_n):
        #             return np.nan
        #         alpha_n = alpha_n.copy()
        #         alpha_n[invalid_alpha_n] = np.nan

        if analytical and np.isclose(self.nu, 4):
            wn = self.bag_wn_const / (alpha_n + (4/self.mu - 1)/3)
            if np.any(wn < 0):
                msg = self.w_n_error_msg(alpha_n=alpha_n, param=wn, param_name="wn")
                if log_invalid:
                    logger.error(msg)
                if error_on_invalid:
                    raise ValueError(msg)
                if nan_on_invalid:
                    if np.isscalar(alpha_n):
                        return np.nan
                    wn[wn < 0] = np.nan
            return wn

        diff = alpha_n - self.const_cs_wn_const
        if np.any(diff < 0) and log_invalid:
            logger.warning(self.w_n_error_msg(alpha_n=alpha_n, param=diff, param_name="diff"))

        return super().w_n(
            alpha_n, wn_guess=wn_guess,
            error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
        )
