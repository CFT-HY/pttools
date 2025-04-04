"""Template for equations of state"""

import abc
import logging
import os
import time
import typing as tp

import numpy as np
from scipy.optimize import fminbound, fsolve, root_scalar

from pttools.bubble.boundary import Phase, SolutionType
from pttools.bubble.chapman_jouguet import v_chapman_jouguet
from pttools.bubble.check import find_most_negative_vals
from pttools.bubble.integrate import add_df_dtau, differentials
from pttools.bubble import transition
from pttools.models.base import BaseModel
from pttools.speedup.differential import DifferentialPointer
from pttools.speedup.options import FORKING
import pttools.type_hints as th

logger = logging.getLogger(__name__)


class Model(BaseModel, abc.ABC):
    r"""Template for equations of state

    :param T_ref: reference temperature.
        Be careful when using a thermodynamics-based model that there are no conflicts in the choices of units.
    :param T_min: minimum temperature at which the model is valid
    :param V_s: the constant term in the expression of $p$ in the symmetric phase
    :param V_b: the constant term in the expression of $p$ in the broken phase
    :param name: custom name for the model
    :param gen_cs2: used internally for postponing the generation of the cs2 function
    """
    ALPHA_N_MIN_FIND_SAFETY_FACTOR_ALPHA: float = 0.999
    DEFAULT_V_B = 0

    def __init__(
            self,
            V_s: float, V_b: float = DEFAULT_V_B,
            T_ref: float = 1, T_min: float = None, T_max: float = None,
            T_crit_guess: float = None,
            name: str = None,
            label_latex: str = None,
            label_unicode: str = None,
            gen_critical: bool = True,
            gen_cs2: bool = True,
            gen_cs2_neg: bool = True,
            implicit_V: bool = False,
            temperature_is_physical: bool = None,
            silence_temp: bool = False,
            allow_invalid: bool = False,
            log_info: bool = True):

        if log_info and implicit_V:
            if V_s != 0 or V_b != 0:
                logger.warning(
                    "Potentials have been specified for the implicit model: %s. "
                    "This is for debugging purposes only. Be careful that the definitions of g and V are consistent.",
                    self.DEFAULT_NAME if name is None else name
                )
        else:
            if V_s < V_b:
                msg = f"The bubble will not expand, when V_s < V_b. Got: V_s={V_s}, V_b={V_b}."
                logger.error(msg)
                if not allow_invalid:
                    raise ValueError(msg)
            # This should not be a problem as long as a critical temperature exists.
            # if V_s == V_b:
            #     logger.warning("The bubble will not expand, when V_s <= V_b. Got: V_b = V_s = %s.", V_s)

        self.temperature_is_physical = self.TEMPERATURE_IS_PHYSICAL if temperature_is_physical is None else temperature_is_physical
        if self.temperature_is_physical is None:
            raise ValueError(
                "It has not been specified whether the temperature scale for the model is physical. "
                "Please specify it in the model definition."
            )

        self.T_ref: float = T_ref
        self.V_s: float = V_s
        self.V_b: float = V_b
        self.__df_dtau_ptr = None
        self.__df_dtau_pid = None

        #: $$\frac{90}{\pi^2} (V_b - V_s)$$
        self.critical_temp_const: float = 90 / np.pi ** 2 * (self.V_b - self.V_s)

        super().__init__(
            T_min=T_min, T_max=T_max,
            name=name, label_latex=label_latex, label_unicode=label_unicode,
            gen_cs2=gen_cs2, gen_cs2_neg=gen_cs2_neg,
            silence_temp=silence_temp
        )
        self.w_min_s = self.w(self.T_min, Phase.SYMMETRIC)
        self.w_min_b = self.w(self.T_min, Phase.BROKEN)
        self.w_max_s = self.w(self.T_max, Phase.SYMMETRIC)
        self.w_max_b = self.w(self.T_max, Phase.BROKEN)
        self.w_min = max(self.w_min_s, self.w_min_b)
        self.w_max = min(self.w_max_s, self.w_max_b)

        # A model could have t_ref = 1 GeV and be valid only for e.g. > 10 GeV
        # if t_ref < self.t_min:
        #     raise logger.warning(f"T_ref should be higher than T_min. Got: T_ref={t_ref}, T_min={self.t_min}")
        if T_ref >= self.T_max:
            raise ValueError(f"T_ref should be lower than T_max. Got: T_ref={T_ref}, T_max={self.T_max}")

        if gen_critical:
            # w_crit = wn_max
            self.T_crit, self.w_crit = self.criticals(T_crit_guess, allow_fail=allow_invalid, log_info=log_info)
            self.w_at_alpha_n_min, self.alpha_n_min = self.alpha_n_min_find()

    # Concrete methods

    @staticmethod
    def _cs2_limit(
            w_max: float, phase: Phase,
            is_max: bool, cs2_fun: th.CS2Fun, w_min: float = 0,
            allow_fail: bool = False, **kwargs) -> tp.Tuple[float, float]:
        r"""Find the minimum or maximum of $c_s^2(w)$ for $w \in [w_\text{min}, w_\text{max}]$"""
        name = "max" if is_max else "min"
        sol = fminbound(cs2_fun, x1=w_min, x2=w_max, args=(phase,), full_output=True, **kwargs)
        w: float = sol[0]
        cs2: float = -sol[1] if is_max else sol[1]
        if sol[2]:
            msg = f"Could not find cs2_{name}. Using cs2_{name}={cs2} at w={w}. Iterations: {sol[3]}"
            logger.error(msg)
            if not allow_fail:
                raise RuntimeError(msg)
        invalid_w = w < 0
        invalid_cs2 = (cs2 < 0 or cs2 > 1)
        if invalid_w or invalid_cs2:
            msg = f"Got {'invalid' if invalid_cs2 else ''} cs2_{name}={cs2} at {'invalid' if invalid_w else ''} w={w}"
            logger.error(msg)
            if not allow_fail:
                raise RuntimeError(msg)
        if cs2 > 1/3:
            logger.warning(f"Got physically impossible cs2={cs2} > 1/3 at w={w}. Check that the model is valid.")
        return cs2, w

    def check_w_for_alpha(
            self,
            w: th.FloatOrArr,
            w_min: float = None,
            w_max: float = None,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True,
            name: str = "w",
            alpha_name: str = "alpha") -> tp.Union[float, np.ndarray]:
        r"""Check that $w \in ({w}_\text{min}, {w}_\text{max})$ for the given $w$."""
        too_small = False
        too_large = False
        if w_min is None:
            w_min = self.w_min
        if w_max is None:
            w_max = self.w_max

        if w is None or np.any(np.isnan(w)):
            if log_invalid:
                logger.error(f"Got w=nan for {name}.")
            # Scalar None cannot be tested for negativity.
            if w is None:
                return np.nan
        elif np.any(w < w_min):
            if np.isscalar(w):
                info = f"Got {name}={w} < w_min={w_min} for {alpha_name}."
            else:
                info = f"Got {name} < w_min={w_min} for {alpha_name}. Most problematic value: w={np.min(w)}"
            if log_invalid:
                logger.error(info)
            if error_on_invalid:
                raise ValueError(info)
            too_small = True
        elif np.any(w > w_max):
            if np.isscalar(w):
                info = f"Got {name}={w} > w_max={w_max} for {alpha_name}."
            else:
                info = f"Got {name} > w_max={w_max} for {alpha_name}. Most problematic value: w={np.max(w)}"
            if log_invalid:
                logger.error(info)
            if error_on_invalid:
                raise ValueError(info)
            too_large = True

        if nan_on_invalid:
            if too_small or too_large:
                if np.isscalar(w):
                    return np.nan
                w = w.copy()
            if too_small:
                w[w < w_min] = np.nan
            if too_large:
                w[w > w_max] = np.nan
        return w

    def alpha_n(
            self,
            wn: th.FloatOrArr,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        r"""Transition strength parameter at nucleation temperature, $\alpha_n$, :notes:`\ `, eq. 7.40.
        $$\alpha_n = \frac{4(\theta(w_n,\phi_s) - \theta(w_n,\phi_b)}{3w_n}$$

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
        # if np.isscalar(wn):
        #     if not self.w_min < wn < self.w_max:
        #         return np.nan
        # else:
        #     wn_invalid = np.logical_or(wn < self.w_min, wn > self.w_max)
        #     if np.any(wn_invalid):
        #         wn = wn.copy()
        #         wn[wn_invalid] = np.nan

        # :param allow_no_transition: allow $w_n$ for which there is no phase transition
        # self.check_p(wn, allow_fail=allow_no_transition)
        diff = self.delta_theta(wp=wn, wm=wn, error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid)

        return 4 * diff / (3 * wn)

    def alpha_n_from_alpha_theta_bar_n(
            self,
            alpha_theta_bar_n: th.FloatOrArr,
            wn: float = None,
            wn_guess: float = None,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        r"""Conversion from $\alpha_{\bar{\theta}_n}$ of :giese_2021:`\ `, eq. 13 to $\alpha_n$"""
        if wn is None or np.isnan(wn):
            wn = self.wn(
                alpha_theta_bar_n, wn_guess=wn_guess, theta_bar=True,
                error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid)
        tn = self.temp(wn, Phase.SYMMETRIC)
        diff = (1 - 1 / (3 * self.cs2(wn, Phase.BROKEN))) * \
            (self.p_temp(tn, Phase.SYMMETRIC) - self.p_temp(tn, Phase.BROKEN)) / wn
        return alpha_theta_bar_n - diff

    def alpha_n_min_find(self, w_min: float = None, w_max: float = None) -> tp.Tuple[float, float]:
        r"""Find $\text{min} \alpha_n(w)$ for $w \in ({w}_\text{min}, {w}_\text{max})$"""
        if w_min is None:
            w_min = self.w_min
        if w_max is None:
            w_max = self.w_crit
        xopt, fval, ierr, numfunc = fminbound(self.alpha_n, x1=w_min, x2=w_max, args=(False, ), full_output=True)
        if ierr:
            raise RuntimeError(f"Finding alpha_n_min failed: xopt={xopt}, fval={fval}, ierr={ierr}, numfunc={numfunc}.")
        alpha_n_w_max = self.alpha_n(w_max)
        if alpha_n_w_max < fval:
            xopt = w_max
            fval = alpha_n_w_max
        logger.debug("alpha_n_min=%s found at w=%s in range w_min=%s, w_max=%s", fval, xopt, w_min, w_max)
        return xopt, fval

    def alpha_plus(
            self,
            wp: th.FloatOrArr,
            wm: th.FloatOrArr,
            vp_tilde: float = None,
            sol_type: SolutionType = None,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        # Todo: This docstring causes the Sphinx error "ERROR: Unknown target name: "w"."
        r"""Transition strength parameter $\alpha_+$
        $$\alpha_+ = \frac{4\Delta \theta}{3{w}_+} = \frac{4(\theta({w}_+,\phi_s) - \theta({w}_-,\phi_b)}{3{w}_+}$$

        :param wp: $w_+$
        :param wm: $w_-$
        :param error_on_invalid: raise error for invalid values
        :param nan_on_invalid: return nan for invalid values
        :param log_invalid: whether to log invalid values
        """
        self.check_w_for_alpha(
            wp,
            # wp can be lower than w_crit when wn < w_crit
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
        # if np.isscalar(wp):
        #     if not self.w_crit < wp < self.w_max:
        #         wp = np.nan
        # else:
        #     wp_invalid = np.logical_or(wp < self.w_crit, wp > self.w_max)
        #     if np.any(wp_invalid):
        #         wp = wp.copy()
        #         wp[wp_invalid] = np.nan
        #
        # if np.isscalar(wm):
        #     if not self.w_min < wm < self.w_max:
        #         wm = np.nan
        # else:
        #     wm_invalid = np.logical_or(wm < self.w_min, wm > self.w_max)
        #     if np.any(wm_invalid):
        #         wm = wm.copy()
        #         wm[wm_invalid] = np.nan

        alpha_plus = 4 * self.delta_theta(
            wp, wm, error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
        ) / (3 * wp)
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
        r"""Transition strength parameter, :giese_2021:`\ `, eq. 13

        $$\alpha_{\bar{\theta}+} = \frac{D \bar{\theta}(T_n)}{3 w_n}$$
        """
        self.check_w_for_alpha(
            wn,
            error_on_invalid=error_on_invalid,
            nan_on_invalid=nan_on_invalid,
            log_invalid=log_invalid,
            name="wn", alpha_name="alpha_theta_bar_n"
        )
        return self.delta_theta_bar(wn, Phase.SYMMETRIC) / (3 * wn)

    def alpha_theta_bar_n_from_alpha_n(self, alpha_n: float, wn: float = None, wn_guess: float = None) -> float:
        r"""Conversion from $\alpha_n$ to $\alpha_{\bar{\theta}_n}$ of :giese_2021:`\ `, eq. 13"""
        if wn is None or np.isnan(wn):
            wn = self.wn(alpha_n, wn_guess=wn_guess)
        tn = self.temp(wn, Phase.SYMMETRIC)
        diff = (1 - 1 / (3 * self.cs2(wn, Phase.BROKEN))) * \
            (self.p_temp(tn, Phase.SYMMETRIC) - self.p_temp(tn, Phase.BROKEN)) / wn
        return alpha_n + diff

    def alpha_theta_bar_plus(
            self,
            wp: th.FloatOrArr,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        r"""Transition strength parameter, :giese_2021:`\ `, eq. 9

        $$\alpha_{\bar{\theta}+} = \frac{D \bar{\theta}(T_+)}{3 w_+}$$
        """
        return self.delta_theta_bar(wp, Phase.SYMMETRIC) / (3 * wp)

    @staticmethod
    def check_alpha_plus(
            alpha_plus: th.FloatOrArr,
            vp_tilde: th.FloatOrArr = None,
            sol_type: SolutionType = None,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        r"""Check that the given $\alpha_+$ values are in the valid range $0 <= \alpha_+ < 1/3

        Modifies the given array.
        """
        if error_on_invalid or nan_on_invalid or log_invalid:
            if sol_type == SolutionType.SUB_DEF or sol_type == SolutionType.HYBRID:
                invalid = np.logical_or(alpha_plus < 0, alpha_plus >= 1/3)
            elif vp_tilde is not None:
                # The square root in the vm_tilde equation must be positive
                sqrt_invalid = ((1 + alpha_plus) * vp_tilde + (1 - 3 * alpha_plus) / (3 * vp_tilde)) ** 2 - 4 / 3 < 0
                invalid = np.logical_or(alpha_plus < 0, sqrt_invalid)
            else:
                invalid = alpha_plus < 0
            if np.any(invalid):
                if np.isscalar(alpha_plus):
                    info = f"Got invalid alpha_plus = {alpha_plus} for sol_type={sol_type}."
                else:
                    info = \
                        f"Got invalid alpha_plus in range: {np.min(alpha_plus)} - {np.max(alpha_plus)}, " \
                        f"for sol_type={sol_type}."
                if log_invalid:
                    logger.error(info)
                if error_on_invalid:
                    raise ValueError(info)
                if nan_on_invalid:
                    if np.isscalar(alpha_plus):
                        return np.nan
                    else:
                        alpha_plus[invalid] = np.nan
        return alpha_plus

    def check_p(self, wn: th.FloatOrArr, allow_fail: bool = False):
        temp = self.temp(wn, Phase.SYMMETRIC)
        self.check_p_temp(temp, allow_fail=allow_fail)

    def check_p_temp(self, temp_n: th.FloatOrArr, allow_fail: bool = False):
        """For the phase transition to happen $p_s(T_n) < p_b(T_n)$"""
        p_s = self.p_temp(temp_n, Phase.SYMMETRIC)
        p_b = self.p_temp(temp_n, Phase.BROKEN)
        diff = p_b - p_s
        prob_diff, prob_temp, prob_p_s, prob_p_b = find_most_negative_vals(diff, temp_n, p_s, p_b)
        if prob_diff is not None:
            text = "Got" if np.isscalar(diff) else "Most problematic values"
            msg = \
                f"Failed the check p_s(T_n) < p_b(T_n). {text}: " \
                f"T_n={prob_temp}, p_s(T_n)={prob_p_s}, p_b(T_n)={prob_p_b}, diff={prob_diff}"
            if not allow_fail:
                raise ValueError(msg)

    @staticmethod
    def check_delta_theta(
            delta_theta: th.FloatOrArr,
            wp: th.FloatOrArr,
            wm: th.FloatOrArr,
            theta_s: th.FloatOrArr = None, theta_b: th.FloatOrArr = None,
            error_on_invalid: bool = True, nan_on_invalid: bool = True, log_invalid: bool = True) -> th.FloatOrArr:
        theta_given = theta_s is not None and theta_b is not None
        if theta_given:
            prob_diff, prob_wp, prob_wm, prob_theta_s, prob_theta_b = \
                find_most_negative_vals(delta_theta, wp, wm, theta_s, theta_b)
        else:
            prob_diff, prob_wp, prob_wm = find_most_negative_vals(delta_theta, wp, wm)
            prob_theta_s = prob_theta_b = None

        if prob_diff is not None:
            text = "Got" if np.isscalar(delta_theta) else "Most problematic values"
            msg = (
                "For a physical equation of state theta_+ > theta_-. "
                f"{text}: wp={prob_wp}, wm={prob_wm}, "
                f"theta_s={prob_theta_s}, theta_b={prob_theta_b}" if theta_given else ""
                f"theta_diff={prob_diff}. "
                "See p. 33 of Hindmarsh and Hijazi, 2019."
            )
            if log_invalid:
                logger.error(msg)
            if error_on_invalid:
                raise ValueError(msg)
            if nan_on_invalid:
                if np.isscalar(delta_theta):
                    return np.nan
                delta_theta[delta_theta < 0] = np.nan

        return delta_theta

    def criticals(self, t_crit_guess: float, allow_fail: bool = False, log_info: bool = True):
        t_crit = self.critical_temp(guess=t_crit_guess, allow_fail=allow_fail)
        # self.t_crit has to be set already here for alpha_n error messages to work.
        self.T_crit = t_crit
        wn_min = self.w(t_crit, Phase.SYMMETRIC)
        alpha_n_at_wn_min = self.alpha_n(wn_min)

        if log_info:
            logger.info(
                f"Initialised model with name={self.name}, T_crit={t_crit}, alpha_n_at_wn_min={alpha_n_at_wn_min}. "
                f"At T_crit: w_s={wn_min}, w_b={self.w(t_crit, Phase.BROKEN)}, "
                f"e_s={self.e_temp(t_crit, Phase.SYMMETRIC)}, e_b={self.e_temp(t_crit, Phase.BROKEN)}, "
                f"p_s={self.p_temp(t_crit, Phase.SYMMETRIC)}, p_b={self.p_temp(t_crit, Phase.BROKEN)}"
            )
        return t_crit, wn_min

    def critical_temp(
            self,
            guess: float = None,
            guess_backup: float = 2,
            t_max_backup: float = 10000,
            allow_fail: bool = False) -> float:
        r"""Solves for the critical temperature $T_c$, where $p_s(T_c)=p_b(T_c)$

        :param guess: starting guess for $T_\text{crit}$
        :param guess_backup: alternative guess that is used if guess is None
        :param t_max_backup: alternative $T_\text{max}$ that is used if $T_\text{max}$ is None
        :param allow_fail: do not raise exceptions on errors
        """
        if guess is None:
            if np.isfinite(self.T_max):
                guess = np.exp((np.log(self.T_min) + np.log(self.T_max)) / 2)
            else:
                guess = guess_backup

        p_s_min = self.p_temp(self.T_min, Phase.SYMMETRIC)
        p_b_min = self.p_temp(self.T_min, Phase.BROKEN)
        if p_s_min >= p_b_min:
            msg = \
                "All models should have p_s(T=T_min) < p_b(T=T_min) for T_crit to exist. " \
                f"Got: T_min={self.T_min}, p_s={p_s_min}, p_b={p_b_min}."
            logger.error(msg)
            if not allow_fail:
                raise ValueError(msg)

        t_max = self.T_max if np.isfinite(self.T_max) else t_max_backup
        t_arr = np.logspace(np.log10(self.T_min), np.log10(t_max), 10)
        p_s_arr = self.p_temp(t_arr, Phase.SYMMETRIC)
        p_b_arr = self.p_temp(t_arr, Phase.BROKEN)
        if np.all(p_s_arr <= p_b_arr):
            msg = \
                "All models should have p_s(T>T_crit) > p_b(T>T_crit) for T_crit to exist. " \
                f"Got: T_max={t_max}, p_s={p_s_arr[-1]}, p_b={p_b_arr[-1]}."
            logger.error(msg)
            if not allow_fail:
                raise ValueError(msg)

        sol = fsolve(
            self.critical_temp_opt,
            x0=np.array([guess]),
            full_output=True
        )
        t_crit = sol[0][0]
        if sol[2] != 1:
            msg = \
                f"Could not find Tc with guess={guess}. " \
                f"Using Tc={t_crit}. Reason: {sol[3]}"
            logger.error(msg)
            if not allow_fail:
                raise RuntimeError(msg)

        # Validate temperature
        if t_crit <= self.T_min:
            msg = f"T_crit should be higher than T_min. Got: T_crit={t_crit}, T_min={self.T_min}"
            logger.error(msg)
            if not allow_fail:
                raise ValueError(msg)
        if t_crit >= self.T_max:
            msg = f"T_max should be lower than T_max. Got: T_crit={t_crit}, T_max={self.T_max}"
            logger.error(msg)
            if not allow_fail:
                raise ValueError(msg)

        # Validate pressure
        p_crit_s = self.p_temp(t_crit, Phase.SYMMETRIC)
        p_crit_b = self.p_temp(t_crit, Phase.BROKEN)
        if not np.isclose(p_crit_s, p_crit_b):
            msg = f"Pressures do not match at T_crit. Got: p_s={p_crit_s}, p_b={p_crit_b}"
            logger.error(msg)
            if not allow_fail:
                raise ValueError(msg)
        if p_crit_s < 0 or p_crit_b < 0:
            msg = f"Pressure cannot be negative at T_crit. Got: p_s={p_crit_s}, p_b={p_crit_b}"
            logger.error(msg)
            if not allow_fail:
                raise ValueError(msg)

        return t_crit

    def critical_temp_opt(self, temp: float) -> float:
        """This function should be zero at the critical temperature $T_c$, where $p_s(T_c)=p_b(T_c)."""
        return self.p_temp(temp, Phase.SYMMETRIC) - self.p_temp(temp, Phase.BROKEN)

    def cs2(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Speed of sound squared $c_s^2(w,\phi)$. This must be a Numba-compiled function.

        $$c_s^2 \equiv \left( \frac{\partial p}{\partial e} \right)_s = \frac{dp/dT}{de/dT}$$
        :rel_hydro_book:`\ `, eq. 2.168
        :giese_2021:`\ `, eq. 3

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        """
        raise RuntimeError("The cs2(w, phase) function has not yet been loaded.")

    def cs2_max(
            self,
            w_max: float, phase: Phase,
            w_min: float = 0, allow_fail: bool = False, **kwargs) -> tp.Tuple[float, float]:
        return self._cs2_limit(w_max, phase, True, self.cs2_neg, w_min, allow_fail, **kwargs)

    def cs2_min(
            self,
            w_max: float, phase: Phase,
            w_min: float = 0, allow_fail: bool = False, **kwargs) -> tp.Tuple[float, float]:
        return self._cs2_limit(w_max, phase, False, self.cs2, w_min, allow_fail, **kwargs)

    def cs2_neg(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return -self.cs2(w, phase)

    def cs2_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return self.cs2(self.w(temp, phase), phase)

    def delta_theta(
            self,
            wp: th.FloatOrArr, wm: th.FloatOrArr,
            error_on_invalid: bool = True, nan_on_invalid: bool = True, log_invalid: bool = True) -> th.FloatOrArr:
        theta_s = self.theta(wp, Phase.SYMMETRIC)
        theta_b = self.theta(wm, Phase.BROKEN)
        diff = theta_s - theta_b
        return self.check_delta_theta(
            diff, wp=wp, wm=wm,
            theta_s=theta_s, theta_b=theta_b,
            error_on_invalid=error_on_invalid, nan_on_invalid=nan_on_invalid, log_invalid=log_invalid
        )

    def delta_theta_bar(self, w: th.FloatOrArr, phase_of_w: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pseudotrace difference $D\bar{\theta}(w)$, :giese_2021:`\ `, eq. 10

        $$D\bar{\theta}(w) = \bar{\theta}(T) - \bar{\theta}(T)$$
        """
        return self.delta_theta_bar_temp(self.temp(w, phase_of_w))

    def delta_theta_bar_temp(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pseudotrace difference $D\bar{\theta}(T)$, :giese_2021:`\ `, eq. 10

        $$D\bar{\theta}(T) = \bar{\theta}(T) - \bar{\theta}(T)$$
        """
        return self.theta_bar_temp(temp, Phase.SYMMETRIC) - self.theta_bar_temp(temp, Phase.BROKEN)

    def df_dtau_ptr(self) -> DifferentialPointer:
        if self.__df_dtau_ptr is not None:
            if self.__df_dtau_ptr in differentials:
                return self.__df_dtau_ptr
            if FORKING or self.__df_dtau_pid == os.getpid():
                logger.warning(
                    "Could not find cs2 in the cache for %s in process %s. Recreating.",
                    self.name, os.getpid()
                )

        start_time = time.perf_counter()
        # logger.debug("Compiling cs2 for %s in process %s", self.label_unicode, os.getpid())
        ptr = add_df_dtau(f"{self.name}_{id(self)}", self.cs2)
        logger.debug(
            "Compiled cs2 for %s in process %d in %.3f s",
            self.label_unicode, os.getpid(), time.perf_counter() - start_time
        )
        self.__df_dtau_ptr = ptr
        self.__df_dtau_pid = os.getpid()
        return ptr

    def e(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Energy density $e(w,\phi)$. Calls the temperature-based function.

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        """
        return self.e_temp(self.temp(w, phase), phase)

    def enthalpy_ratio(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        r"""Enthalpy ratio $r(T)$

        $$r(T) = \frac{w_s(T)}{w_b(T)}$$
        :param temp: temperature $T$
        """
        return self.w(temp, Phase.SYMMETRIC) / self.w(temp, Phase.BROKEN)

    def inverse_enthalpy_ratio(self, temp: th.FloatOrArr) -> th.FloatOrArr:
        r"""Inverse enthalpy ratio $\Psi(T)$ :ai_2023:`\ `, eq. 19

        $$\Psi(T) = \frac{w_b(T)}{w_s(T)}$$
        :param temp: temperature $T$
        """
        return self.w(temp, Phase.BROKEN) / self.w(temp, Phase.SYMMETRIC)

    def export(self) -> tp.Dict[str, any]:
        return {
            **super().export(),
            "t_ref": self.T_ref,
            "V_s": self.V_s,
            "V_b": self.V_b
        }

    def ge(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Effective degrees of freedom for energy density, $g_{\text{eff},e}(w,\phi)$"""
        temp = self.temp(w, phase)
        return self.ge_temp(temp, phase)

    def gs(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Effective degrees of freedom for entropy, $g_{\text{eff},s}(w,\phi)$"""
        temp = self.temp(w, phase)
        return self.ge_temp(temp, phase)

    def gp(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Effective degrees of freedom for pressure, $g_{\text{eff},p}(w,\phi)$"""
        temp = self.temp(w, phase)
        return self.ge_temp(temp, phase)

    def nu_gdh2024(self, w: th.FloatOrArr, phase: th.FloatOrArr = Phase.BROKEN) -> th.FloatOrArr:
        r"""$$\nu = \frac{1 - 3\omega}{1 + 3\omega}$$
        :giombi_2024_cs:`\ ` eq. 2.11
        """
        omega = self.omega(w, phase)
        return (1 - 3*omega)/(1 + 3*omega)

    def omega(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Barotropic equation of state parameter $\omega$"""
        temp = self.temp(w, phase)
        return self.p_temp(temp, phase) / self.e_temp(temp, phase)

    def p(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pressure $p(w,\phi)$. Calls the temperature-based function.

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        """
        return self.p_temp(self.temp(w, phase), phase)

    def Psi_n(self, wn: th.FloatOrArr) -> th.FloatOrArr:
        r"""Inverse enthalpy ratio at nucleation temperature $\psi_n$, :ai_2023:`\ `, p. 9

        With validation check
        """
        ret = self.inverse_enthalpy_ratio(self.temp(wn, Phase.SYMMETRIC))
        # The LTE violation merely means that entropy is being generated, which is totally normal.
        # min_ret = np.min(ret)
        # if min_ret < 0.9:
        #     logger.warning(
        #         "Psi_n=%s < 0.9. "
        #         "Local thermal equilibrium (LTE) approximations may not be valid, "
        #         "and therefore the model may not allow a constant v_wall to exist "
        #         "without accounting for out-of-equilibrium effects. "
        #         "See Ai et al. (2023) p. 15.",
        #         min_ret
        #     )
        return ret

    def s(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Entropy density $s(w,\phi)$. Calls the temperature-based function.

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        """
        return self.s_temp(self.temp(w, phase), phase)

    def solution_type(
            self,
            v_wall: float,
            alpha_n: float,
            wn: float = None,
            wn_guess: float = None,
            wm_guess: float = None) -> SolutionType:
            if wn is None:
                wn = self.wn(alpha_n, wn_guess)
            v_cj = v_chapman_jouguet(self, alpha_n, wn=wn, wm_guess=wm_guess)

            if transition.is_surely_detonation(v_wall, v_cj):
                return SolutionType.DETON
            if transition.is_surely_sub_def(self, v_wall, wn):
                return SolutionType.SUB_DEF
            if transition.cannot_be_detonation(v_wall, v_cj) and transition.cannot_be_sub_def(self, v_wall, wn):
                return SolutionType.HYBRID
            logger.warning(
                f"Could not determine solution type for %s with v_wall=%s, alpha_n=%s, v_cj=%s",
                self.name, v_wall, alpha_n, v_cj
            )
            return SolutionType.UNKNOWN

    def theta(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Trace anomaly $\theta(w,\phi)$, :notes:`\ `, eq. 7.24

        $$\theta = \frac{1}{4}(e - 3p)$$

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        """
        return 1/4 * (self.e(w, phase) - 3*self.p(w, phase))

    def theta_bar(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pseudotrace $\bar{\theta}$, :giese_2021:`\ `, eq. 9, :ai_2023:`\ `, eq. 19

        $$\bar{\theta} = e - \frac{p}{c_{s,b}^2}$$

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        """
        return self.e(w, phase) - self.p(w, phase) / self.cs2_temp(self.temp(w, phase), Phase.BROKEN)

    def theta_bar_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pseudotrace $\bar{\theta}$, :giese_2021:`\ `, eq. 9, :ai_2023:`\ `, eq. 19

        $$\bar{\theta} = e - \frac{p}{c_{s,b}^2}$$

        :param temp: temperature $T$
        :param phase: phase $\phi$
        """
        return self.e_temp(temp, phase) - self.p_temp(temp, phase) / self.cs2_temp(temp, Phase.BROKEN)

    def tn(
            self,
            alpha_n: th.FloatOrArr,
            wn_guess: float = None,
            theta_bar: bool = False,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        return self.temp(
            self.wn(
                alpha_n, wn_guess=wn_guess, theta_bar=theta_bar,
                error_on_invalid=error_on_invalid,
                nan_on_invalid=nan_on_invalid,
                log_invalid=log_invalid
            ),
            Phase.SYMMETRIC
        )

    def V(self, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Potential $V(\phi)$

        :param phase: phase $\phi$
        """
        return phase*self.V_b + (1 - phase)*self.V_s

    def validate_alpha_n(self, alpha_n: float, allow_invalid: bool = False, log_invalid: bool = True):
        r"""Validate that $\alpha_{n,\text{min}} < \alpha_n < 1$"""
        if alpha_n < 0 or alpha_n < self.alpha_n_min:
            msg = f"Invalid alpha_n={alpha_n}. Minimum for the model: {self.alpha_n_min}"
            if log_invalid:
                logger.error(msg)
            if not allow_invalid:
                raise ValueError(msg)
        elif alpha_n > 1:
            if log_invalid:
                logger.warning("Got alpha_n=%s > 1. Please be careful that it's valid.", alpha_n)

    def vp_vm_tilde_ratio_giese(
            self, vp_tilde: th.FloatOrArr, vm_tilde: th.FloatOrArr, wp: float, wm: th.FloatOrArr) -> th.FloatOrArr:
        r"""Giese approximation for $\frac{\tilde{v}_+}{\tilde{v}_-}$, :giese_2021:`\ ` eq. 11

        $$\frac{\tilde{v}_+}{\tilde{v}_-} \approx \frac{
            (\tilde{v}_+ \tilde{v}_- / c_{s,b}^2 - 1) + 3\alpha_{\bar{\theta}_+} }{
            (\tilde{v}_+ \tilde{v}_- / c_{s,b}^2 - 1) + 3 \tilde{v}_+ \tilde{v}_- \alpha_{\bar{\theta}}_+
        }$$

        :param vp_tilde: $\tilde{v}_+$
        :param vm_tilde: $\tilde{v}_-$
        :param wp: $w_+$
        :param wm: $w_-$
        :return: $\frac{\tilde{v}_+}{\tilde{v}_-}$
        """
        alpha_tbp = self.alpha_theta_bar_plus(wp)
        cs2b = self.cs2(wm, Phase.BROKEN)
        a = vp_tilde*vm_tilde / cs2b - 1
        return (a + 3*alpha_tbp) / (a + 3*vp_tilde*vm_tilde*alpha_tbp)

    def wn_error_msg(self, alpha_n: th.FloatOrArr, param: th.FloatOrArr, param_name: str, info: str = None) -> str:
        if np.isscalar(alpha_n):
            info2 = f"Got: alpha_n={alpha_n}, {param_name}={param}."
        else:
            i = np.argmin(param)
            info2 = f"Most problematic values: alpha_n={alpha_n[i]}, {param_name}={param[i]}"
        if info is not None:
            info2 += f", {info}"
        return \
            f"Got too small alpha_n for the model \"{self.label_unicode}\". {info2}"

    def _wn_scalar(
            self,
            alpha_n: float,
            wn_guess: float,
            theta_bar: bool = False,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> float:
        wn = np.nan
        reason = None
        solution_found = False

        if not solution_found:
            wn_sol = fsolve(self._wn_solvable, x0=np.array([wn_guess]), args=(alpha_n, theta_bar), full_output=True)
            solution_found = wn_sol[2] == 1
            reason = wn_sol[3]
            if solution_found or np.isnan(wn):
                wn = wn_sol[0][0]

        if not solution_found:
            wn_sol = fsolve(self._wn_solvable, x0=np.array([1]), args=(alpha_n, theta_bar), full_output=True)
            solution_found = wn_sol[2] == 1
            # reason = wn_sol[3]
            if solution_found or np.isnan(wn):
                # logger.debug("wn solution was found with wn_guess=1, but not wn_guess=%s", wn_guess)
                wn = wn_sol[0][0]

        if not (solution_found or self.w_crit is None or np.isnan(self.w_crit)):
            try:
                wn_sol = root_scalar(
                    self._wn_solvable, x0=wn_guess, x1=0.99*self.w_crit,
                    args=(alpha_n, theta_bar), bracket=(self.w_min, self.w_crit)
                )
                wn = wn_sol.root
                solution_found = wn_sol.converged
                if np.isclose(wn, 0):
                    solution_found = False
                if solution_found:
                    reason = wn_sol.flag
            except ValueError:
                solution_found = False

        if not solution_found:
            msg = (
                f"wn solution was not found for model={self.name}, "
                f"alpha_n={alpha_n}, wn_guess={wn_guess}, theta_bar={theta_bar}, w_crit={self.w_crit}. " +
                ("" if (error_on_invalid or nan_on_invalid) else f"Using wn={wn}. ") +
                f"Reason: {reason}"
            )
            if log_invalid:
                logger.error(msg)
            if error_on_invalid:
                raise RuntimeError(msg)
            if nan_on_invalid:
                return np.nan

        if wn < 0:
            msg = f"Got wn < 0: wn={wn} for "\
                  f"model={self.name}, alpha_n={alpha_n}, wn_guess={wn_guess}, theta_bar={theta_bar}"
            if log_invalid:
                logger.error(msg)
            if error_on_invalid:
                raise RuntimeError(msg)
            if nan_on_invalid:
                return np.nan

        return wn

    def _wn_solvable(self, wn: th.FloatOrArr, alpha_n: float, theta_bar: bool) -> th.FloatOrArr:
        if not np.isscalar(wn):
            wn = wn[0]
        if theta_bar:
            return self.alpha_theta_bar_n(wn, error_on_invalid=False, nan_on_invalid=True, log_invalid=False) - alpha_n
        return self.alpha_n(wn, error_on_invalid=False, nan_on_invalid=True, log_invalid=False) - alpha_n

    def wn(
            self,
            alpha_n: th.FloatOrArr,
            wn_guess: float = None,
            theta_bar: bool = False,
            error_on_invalid: bool = True,
            nan_on_invalid: bool = True,
            log_invalid: bool = True) -> th.FloatOrArr:
        r"""Enthalpy at nucleation temperature $w_n$ with given $\alpha_n$"""
        invalid_w_crit = self.w_crit is None or np.isnan(self.w_crit) or self.w_crit < 0
        if wn_guess is None or np.isnan(wn_guess) or wn_guess < 0:
            if invalid_w_crit:
                raise ValueError(
                    f"Got invalid wn_guess={wn_guess} "
                    f"and cannot fix it due to invalid w_crit={self.w_crit}")
            wn_guess = 0.9 * self.w_crit
        elif not invalid_w_crit and wn_guess > self.w_crit:
            wn_guess2 = 0.9 * self.w_crit
            logger.warning(
                "Got wn_guess > w_crit, wn_guess=%s, w_crit=%s, fixing with wn_guess=%s",
                wn_guess, self.w_crit, wn_guess2
            )
            wn_guess = wn_guess2

        if np.isscalar(alpha_n):
            return self._wn_scalar(
                alpha_n,
                wn_guess=wn_guess,
                theta_bar=theta_bar,
                error_on_invalid=error_on_invalid,
                nan_on_invalid=nan_on_invalid,
                log_invalid=log_invalid
            )
        ret = np.zeros_like(alpha_n)
        for i in range(alpha_n.size):
            ret[i] = self._wn_scalar(
                alpha_n[i],
                wn_guess=wn_guess,
                theta_bar=theta_bar,
                error_on_invalid=error_on_invalid,
                nan_on_invalid=nan_on_invalid,
                log_invalid=log_invalid
            )
        return ret

    # Abstract methods

    def alpha_n_min_find_params(
            self,
            alpha_n_min_target: float,
            V_s_default: float,
            V_b: float,
            safety_factor_alpha: float = ALPHA_N_MIN_FIND_SAFETY_FACTOR_ALPHA):
        r"""Find the model parameters that allow the given $\alpha_{n,\text{min,target}}$"""
        raise NotImplementedError

    @abc.abstractmethod
    def e_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Energy density $e(T,\phi)$

        $$e \equiv T \frac{\partial p}{\partial T} - p$$
        :giese_2021:`\ `, eq. 2

        :param temp: temperature $T$
        :param phase: phase $\phi$
        """

    @abc.abstractmethod
    def ge_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        pass

    @abc.abstractmethod
    def gp_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        pass

    @abc.abstractmethod
    def gs_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        pass

    @abc.abstractmethod
    def p_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pressure $p(T,\phi)$

        :param temp: temperature $T$
        :param phase: phase $\phi$
        """

    @abc.abstractmethod
    def params_str(self) -> str:
        """Model parameters as a string"""
        pass

    @abc.abstractmethod
    def s_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Entropy density $s(T,\phi)=\frac{dp}{dT}$

        :param temp: temperature $T$
        :param phase: phase $\phi$
        """

    @abc.abstractmethod
    def temp(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Temperature $T(w,\phi)$

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        """

    @abc.abstractmethod
    def w(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Enthalpy $w(T,\phi)$

        :param temp: temperature $T$
        :param phase: phase $\phi$
        """
