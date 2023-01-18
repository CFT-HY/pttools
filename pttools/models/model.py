"""Base class for models"""

import abc
import logging
import typing as tp

import numba
import numpy as np
from scipy.optimize import fminbound, fsolve

import pttools.type_hints as th
from pttools.bubble.boundary import Phase
from pttools.bubble.check import find_most_negative_vals
from pttools.bubble.integrate import add_df_dtau
from pttools.models.base import BaseModel

logger = logging.getLogger(__name__)


class Model(BaseModel, abc.ABC):
    r"""Template for equations of state

    :param t_ref: reference temperature.
        Be careful when using a thermodynamics-based model that there are no conflicts in the choices of units.
    :param t_min: minimum temperature at which the model is valid
    :param V_s: the constant term in the expression of $p$ in the symmetric phase
    :param V_b: the constant term in the expression of $p$ in the broken phase
    :param name: custom name for the model
    :param gen_cs2: used internally for postponing the generation of the cs2 function
    """
    def __init__(
            self,
            V_s: float, V_b: float = 0,
            t_ref: float = 1, t_min: float = None, t_max: float = None,
            t_crit_guess: float = None,
            name: str = None,
            label_latex: str = None,
            label_unicode: str = None,
            gen_critical: bool = True,
            gen_cs2: bool = True,
            implicit_V: bool = False,
            allow_invalid: bool = False):

        if implicit_V:
            if V_s != 0 or V_b != 0:
                logger.warning(
                    "Potentials have been specified for the implicit model: %s. "
                    "This is for debugging purposes only. Be careful that the definitions of g and V are consistent.",
                    self.DEFAULT_NAME if name is None else name
                )
        else:
            if V_s < V_b:
                msg = f"The bubble will not expand, when V_s <= V_b. Got: V_s={V_s}, V_b={V_b}."
                logger.error(msg)
                if not allow_invalid:
                    raise ValueError(msg)
            if V_s == V_b:
                logger.warning("The bubble will not expand, when V_s <= V_b. Got: V_b = V_s = %s.", V_s)

        self.t_ref: float = t_ref
        self.V_s: float = V_s
        self.V_b: float = V_b
        self.__df_dtau_ptr = None

        #: $$\frac{90}{\pi^2} (V_b - V_s)$$
        self.critical_temp_const: float = 90 / np.pi ** 2 * (self.V_b - self.V_s)

        super().__init__(
            t_min=t_min, t_max=t_max,
            name=name, label_latex=label_latex, label_unicode=label_unicode,
            gen_cs2=gen_cs2
        )

        # A model could have t_ref = 1 GeV and be valid only for e.g. > 10 GeV
        # if t_ref < self.t_min:
        #     raise logger.warning(f"T_ref should be higher than T_min. Got: T_ref={t_ref}, T_min={self.t_min}")
        if t_ref >= self.t_max:
            raise ValueError(f"T_ref should be lower than T_max. Got: T_ref={t_ref}, T_max={self.t_max}")

        if gen_critical:
            # TODO: rename wn_max to w_crit
            self.t_crit, self.wn_max, self.alpha_n_min = self.criticals(t_crit_guess, allow_invalid)

    # Concrete methods

    @staticmethod
    def _cs2_limit(
            w_max: float, phase: Phase,
            is_max: bool, cs2_fun: th.CS2Fun, w_min: float = 0,
            allow_fail: bool = False, **kwargs) -> tp.Tuple[float, float]:
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

    @staticmethod
    def check_w_for_alpha(w: th.FloatOrArr, allow_negative: bool = False):
        if w is None or np.any(np.isnan(w)):
            logger.error("Got w=nan for alpha.")
            # Scalar nan cannot be tested for negativity.
            if w is None:
                return
        elif np.any(w < 0):
            if np.isscalar(w):
                info = f"Got negative w={w} for alpha."
            else:
                info = f"Got negative w for alpha. Most problematic value: w={np.min(w)}"
            logger.error(info)
            if not allow_negative:
                raise ValueError(info)

    def alpha_n(self, wn: th.FloatOrArr, allow_negative: bool = False, allow_no_transition: bool = False) \
            -> th.FloatOrArr:
        r"""Transition strength parameter at nucleation temperature, $\alpha_n$, :notes:`\ `, eq. 7.40.
        $$\alpha_n = \frac{4(\theta(w_n,\phi_s) - \theta(w_n,\phi_b)}{3w_n}$$

        :param wn: $w_n$, enthalpy of the symmetric phase at the nucleation temperature
        :param allow_negative: allow unphysical negative output values
        :param allow_no_transition: allow $w_n$ for which there is no phase transition
        """
        # self.check_p(wn, allow_fail=allow_no_transition)
        theta_s = self.theta(wn, Phase.SYMMETRIC)
        theta_b = self.theta(wn, Phase.BROKEN)
        diff = theta_s - theta_b
        prob_diff, prob_theta_s, prob_theta_b = find_most_negative_vals(theta_s, theta_b, diff)
        if prob_diff is not None:
            text = "Got" if np.isscalar(diff) else "Most problematic values"
            msg = \
                f"For a physical equation of state theta_+ > theta_-. {text}: " \
                f"theta_s={prob_theta_s}, theta_b={prob_theta_b}, diff={prob_diff}. " \
                "See p. 33 of Hindmarsh and Hijazi, 2019."
            logger.error(msg)
            if not allow_negative:
                raise ValueError(msg)

        return 4 * diff / (3 * wn)

    def alpha_plus(self, wp: th.FloatOrArr, wm: th.FloatOrArr, allow_negative: bool = False) -> th.FloatOrArr:
        # Todo: This docstring causes the Sphinx error "ERROR: Unknown target name: "w"."
        r"""Transition strength parameter $\alpha_+$
        $$\alpha_+ = \frac{4\Delta \theta}{3w_+} = \frac{4(\theta(w_+,\phi_s) - \theta(w_-,\phi_b)}{3w_+}$$

        :param wp: $w_+$
        :param wm: $w_-$
        :param allow_negative: whether to allow unphysical negative output values
        """
        return 4 * self.delta_theta(wp, wm, allow_negative) / (3 * wp)

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

    def criticals(self, t_crit_guess: float, allow_fail: bool = False, log_info: bool = True):
        t_crit = self.critical_temp(guess=t_crit_guess, allow_fail=allow_fail)
        wn_min = self.w(t_crit, Phase.SYMMETRIC)
        alpha_n_min = self.alpha_n(wn_min)

        logger.info(
            f"Initialized model with name={self.name}, T_crit={t_crit}, alpha_n_min={alpha_n_min}. "
            f"At T_crit: w_s={wn_min}, w_b={self.w(t_crit, Phase.BROKEN)}, "
            f"e_s={self.e_temp(t_crit, Phase.SYMMETRIC)}, e_b={self.e_temp(t_crit, Phase.BROKEN)}, "
            f"p_s={self.p_temp(t_crit, Phase.SYMMETRIC)}, p_b={self.p_temp(t_crit, Phase.BROKEN)}"
        )
        return t_crit, wn_min, alpha_n_min

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
            if np.isfinite(self.t_max):
                guess = np.exp((np.log(self.t_min) + np.log(self.t_max)) / 2)
            else:
                guess = guess_backup

        p_s_min = self.p_temp(self.t_min, Phase.SYMMETRIC)
        p_b_min = self.p_temp(self.t_min, Phase.BROKEN)
        if p_s_min >= p_b_min:
            msg = \
                "All models should have p_s(T=T_min) < p_b(T=T_min) for T_crit to exist. " \
                f"Got: T_min={self.t_min}, p_s={p_s_min}, p_b={p_b_min}."
            logger.error(msg)
            if not allow_fail:
                raise ValueError(msg)

        t_max = self.t_max if np.isfinite(self.t_max) else t_max_backup
        t_arr = np.logspace(np.log10(self.t_min), np.log10(t_max), 10)
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
        if t_crit <= self.t_min:
            msg = f"T_crit should be higher than T_min. Got: T_crit={t_crit}, T_min={self.t_min}"
            logger.error(msg)
            if not allow_fail:
                raise ValueError(msg)
        if t_crit >= self.t_max:
            msg = f"T_max should be lower than T_max. Got: T_crit={t_crit}, T_max={self.t_max}"
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
        raise RuntimeError("The cs2_neg(w, phase) function has not yet been loaded.")

    def cs2_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        return self.cs2(self.w(temp, phase), phase)

    def delta_theta(self, wp: th.FloatOrArr, wm: th.FloatOrArr, allow_negative: bool = False) -> th.FloatOrArr:
        theta_s = self.theta(wp, Phase.SYMMETRIC)
        theta_b = self.theta(wm, Phase.BROKEN)
        diff = theta_s - theta_b
        prob_diff, prob_wp, prob_wm, prob_theta_s, prob_theta_b = find_most_negative_vals(diff, wp, wm, theta_s, theta_b)
        if prob_diff is not None:
            text = "Got" if np.isscalar(diff) else "Most problematic values"
            msg = "For a physical equation of state theta_+ > theta_-. " \
                  f"{text}: wp={prob_wp}, wm={prob_wm}, " \
                  f"theta_s={prob_theta_s}, theta_b={prob_theta_b}, theta_diff={prob_diff}. " \
                  "See p. 33 of Hindmarsh and Hijazi, 2019."
            logger.error(msg)
            if not allow_negative:
                raise ValueError(msg)

        return diff

    def df_dtau_ptr(self) -> int:
        if self.__df_dtau_ptr is not None:
            return self.__df_dtau_ptr

        val = add_df_dtau(f"{self.name}_{id(self)}", self.cs2)
        self.__df_dtau_ptr = val
        return val

    def e(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Energy density $e(w,\phi)$. Calls the temperature-based function.

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        """
        return self.e_temp(self.temp(w, phase), phase)

    def export(self) -> tp.Dict[str, any]:
        return {
            **super().export(),
            "t_ref": self.t_ref,
            "V_s": self.V_s,
            "V_b": self.V_b
        }

    def ge(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Effective degrees of freedom for energy density, $g_{\text{eff},e}(w,\phi)$"""
        temp = self.temp(w, phase)
        return self.ge_temp(temp, phase)

    def gen_cs2_neg(self) -> th.CS2Fun:
        cs2 = self.cs2

        @numba.njit
        def cs2_neg(w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
            return -cs2(w, phase)

        return cs2_neg

    def gs(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Effective degrees of freedom for entropy, $g_{\text{eff},s}(w,\phi)$"""
        temp = self.temp(w, phase)
        return self.ge_temp(temp, phase)

    def gp(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Effective degrees of freedom for pressure, $g_{\text{eff},p}(w,\phi)$"""
        temp = self.temp(w, phase)
        return self.ge_temp(temp, phase)

    def p(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Pressure $p(w,\phi)$. Calls the temperature-based function.

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        """
        return self.p_temp(self.temp(w, phase), phase)

    def s(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Entropy density $s(w,\phi)$. Calls the temperature-based function.

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        """
        return self.s_temp(self.temp(w, phase), phase)

    def theta(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Trace anomaly $\theta(w,\phi)$, :notes:`\ `, eq. 7.24

        $$\theta = \frac{1}{4}(e - 3p)$$

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        """
        return 1/4 * (self.e(w, phase) - 3*self.p(w, phase))

    def V(self, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Potential $V(\phi)$

        :param phase: phase $\phi$
        """
        return phase*self.V_b + (1 - phase)*self.V_s

    def _w_n_scalar(self, alpha_n: float, wn_guess: float) -> float:
        wn_sol = fsolve(self._w_n_solvable, x0=np.array([wn_guess]), args=(alpha_n, ), full_output=True)
        wn: float = wn_sol[0][0]
        if wn_sol[2] != 1:
            logger.error(
                f"w_n solution was not found for model={self.name}, alpha_n={alpha_n}, wn_guess={wn_guess}. "
                f"Using w_n={wn}. "
                f"Reason: {wn_sol[3]}")
        return wn

    def _w_n_solvable(self, param: np.ndarray, alpha_n: float) -> th.FloatOrArr:
        return self.alpha_n(param[0]) - alpha_n

    def w_n(self, alpha_n: th.FloatOrArr, wn_guess: float = 1) -> th.FloatOrArr:
        r"""Enthalpy at nucleation temperature with given $\alpha_n$"""
        # TODO: rename this to wn
        if np.isscalar(alpha_n):
            return self._w_n_scalar(alpha_n, wn_guess)
        ret = np.zeros_like(alpha_n)
        for i in range(alpha_n.size):
            ret[i] = self._w_n_scalar(alpha_n[i], wn_guess)
        return ret

    # Abstract methods

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
