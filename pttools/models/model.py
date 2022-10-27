import abc
import logging

import numpy as np
from scipy.optimize import fsolve

import pttools.type_hints as th
from pttools.bubble.boundary import Phase
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
            name: str = None,
            label: str = None,
            gen_cs2: bool = True,
            implicit_V: bool = False):

        if implicit_V:
            if V_s != 0 or V_b != 0:
                logger.warning(
                    "Potentials have been specified for the implicit model: %s. "
                    "This is for debugging purposes only. Be careful that the definitions of g and V are consistent.",
                    self.DEFAULT_NAME if name is None else name
                )
        else:
            if V_s < V_b:
                raise ValueError(f"The bubble will not expand, when V_s <= V_b. Got: V_s={V_s}, V_b={V_b}.")
            if V_s == V_b:
                logger.warning("The bubble will not expand, when V_s <= V_b. Got: V_b = V_s = %s.", V_s)

        self.t_ref: float = t_ref
        self.V_s: float = V_s
        self.V_b: float = V_b

        #: $$\frac{90}{\pi^2} (V_b - V_s)$$
        self.critical_temp_const: float = 90 / np.pi ** 2 * (self.V_b - self.V_s)

        super().__init__(name=name, t_min=t_min, t_max=t_max, label=label, gen_cs2=gen_cs2)

        if t_ref <= self.t_min:
            raise ValueError("The reference temperature has to be higher than the minimum temperature.")

    # Concrete methods

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

    def alpha_n(self, wn: th.FloatOrArr, allow_negative: bool = False) -> th.FloatOrArr:
        r"""Transition strength parameter at nucleation temperature, $\alpha_n$, :notes:`\ `, eq. 7.40.
        $$\alpha_n = \frac{4(\theta(w_n,\phi_s) - \theta(w_n,\phi_b)}{3w_n}$$

        :param wn: $w_n$, enthalpy of the symmetric phase at the nucleation temperature
        :param allow_negative: whether to allow unphysical negative output values
        """
        theta_s = self.theta(wn, Phase.SYMMETRIC)
        theta_b = self.theta(wn, Phase.BROKEN)
        theta_diff = theta_s - theta_b
        if np.any(theta_diff < 0):
            if np.isscalar(wn):
                info = f"Got: wn={wn}, theta_s={theta_s}, theta_b={theta_b}, diff: {theta_diff}."
            else:
                i = np.argmin(theta_diff)
                info = f"Most problematic values: wn={wn[i]}, "\
                       f"theta_s={theta_s[i]}, theta_b={theta_b[i]}, diff: {theta_diff[i]}."
            msg = f"For a physical equation of state theta_+ > theta_-. {info} See p. 33 of Hindmarsh and Hijazi, 2019."
            logger.error(msg)
            if not allow_negative:
                raise ValueError(msg)

        return 4 * theta_diff / (3 * wn)

    def alpha_plus(self, wp: th.FloatOrArr, wm: th.FloatOrArr, allow_negative: bool = False) -> th.FloatOrArr:
        r"""Transition strength parameter $\alpha_+$
        $$\alpha_+ = \frac{4\Delta \theta}{3w_+} = \frac{4(\theta(w_+,\phi_s) - \theta(w_-,\phi_b)}{3w_+}$$

        :param wp: $w_+$
        :param wm: $w_-$
        :param allow_negative: whether to allow unphysical negative output values
        """
        theta_s = self.theta(wp, Phase.SYMMETRIC)
        theta_b = self.theta(wm, Phase.BROKEN)
        theta_diff = theta_s - theta_b

        # Error handling
        if np.any(theta_diff < 0):
            if np.isscalar(wp) and np.isscalar(wm):
                info = "Got:"
                i = None
                theta_diff_prob = theta_diff[i]
            else:
                i = np.argmin(theta_diff)
                theta_diff_prob = theta_diff[i]
                info = "Most problematic values:"
            if np.isscalar(wp):
                wp_prob = wp
                theta_s_prob = theta_s
            else:
                wp_prob = wp[i]
                theta_s_prob = theta_s[i]
            if np.isscalar(wm):
                wm_prob = wm
                theta_b_prob = theta_b
            else:
                wm_prob = wm[i]
                theta_b_prob = theta_b[i]

            msg = "For a physical equation of state theta_+ > theta_-. "\
                  f"{info} wp={wp_prob}, wm={wm_prob}, "\
                  f"theta_s={theta_s_prob}, theta_b={theta_b_prob}, theta_diff={theta_diff_prob}. "\
                  "See p. 33 of Hindmarsh and Hijazi, 2019."
            logger.error(msg)
            if not allow_negative:
                raise ValueError(msg)

        return 4 * theta_diff / (3 * wp)

    def critical_temp(self, guess: float) -> float:
        r"""Solves for the critical temperature $T_c$, where $p_s(T_c)=p_b(T_c)$

        :param guess: starting guess for the critical temperature
        """
        # This returns np.float64
        return fsolve(
            self.critical_temp_opt,
            guess
            # args=(const),
            # xtol=
            # factor=0.1
        )[0]

    def critical_temp_opt(self, temp: float) -> float:
        """This function should have its minimum at the critical temperature

        A subclass does not have to implement this if it reimplements :func:`critical_temp`.
        """
        raise NotImplementedError

    def cs2(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Speed of sound squared $c_s^2(w,\phi)$. This must be a Numba-compiled function.

        $$c_s^2 \equiv \left( \frac{\partial p}{\partial e} \right)_s = \frac{dp/dT}{de/dT}$$
        :rel_hydro_book:`\ `, eq. 2.168
        :giese_2021:`\ `, eq. 3

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        """
        raise RuntimeError("The cs2(w, phase) function has not yet been loaded")

    def e(self, w: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Energy density $e(w,\phi)$. Calls the temperature-based function.

        :param w: enthalpy $w$
        :param phase: phase $\phi$
        """
        return self.e_temp(self.temp(w, phase), phase)

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
