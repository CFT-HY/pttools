import abc

import numpy as np
import scipy.optimize

import pttools.type_hints as th
from pttools.bubble.bag import CS2Fun
from pttools.bubble.boundary import Phase


class Model(abc.ABC):
    r"""Template for equations of state

    :param t_ref: reference temperature
    :param t_min: minimum temperature at which the model is valid
    :param V_s: the constant term in the expression of $p$ in the symmetric phase
    :param V_b: the constant term in the expression of $p$ in the broken phase
    :param name: custom name for the model
    """
    BASE_NAME: str

    def __init__(
            self,
            t_ref: float = 1, t_min: float = 0,
            V_s: float = 0, V_b: float = 0,
            name: str = None, gen_cs2: bool = True):
        if t_ref <= t_min:
            raise ValueError("The reference temperature has to be higher than the minimum temperature.")

        # Equal values are allowed so that the default values are accepted.
        if V_b > V_s:
            raise ValueError("The bubble does not expand if V_b >= V_s.")

        self.t_ref: float = t_ref
        self.t_min: float = t_min
        self.V_s: float = V_s
        self.V_b: float = V_b
        self.name = self.BASE_NAME if name is None else name

        #: $$\frac{90}{\pi^2} (V_b - V_s)$$
        self.critical_temp_const: float = 90 / np.pi ** 2 * (self.V_b - self.V_s)

        self.cs2: CS2Fun = self.gen_cs2() if gen_cs2 else None

    def alpha_n(self, wn: th.FloatOrArr) -> th.FloatOrArr:
        r"""Transition strength parameter at nucleation temperature, $\alpha_n$, :notes:`\ `, eq. 7.40.
        $$\alpha_n = \frac{4(\theta(w_n,\phi_s) - \theta(w_n,\phi_b)}{3w_n}$$

        :param wn: enthalpy of the symmetric phase at the nucleation temperature
        """
        theta_diff = self.theta(wn, Phase.SYMMETRIC) - self.theta(wn, Phase.BROKEN)
        if np.any(theta_diff < 0):
            raise ValueError(
                "For a physical equation of state theta_- < theta_+. See p. 33 of Hindmarsh and Hijazi, 2019")
        return 4 * theta_diff / (3 * wn)

    def alpha_plus(self, wp: th.FloatOrArr, wm: th.FloatOrArr):
        r"""Transition strength parameter $\alpha_+$
        $$\alpha_+ = \frac{4\Delta \theta}{3w_+} = \frac{4(\theta(w_+,\phi_s) - \theta(w_-,\phi_b)}{3w_+}$$

        :param wp: $w_+$
        :param wm: $w_-$
        """
        theta_diff = self.theta(wp, Phase.SYMMETRIC) - self.theta(wm, Phase.BROKEN)
        if np.any(theta_diff < 0):
            raise ValueError(
                "For a physical equation of state theta_- < theta_+. See p. 33 of Hindmarsh and Hijazi, 2019")
        return 4 * theta_diff / (3 * wp)

    def critical_temp(self, guess: float) -> float:
        r"""Solves for the critical temperature $T_c$, where $p_s(T_c)=p_b(T_c)$

        :param guess: starting guess for the critical temperature
        """
        # This returns np.float64
        return scipy.optimize.fsolve(
            self.critical_temp_opt,
            guess
            # args=(const),
            # xtol=
            # factor=0.1
        )[0]

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

    @abc.abstractmethod
    def gen_cs2(self) -> CS2Fun:
        r"""This function should generate a Numba-jitted $c_s^2$ function for the model."""

    @abc.abstractmethod
    def e_temp(self, temp: th.FloatOrArr, phase: th.FloatOrArr) -> th.FloatOrArr:
        r"""Energy density $e(T,\phi)$

        $$e \equiv T \frac{\partial p}{\partial T} - p$$
        :giese_2021:`\ `, eq. 2

        :param temp: temperature $T$
        :param phase: phase $\phi$
        """

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
